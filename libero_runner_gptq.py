"""
GPTQ-quantized OpenVLA runner for LIBERO.

Loads pre-quantized checkpoints created by quantize_checkpoints.py:
  - Vision encoder + projector in fp16 (non_llm_weights.pt)
  - LLM backbone as GPTQ 4-bit (llm_quantized/)
  - norm_stats for action unnormalization

~3-5x faster inference than bitsandbytes on-the-fly dequantization.

Usage:
    python libero_runner_gptq.py \
        --checkpoint checkpoints/openvla-7b-finetuned-libero-spatial-gptq-4bit \
        --suite libero_spatial --task_id 0 --n_episodes 5 \
        --device cuda:0 --gpu_id 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# GPTQ OpenVLA loader
# ---------------------------------------------------------------------------

def load_openvla_gptq(checkpoint_path, device="cuda:0"):
    """Load OpenVLA with GPTQ-quantized LLM backbone.

    Returns (model, processor) where the LLM is 4-bit GPTQ and
    vision encoder + projector are fp16.
    """
    sys.path.insert(0, "/mnt/ssd1/devashri/TOM_revision/sptom_icml/openvla")
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from auto_gptq import AutoGPTQForCausalLM
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load metadata
    with open(os.path.join(checkpoint_path, "quantize_config.json")) as f:
        quant_meta = json.load(f)
    original_model_id = quant_meta["original_model_id"]
    print(f"  Original model: {original_model_id}")
    print(f"  Quantization: {quant_meta['bits']}-bit GPTQ, group_size={quant_meta['group_size']}")

    # Step 1: Load processor
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Step 2: Create full model shell in fp16 (on CPU first to save GPU memory)
    print(f"  Loading model shell from {original_model_id}...")
    model = AutoModelForVision2Seq.from_pretrained(
        original_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Step 3: Load GPTQ-quantized LLM
    llm_path = os.path.join(checkpoint_path, "llm_quantized")
    print(f"  Loading GPTQ LLM from {llm_path}...")
    gptq_llm = AutoGPTQForCausalLM.from_quantized(
        llm_path,
        device=device,
        use_safetensors=True,
        trust_remote_code=True,
    )

    # Step 4: Replace the LLM backbone with the quantized version
    # The GPTQ model wraps the actual model inside .model
    model.language_model = gptq_llm.model

    # Step 5: Load non-LLM weights (vision encoder + projector) and move to device
    non_llm_path = os.path.join(checkpoint_path, "non_llm_weights.pt")
    if os.path.exists(non_llm_path):
        print(f"  Loading vision encoder + projector (fp16)...")
        non_llm_state = torch.load(non_llm_path, map_location="cpu")
        # Load only non-LLM weights into the model
        missing, unexpected = model.load_state_dict(non_llm_state, strict=False)
        # Missing keys are expected (they're the LLM weights we replaced)
        print(f"  Loaded {len(non_llm_state)} non-LLM tensors")

    # Step 6: Load norm_stats for action unnormalization
    norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path) as f:
            raw_stats = json.load(f)
        # Convert lists back to numpy arrays
        norm_stats = {}
        for key, val in raw_stats.items():
            norm_stats[key] = {}
            for k, v in val.items():
                if isinstance(v, list):
                    norm_stats[key][k] = np.array(v)
                else:
                    norm_stats[key][k] = v
        model.norm_stats = norm_stats
        print(f"  Loaded norm_stats ({len(norm_stats)} keys)")

    # Move non-quantized parts to device
    # The GPTQ LLM is already on device; move vision + projector
    for name, param in model.named_parameters():
        if not name.startswith("language_model."):
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if not name.startswith("language_model."):
            buf.data = buf.data.to(device)

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Image preprocessing (same as libero_runner.py)
# ---------------------------------------------------------------------------

def preprocess_libero_image(obs, center_crop=True):
    """Extract and preprocess image from LIBERO obs dict."""
    import tensorflow as tf

    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180-degree rotation

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()

    if center_crop:
        img_tf = tf.convert_to_tensor(img)
        img_tf = tf.image.convert_image_dtype(img_tf, tf.float32)
        img_tf = tf.expand_dims(img_tf, axis=0)
        crop_scale = 0.9
        new_size = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
        offset = (1 - new_size) / 2
        bbox = tf.reshape(tf.stack([offset, offset, offset + new_size, offset + new_size]), (1, 4))
        img_tf = tf.image.crop_and_resize(img_tf, bbox, [0], (224, 224))
        img_tf = tf.clip_by_value(img_tf[0], 0, 1)
        img_tf = tf.image.convert_image_dtype(img_tf, tf.uint8, saturate=True)
        img = img_tf.numpy()

    return Image.fromarray(img).convert("RGB")


def get_action(model, processor, image, instruction, unnorm_key, device="cuda:0"):
    """Single-step VLA inference -> 7D action."""
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.float16)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    action[-1] = -action[-1]

    return action, {"mean_entropy": 0.0}


# ---------------------------------------------------------------------------
# LIBERO env helpers
# ---------------------------------------------------------------------------

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
NUM_STEPS_WAIT = 10


def make_libero_env(suite_name, task_id, camera_size=256, gpu_id=0):
    """Create a LIBERO env."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    b = benchmark.get_benchmark(suite_name)()
    task = b.get_task(task_id)
    task_name = b.get_task_names()[task_id]
    init_states = b.get_task_init_states(task_id)
    instruction = task.language

    from libero.libero import get_libero_path
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=camera_size,
        camera_widths=camera_size,
        render_gpu_device_id=gpu_id,
    )
    env.seed(0)
    return env, task_name, init_states, instruction


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, init_state, instruction, model, processor, unnorm_key,
                max_steps=300, center_crop=True, device="cuda:0"):
    """Run one closed-loop episode."""
    env.reset()
    obs = env.set_init_state(init_state)

    trajectory = {
        "instruction": instruction,
        "actions": [],
        "rewards": [],
        "entropies": [],
        "success": False,
        "n_steps": 0,
    }

    total_steps = max_steps + NUM_STEPS_WAIT

    for step in range(total_steps):
        if step < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            continue

        img = preprocess_libero_image(obs, center_crop=center_crop)
        action, signals = get_action(model, processor, img, instruction, unnorm_key, device)

        obs, reward, done, info = env.step(action.tolist())

        trajectory["actions"].append(action.tolist())
        trajectory["rewards"].append(float(reward))
        trajectory["entropies"].append(signals["mean_entropy"])
        trajectory["n_steps"] = step - NUM_STEPS_WAIT + 1

        if done:
            trajectory["success"] = True
            break

    return trajectory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPTQ-quantized OpenVLA on LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to GPTQ checkpoint directory")
    parser.add_argument("--suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--no_center_crop", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results/gptq_4bit")
    args = parser.parse_args()

    if args.max_steps is None:
        args.max_steps = SUITE_MAX_STEPS.get(args.suite, 300)
    if args.unnorm_key is None:
        args.unnorm_key = args.suite
    center_crop = not args.no_center_crop

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading GPTQ OpenVLA from {args.checkpoint} on {args.device}...")
    sys.stdout.flush()
    model, processor = load_openvla_gptq(args.checkpoint, device=args.device)

    # Check unnorm key
    if hasattr(model, 'norm_stats') and model.norm_stats is not None:
        if args.unnorm_key not in model.norm_stats:
            alt_key = f"{args.unnorm_key}_no_noops"
            if alt_key in model.norm_stats:
                args.unnorm_key = alt_key

    print(f"Creating LIBERO env: {args.suite} task {args.task_id}...")
    env, task_name, init_states, instruction = make_libero_env(
        args.suite, args.task_id, gpu_id=args.gpu_id
    )
    print(f"  Task: {task_name}")
    print(f"  Max steps: {args.max_steps} (+{NUM_STEPS_WAIT} stabilization)")
    sys.stdout.flush()

    results = {
        "suite": args.suite,
        "task_name": task_name,
        "instruction": instruction,
        "checkpoint": args.checkpoint,
        "precision": "gptq_4bit",
        "unnorm_key": args.unnorm_key,
        "max_steps": args.max_steps,
        "center_crop": center_crop,
        "episodes": [],
    }

    for ep in range(args.n_episodes):
        init_idx = ep % len(init_states)
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} (init_state {init_idx}) ---")
        sys.stdout.flush()
        t0 = time.time()

        traj = run_episode(
            env, init_states[init_idx], instruction,
            model, processor, args.unnorm_key,
            max_steps=args.max_steps, center_crop=center_crop, device=args.device,
        )
        elapsed = time.time() - t0
        traj["wall_time_s"] = round(elapsed, 1)

        results["episodes"].append(traj)
        status = "SUCCESS" if traj["success"] else "FAIL"
        print(f"  {status} in {traj['n_steps']} steps ({elapsed:.1f}s)")
        sys.stdout.flush()

    successes = sum(1 for e in results["episodes"] if e["success"])
    results["summary"] = {
        "n_episodes": args.n_episodes,
        "successes": successes,
        "success_rate": successes / args.n_episodes,
        "mean_steps": float(np.mean([e["n_steps"] for e in results["episodes"]])),
        "mean_wall_time": float(np.mean([e["wall_time_s"] for e in results["episodes"]])),
    }
    print(f"\n=== Summary: {successes}/{args.n_episodes} success "
          f"({results['summary']['success_rate']:.0%}) ===")

    out_path = os.path.join(args.out_dir, f"{args.suite}_task{args.task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
