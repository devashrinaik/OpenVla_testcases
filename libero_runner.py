"""
Minimal closed-loop runner: frozen OpenVLA in LIBERO.

Matches the official OpenVLA eval pipeline:
  - 180-degree image rotation
  - JPEG encode-decode + Lanczos3 resize (matches RLDS training preprocessing)
  - Center crop (for models trained with image augmentation)
  - Gripper normalization [0,1] -> [-1,+1] + sign inversion
  - 10-step stabilization wait at episode start

Usage:
    python libero_runner.py --suite libero_spatial --task_id 0 --n_episodes 5
    python libero_runner.py --suite libero_10 --task_id 0 --n_episodes 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# OpenVLA loader
# ---------------------------------------------------------------------------

def load_openvla(model_id="openvla/openvla-7b", device="cuda:0"):
    """Load frozen OpenVLA. Returns (model, processor)."""
    sys.path.insert(0, "/mnt/ssd1/devashri/TOM_revision/sptom_icml/openvla")
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Image preprocessing (matches official OpenVLA LIBERO eval exactly)
# ---------------------------------------------------------------------------

def preprocess_libero_image(obs, center_crop=True):
    """Extract and preprocess image from LIBERO obs dict.

    Steps (matching official eval):
      1. 180-degree rotation (LIBERO renders upside-down)
      2. JPEG encode-decode (matches RLDS dataloader used during training)
      3. Lanczos3 resize to 224x224
      4. Optional center crop (for models trained with random crop augmentation)
    """
    import tensorflow as tf

    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180-degree rotation

    # JPEG encode-decode + Lanczos3 resize (matches Octo/OpenVLA RLDS preprocessing)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()

    if center_crop:
        # Center crop to 90% area, then resize back (matches training augmentation)
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


def get_action(model, processor, image: Image.Image, instruction: str,
               unnorm_key: str, device: str = "cuda:0"):
    """Single-step VLA inference -> 7D action.

    Uses predict_action() (works with sdpa attention) for correct unnormalization,
    then applies gripper normalization and sign inversion to match LIBERO convention.
    """
    # Official OpenVLA prompt format
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # Gripper normalization: [0,1] -> [-1,+1], binarized
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])

    # Invert gripper sign: OpenVLA training uses 0=close,1=open
    # but LIBERO env expects -1=open, +1=close
    action[-1] = -action[-1]

    return action, {
        "mean_entropy": 0.0,  # TODO: add entropy tracking via generate() if needed
    }


# ---------------------------------------------------------------------------
# LIBERO env helpers
# ---------------------------------------------------------------------------

# Max steps per suite (from official eval, based on longest training demo)
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

NUM_STEPS_WAIT = 10  # stabilization wait


def make_libero_env(suite_name: str, task_id: int, camera_size: int = 256, gpu_id: int = 0):
    """Create a LIBERO env. Returns (env, task_name, init_states, language_instruction)."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    b = benchmark.get_benchmark(suite_name)()
    task = b.get_task(task_id)
    task_name = b.get_task_names()[task_id]
    init_states = b.get_task_init_states(task_id)

    # Use the task's own language description (official way)
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
    """Run one closed-loop episode. Returns dict with trajectory data."""
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
        # Stabilization: do nothing for first few steps (objects settling)
        if step < NUM_STEPS_WAIT:
            dummy_action = [0, 0, 0, 0, 0, 0, -1]  # gripper open
            obs, reward, done, info = env.step(dummy_action)
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
    parser = argparse.ArgumentParser(description="Frozen OpenVLA baseline on LIBERO")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps (default: suite-specific from official eval)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU for rendering")
    parser.add_argument("--model_id", type=str, default="openvla/openvla-7b",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="Action unnormalization key (default: same as suite name)")
    parser.add_argument("--no_center_crop", action="store_true",
                        help="Disable center cropping")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    # Defaults
    if args.max_steps is None:
        args.max_steps = SUITE_MAX_STEPS.get(args.suite, 300)
    if args.unnorm_key is None:
        args.unnorm_key = args.suite
    center_crop = not args.no_center_crop

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading OpenVLA ({args.model_id}) on {args.device}...")
    model, processor = load_openvla(model_id=args.model_id, device=args.device)

    # Check unnorm key exists
    if hasattr(model, 'norm_stats') and model.norm_stats is not None:
        if args.unnorm_key not in model.norm_stats:
            # Try with _no_noops suffix
            alt_key = f"{args.unnorm_key}_no_noops"
            if alt_key in model.norm_stats:
                args.unnorm_key = alt_key
                print(f"  Using unnorm_key: {alt_key}")
            else:
                print(f"  WARNING: unnorm_key '{args.unnorm_key}' not in model.norm_stats!")
                print(f"  Available keys: {list(model.norm_stats.keys())}")

    print(f"Creating LIBERO env: {args.suite} task {args.task_id}...")
    env, task_name, init_states, instruction = make_libero_env(
        args.suite, args.task_id, gpu_id=args.gpu_id
    )
    print(f"  Task: {task_name}")
    print(f"  Instruction: {instruction}")
    print(f"  Max steps: {args.max_steps} (+{NUM_STEPS_WAIT} stabilization)")
    print(f"  Center crop: {center_crop}")
    print(f"  Init states available: {len(init_states)}")

    results = {
        "suite": args.suite,
        "task_name": task_name,
        "instruction": instruction,
        "model_id": args.model_id,
        "unnorm_key": args.unnorm_key,
        "max_steps": args.max_steps,
        "center_crop": center_crop,
        "episodes": [],
    }

    for ep in range(args.n_episodes):
        init_idx = ep % len(init_states)
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} (init_state {init_idx}) ---")
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

    # Summary
    successes = sum(1 for e in results["episodes"] if e["success"])
    results["summary"] = {
        "n_episodes": args.n_episodes,
        "successes": successes,
        "success_rate": successes / args.n_episodes,
        "mean_steps": np.mean([e["n_steps"] for e in results["episodes"]]),
        "mean_entropy": np.mean([np.mean(e["entropies"]) for e in results["episodes"] if e["entropies"]]),
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
