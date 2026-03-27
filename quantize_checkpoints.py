"""
Quantize OpenVLA finetuned checkpoints to 4-bit and 8-bit GPTQ.

Strategy: OpenVLA = vision_backbone (DinoSigLIP) + projector + LLM (Llama-2-7b).
We quantize ONLY the LLM backbone with GPTQ (fast CUDA kernels at inference),
and keep the vision encoder + projector in fp16. This is the standard approach
for VLMs — vision encoders are small and sensitive to quantization.

The result is a checkpoint that:
  - Loads in ~4 GB (4-bit) or ~7 GB (8-bit) instead of ~14 GB
  - Runs inference 3-5x faster than bitsandbytes on-the-fly dequantization
  - Can be shared via HuggingFace Hub

Usage:
    # Quantize one checkpoint to 4-bit on GPU 1
    CUDA_VISIBLE_DEVICES=1 python quantize_checkpoints.py \
        --model_id openvla/openvla-7b-finetuned-libero-spatial --bits 4

    # Quantize all 4 finetuned checkpoints (4-bit + 8-bit)
    CUDA_VISIBLE_DEVICES=1 python quantize_checkpoints.py --all --all_bits

    # Quantize + push to HuggingFace Hub
    CUDA_VISIBLE_DEVICES=1 python quantize_checkpoints.py --all --push_to_hub --hf_org devashrinaik
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# All finetuned checkpoints to quantize
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}


def register_openvla():
    """Register custom OpenVLA classes with transformers AutoModel."""
    sys.path.insert(0, "/mnt/ssd1/devashri/TOM_revision/sptom_icml/openvla")
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def create_calibration_data(tokenizer, n_samples=128):
    """Create calibration dataset of tokenized LIBERO-style prompts.

    GPTQ only needs text token IDs to calibrate LLM weight ranges.
    """
    prompts = [
        "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:",
        "In: What action should the robot take to open the top drawer of the cabinet?\nOut:",
        "In: What action should the robot take to put the cream cheese in the bowl?\nOut:",
        "In: What action should the robot take to push the plate to the front of the stove?\nOut:",
        "In: What action should the robot take to turn on the stove?\nOut:",
        "In: What action should the robot take to pick up the red mug and place it on the plate?\nOut:",
        "In: What action should the robot take to close the microwave?\nOut:",
        "In: What action should the robot take to pick up the butter and put it in the bowl?\nOut:",
        "In: What action should the robot take to stack the black bowl on top of the white bowl?\nOut:",
        "In: What action should the robot take to move the ketchup to the left side of the table?\nOut:",
        "In: What action should the robot take to lift the pot lid?\nOut:",
        "In: What action should the robot take to slide the mug under the coffee machine?\nOut:",
        "In: What action should the robot take to grasp the spatula and flip the object?\nOut:",
        "In: What action should the robot take to open the bottom drawer?\nOut:",
        "In: What action should the robot take to pick up the wine glass from the rack?\nOut:",
        "In: What action should the robot take to place the bowl on the stove?\nOut:",
    ]

    calibration_data = []
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        tokens = tokenizer(prompt, return_tensors="pt")
        calibration_data.append(tokens.input_ids.squeeze(0))

    return calibration_data


def get_dir_size_gb(path):
    """Get total size of a directory in GB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9


def quantize_model(model_id, bits, out_dir, n_calibration=128, num_workers=4,
                   push_to_hub=False, hf_org=None):
    """Quantize a single OpenVLA checkpoint.

    Approach:
    1. Load full model in fp16
    2. Extract the Llama LLM backbone
    3. Quantize LLM with auto-gptq
    4. Save: quantized LLM weights + original vision/projector weights + config
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor, LlamaTokenizer, AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    print(f"\n{'='*70}")
    print(f"Quantizing: {model_id} -> {bits}-bit GPTQ")
    print(f"{'='*70}")
    sys.stdout.flush()

    model_short = model_id.split("/")[-1]
    quant_name = f"{model_short}-gptq-{bits}bit"
    save_path = os.path.join(out_dir, quant_name)

    if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "quantize_config.json")):
        print(f"  Already exists at {save_path}, skipping.")
        return save_path

    os.makedirs(save_path, exist_ok=True)
    t0 = time.time()

    # Step 1: Load full OpenVLA model
    print(f"  [1/5] Loading full model in fp16...")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Find the LLM backbone
    llm = None
    for attr in ['language_model', 'lm_backbone']:
        if hasattr(model, attr):
            llm = getattr(model, attr)
            llm_attr = attr
            break

    if llm is None:
        raise RuntimeError(
            f"Could not find LLM backbone. Model children: {[n for n, _ in model.named_children()]}"
        )

    print(f"  Found LLM: {llm_attr} ({type(llm).__name__})")
    llm_param_count = sum(p.numel() for p in llm.parameters()) / 1e9
    total_param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  LLM params: {llm_param_count:.2f}B / {total_param_count:.2f}B total")

    # Step 2: Save non-LLM components (vision encoder, projector, embeddings)
    print(f"  [2/5] Saving vision encoder + projector (fp16)...")
    sys.stdout.flush()

    # Save the full model state dict, then we'll replace LLM weights with quantized ones
    # First, save processor and config
    processor.save_pretrained(save_path)
    model.config.save_pretrained(save_path)

    # Save non-LLM weights separately
    non_llm_state = {}
    llm_prefix = f"{llm_attr}."
    for name, param in model.state_dict().items():
        if not name.startswith(llm_prefix):
            non_llm_state[name] = param
    torch.save(non_llm_state, os.path.join(save_path, "non_llm_weights.pt"))
    print(f"  Non-LLM weights: {len(non_llm_state)} tensors")

    # Also save norm_stats if present (needed for action unnormalization)
    if hasattr(model, 'norm_stats') and model.norm_stats is not None:
        with open(os.path.join(save_path, "norm_stats.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            stats = {}
            for key, val in model.norm_stats.items():
                stats[key] = {}
                for k, v in val.items():
                    if hasattr(v, 'tolist'):
                        stats[key][k] = v.tolist()
                    else:
                        stats[key][k] = v
            json.dump(stats, f)
        print(f"  Saved norm_stats ({len(model.norm_stats)} keys)")

    # Step 3: Extract and save LLM for GPTQ quantization
    print(f"  [3/5] Preparing LLM for GPTQ quantization...")
    sys.stdout.flush()

    # Save LLM temporarily so auto-gptq can load it
    llm_temp_path = os.path.join(save_path, "_llm_temp")
    os.makedirs(llm_temp_path, exist_ok=True)
    llm.save_pretrained(llm_temp_path)

    # Get tokenizer for calibration
    # OpenVLA uses Llama tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(llm_temp_path, trust_remote_code=True)
    tokenizer.save_pretrained(llm_temp_path)

    # Free memory before quantization
    del model, llm, non_llm_state
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: GPTQ quantization
    print(f"  [4/5] Running GPTQ quantization ({bits}-bit, group_size=128)...")
    print(f"         Calibrating with {n_calibration} samples...")
    sys.stdout.flush()

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
        damp_percent=0.1,
    )

    # Load LLM with auto-gptq
    gptq_model = AutoGPTQForCausalLM.from_pretrained(
        llm_temp_path,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
    )

    # Create calibration data
    calib_data = create_calibration_data(tokenizer, n_samples=n_calibration)
    # auto-gptq expects list of dicts with input_ids + attention_mask
    calib_dataset = [
        {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones_like(ids.unsqueeze(0))}
        for ids in calib_data
    ]

    # Quantize
    quant_t0 = time.time()
    gptq_model.quantize(calib_dataset, batch_size=1)
    quant_elapsed = time.time() - quant_t0
    print(f"         Quantization took {quant_elapsed/60:.1f} minutes")

    # Save quantized LLM
    llm_quant_path = os.path.join(save_path, "llm_quantized")
    os.makedirs(llm_quant_path, exist_ok=True)
    gptq_model.save_quantized(llm_quant_path)
    tokenizer.save_pretrained(llm_quant_path)
    print(f"  Quantized LLM saved to {llm_quant_path}")

    # Cleanup temp
    del gptq_model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(llm_temp_path, ignore_errors=True)

    # Step 5: Write metadata
    print(f"  [5/5] Writing metadata...")
    metadata = {
        "original_model_id": model_id,
        "quantization_method": "gptq",
        "bits": bits,
        "group_size": 128,
        "desc_act": False,
        "llm_type": "LlamaForCausalLM",
        "vision_precision": "float16",
        "n_calibration_samples": n_calibration,
        "quantization_time_minutes": round(quant_elapsed / 60, 1),
    }
    with open(os.path.join(save_path, "quantize_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    size_gb = get_dir_size_gb(save_path)
    print(f"\n  Done in {elapsed/60:.1f} minutes")
    print(f"  Checkpoint size: {size_gb:.2f} GB")
    print(f"  Saved to: {save_path}")

    # Push to HuggingFace Hub
    if push_to_hub and hf_org:
        hub_name = f"{hf_org}/{quant_name}"
        print(f"\n  Pushing to HuggingFace Hub: {hub_name}...")
        sys.stdout.flush()
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(hub_name, exist_ok=True)
            api.upload_folder(
                folder_path=save_path,
                repo_id=hub_name,
                commit_message=f"Add {bits}-bit GPTQ quantized {model_short}",
                num_workers=num_workers,
            )
            print(f"  Uploaded to https://huggingface.co/{hub_name}")
        except Exception as e:
            print(f"  Upload failed: {e}")
            print(f"  You can manually push later:")
            print(f"    huggingface-cli upload {hub_name} {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Quantize OpenVLA checkpoints with GPTQ")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Single model to quantize (HF model ID)")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits (default: 4)")
    parser.add_argument("--all", action="store_true",
                        help="Quantize all 4 finetuned LIBERO checkpoints")
    parser.add_argument("--all_bits", action="store_true",
                        help="Quantize in both 4-bit and 8-bit")
    parser.add_argument("--out_dir", type=str, default="checkpoints",
                        help="Output directory for quantized checkpoints")
    parser.add_argument("--n_calibration", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for upload")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push quantized checkpoints to HuggingFace Hub")
    parser.add_argument("--hf_org", type=str, default="devashrinaik",
                        help="HuggingFace org/user for upload")
    args = parser.parse_args()

    register_openvla()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        models = list(CHECKPOINTS.values())
    elif args.model_id:
        models = [args.model_id]
    else:
        parser.error("Specify --model_id or --all")

    bits_list = [4, 8] if args.all_bits else [args.bits]

    print(f"Quantization plan:")
    print(f"  Models: {len(models)} — {[m.split('/')[-1] for m in models]}")
    print(f"  Bits: {bits_list}")
    print(f"  Output: {os.path.abspath(args.out_dir)}")
    print(f"  Calibration samples: {args.n_calibration}")
    print(f"  Push to hub: {args.push_to_hub}")
    if args.push_to_hub:
        print(f"  HF org: {args.hf_org}")
    print()
    sys.stdout.flush()

    results = {}
    for model_id in models:
        for bits in bits_list:
            key = f"{model_id.split('/')[-1]}_{bits}bit"
            try:
                path = quantize_model(
                    model_id=model_id,
                    bits=bits,
                    out_dir=args.out_dir,
                    n_calibration=args.n_calibration,
                    num_workers=args.num_workers,
                    push_to_hub=args.push_to_hub,
                    hf_org=args.hf_org,
                )
                results[key] = {"status": "success", "path": path}
            except Exception as e:
                print(f"\n  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results[key] = {"status": "failed", "error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("Quantization Summary")
    print(f"{'='*70}")
    for key, res in results.items():
        icon = "OK" if res["status"] == "success" else "FAIL"
        print(f"  [{icon}] {key}")
        if res["status"] == "success":
            size = get_dir_size_gb(res["path"])
            print(f"         {res['path']} ({size:.2f} GB)")
        else:
            print(f"         Error: {res.get('error', 'unknown')}")
    print()


if __name__ == "__main__":
    main()
