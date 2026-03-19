# OpenVLA Memory Benchmark

Evaluation pipeline for measuring where frozen OpenVLA fails on long-horizon tasks in LIBERO — establishing the baseline that external memory modules need to beat.

## What This Does

Runs finetuned OpenVLA (7B, frozen) across four LIBERO task suites of increasing horizon length:

| Suite | Horizon | Tasks | What It Tests |
|-------|---------|-------|---------------|
| `libero_spatial` | Short (10-20 steps) | 10 | Single pick-place, varying spatial positions |
| `libero_object` | Medium (20-50 steps) | 10 | Different objects, same placement target |
| `libero_goal` | Medium-long (30-80 steps) | 10 | Open drawers, place inside, turn on stove |
| `libero_10` | Long (50-100+ steps) | 10 | Chained multi-step tasks |

Each suite uses its own official finetuned checkpoint from [OpenVLA](https://github.com/openvla/openvla). The output is a **degradation curve** showing task success rate dropping as horizon increases.

## Setup

```bash
# Create conda environment
conda create -n vla_bench python=3.10 -y
conda activate vla_bench

# Install dependencies
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 timm==0.9.10 accelerate
pip install libero

# Install OpenVLA (from source)
pip install git+https://github.com/openvla/openvla.git

# LIBERO config (run once — creates ~/.libero/config.yaml on first import)
python -c "from libero.libero import benchmark"
```

**Note**: The `openvla` path in `libero_runner.py` (line 33) points to a local clone. Update this to match your installation.

## Usage

```bash
# Run one suite
python run_all.py --suites libero_spatial --n_episodes 10

# Run all suites (4 suites x 10 tasks x 10 episodes = 400 episodes, ~4-6 hours)
python run_all.py --n_episodes 10

# Run a single task for quick testing
python libero_runner.py --suite libero_spatial --task_id 0 --n_episodes 5 \
    --model_id openvla/openvla-7b-finetuned-libero-spatial

# Generate plots from results
python plot_results.py
```

Finetuned checkpoints (~15GB each) download automatically from HuggingFace on first run.

## Files

| File | Purpose |
|------|---------|
| `libero_runner.py` | Single-task closed-loop eval. Handles image preprocessing (180-deg rotation, JPEG encode-decode, Lanczos3 resize, center crop), action decoding, gripper normalization. |
| `run_all.py` | Batch runner across suites. Auto-selects the correct finetuned checkpoint per suite. |
| `plot_results.py` | Generates degradation curve (TSR vs horizon), per-task breakdown, and entropy trace plots. |

## Key Technical Details

- **Attention**: Uses `sdpa` (PyTorch scaled dot-product attention). The `eager` implementation has a causal mask bug with transformers 4.40.1 + OpenVLA's custom model code. `flash_attention_2` works but requires CUDA toolkit for compilation.
- **Image preprocessing**: LIBERO renders images 180-degrees rotated. The pipeline applies rotation, JPEG encode-decode (matching RLDS training), Lanczos3 resize to 224x224, and center crop (90% area) to match training augmentation.
- **Gripper**: OpenVLA outputs gripper in [0,1] (0=close, 1=open). Pipeline normalizes to [-1,+1] and inverts sign for LIBERO convention (-1=open, +1=close).
- **Stabilization**: 10 no-op steps at episode start for objects to settle in simulation.

## Output

Results are saved to `results/` (gitignored):
- `results/{suite}_task{id}.json` — per-task episode data
- `results/aggregate.json` — summary across all suites
- `results/degradation_curve.png` — the main plot
- `results/per_task_breakdown.png` — per-task TSR within each suite
