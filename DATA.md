# Data roles (WRIVA / ULTRRA on Trinity)

See **[TRINITY.md](TRINITY.md)** for cluster mechanics (no SLURM, scratch, quotas).

## Workflow

1. Copy scene Mehar → **your NAS**: `./scripts/copy_wriva_scene.sh <scene>` → `/data3/tnishida/WRIVA/`
2. On a **compute node**, launcher syncs NAS → `/scratch/tnishida/` and trains from scratch.
3. Never write under `/data2/mehark` (enforced in `stage2_lora/safety.py`).
4. Sync outputs scratch → `/data3/tnishida/outputs/` once when done (not every step).

## Reconstruct vs evaluate

| Path under `…/WRIVA/<scene>/` | Role |
|---|---|
| `sparse/undistorted_mitigated_images/0/` | **INPUT** — RGB |
| `sparse/images.txt` + `cameras.txt` (+ `sparse_nvm/sparse.nvm`) | **INPUT** — cameras |
| `sparse/metric_z_depth_mve/` + `depth_scale.json` | **INPUT (optional)** — depth |
| `model/*_acc_*`, `sparse/gsplat/` | **EVAL ONLY** — Mehar baselines |
| `metrics/` | **EVAL** — refs / artifacts |

Masks: not in WRIVA → `/data3/tnishida/masks/<scene>/` (auto-SAM or Drive copy).

Eval: **DreamSim** on your renders vs held-out/ref RGB.
