# 3d-reconstruction

See **[TRINITY.md](TRINITY.md)** for cluster rules (no SLURM, scratch training, path layout) and **[DATA.md](DATA.md)** for WRIVA input vs eval.

## Quick start (Trinity)

```bash
# copy scene Mehar → your NAS
./scripts/copy_wriva_scene.sh <scene>

# on a free compute node (check monitor first):
SCENE=<scene> ./scripts/run_trinity_stage2_lora.sh
```

Stage-2 LoRA package lives under `stage2_lora/`. DreamSim: `eval/dreamsim_score.py`.
