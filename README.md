# 3d-reconstruction

See **[TRINITY.md](TRINITY.md)** for cluster rules (no SLURM, scratch training, path layout) and **[DATA.md](DATA.md)** for WRIVA input vs eval.

Colab/reference notebooks: **[notebooks/](notebooks/)**. Agents never `git commit` — only you do.

## Quick start (Trinity)

```bash
# copy scene Mehar → your NAS
./scripts/copy_wriva_scene.sh <scene>

# on a free compute node (check monitor first):
SCENE=<scene> ./scripts/run_trinity_stage2_lora.sh
```

Stage-2 LoRA package lives under `stage2_lora/`. DreamSim: `eval/dreamsim_score.py`.

Full JHU APL WRIVA scene with MV-SAM3D (every building, GT depth + poses, multi-GPU), compared with
Mehar's single-view SAM3D pipeline: self-contained in **[full_jhuapl_mvsam3d/](full_jhuapl_mvsam3d/README.md)**.
