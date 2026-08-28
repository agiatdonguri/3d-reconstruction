# Trinity (Deva Lab) — how we use this cluster

Source of truth: `TRINITY Cluster Doc - FALL 2026.pdf`.

## Critical facts (do not fight these)

| Fact | Implication for us |
|------|--------------------|
| **No SLURM** | Self-schedule via monitor; old `submit_*.slurm` is **not** used |
| Head node = login only | Never train / heavy I/O on `trinity` head |
| Train from **local** `/scratch` or `/ssd*` | Copy scene to node local disk, then train |
| NAS `/data*` `/data2` `/data3` = long-term | Slow if used as training I/O; kills cluster for everyone |
| Mehar WRIVA at `/data2/mehark/WRIVA` | **Copy once** → `/data3/tnishida/WRIVA`, never write under mehark |
| Quotas | home 250GB; `/data3|/data|/data2`/$USER ≈ 4TB each |
| `/scratch` purge | Unused files deleted after **7 days**; leave 100–200GB free |
| Fair use | ≤2 nodes / 16 GPUs normal; more → `vnice` |
| Admins | Jay Karhade; **Mehar Khurana** |

Monitor: http://trinity.vision.cs.cmu.edu:53240/ (campus/VPN or SSH tunnel).

```bash
ssh -L 127.0.0.1:53240:localhost:53240 tnishida@trinity.vision.cs.cmu.edu
# then open http://localhost:53240/
```

## Path layout for this project

```
/home/tnishida/                  # code, conda, small software
/data3/tnishida/WRIVA/<scene>/   # long-term scene copy (from Mehar)
/data3/tnishida/masks/<scene>/   # SAM masks (sidecar)
/data3/tnishida/outputs/         # durable results (rsync from scratch)
/scratch/tnishida/…              # per-node working copy WHILE training
```

## Commands

```bash
# 1) one-time NAS copy (from head or any node)
./scripts/copy_wriva_scene.sh t01_v01_s06_r08_ImageDensity_A01_iPad_LTS

# 2) pick a free GPU node on the monitor, then
ssh trinity-X-Y
cd ~/GSV-MSV/3d-reconstruction   # or wherever you keep code
SCENE=... ./scripts/run_trinity_stage2_lora.sh --train-steps 200

# 3) after job: sync outputs NAS ← scratch (script prints the rsync)
```

## Cluster hygiene we follow

- No continuous logging/checkpoints to `/data*` — write under `/scratch`, sync once.
- Don’t open VS Code/Cursor on entire `/home` with huge trees if avoidable.
- Shut down idle Jupyter/remote sessions.
- Prefer 3090 / A6000 / A100-class for Stage-2 (VRAM); don’t hog 8-GPU nodes for 1-GPU jobs.
- Prefer tarball/rsync for many small files (WRIVA images).

## Eval

DreamSim after you can render: `python -m eval.dreamsim_score …`  
Mehar `model/` + `gsplat/` on your **copy** = baselines only.
