# Trinity env + masks

Run on a **GPU compute node** (e.g. `trinity-2-28`), not the head.

## A — Conda (base)

```bash
cd ~/3d-reconstruction
chmod +x scripts/setup_conda_trinity.sh
./scripts/setup_conda_trinity.sh
conda activate mvsam3d
```

That creates `mvsam3d` (Python 3.12), pins **torch 2.5.1+cu121**, clones sibling repos under `~/code/`, installs LoRA + `segment-anything`.

### Still required (same as Colab, long)

The Stage-2 LoRA train script needs the full SAM 3D / MV-SAM3D stack:

- `sam-3d-objects` editable install + Hydra patch  
- Depth-Anything-3  
- flash-attn, kaolin, spconv-cu121 / cumm (match Colab; don’t randomly reinstall)  
- HF access: [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)

```bash
export HF_TOKEN=hf_...          # put in ~/.bashrc if you want
huggingface-cli login --token "$HF_TOKEN"
```

**Shortcut:** if you still have Drive `colab_compiled_pkgs.tar.gz`, rsync it to Trinity and unpack into the env instead of rebuilding.

If your MV-SAM3D remote isn’t the default in the script:

```bash
MVSAM3D_GIT=git@github.com:YOUR/MV-SAM3D.git ./scripts/setup_conda_trinity.sh
```

Point training at:

```bash
export MVSAM3D_ROOT=~/code/MV-SAM3D
```

## B — Masks (WRIVA has none)

Scene (example): `t04_v01_s00_r06_VaryingAltitudes_A01`

Stage-2 LoRA only needs **`micro_n` masks** (default 8). Generate 8 first:

```bash
conda activate mvsam3d
cd ~/3d-reconstruction
SCENE=t04_v01_s00_r06_VaryingAltitudes_A01

# download SAM weights once
mkdir -p /data3/tnishida/ckpts
# vit_b is smaller/faster for a first pass; vit_h is stronger
wget -c -O /data3/tnishida/ckpts/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

python scripts/generate_sam_masks.py \
  --wriva-root /data3/tnishida/WRIVA \
  --scene "$SCENE" \
  --mask-out /data3/tnishida/masks \
  --sam --sam-ckpt /data3/tnishida/ckpts/sam_vit_b_01ec64.pth \
  --sam-type vit_b \
  --max-n 8
```

Check:

```bash
ls /data3/tnishida/masks/$SCENE | head
```

If you already have Drive masks with matching stems:

```bash
python scripts/generate_sam_masks.py \
  --wriva-root /data3/tnishida/WRIVA \
  --scene "$SCENE" \
  --mask-out /data3/tnishida/masks \
  --from-dir /path/to/drive/masks
```

`--full-frame` = debug only (all-ones); don’t use for real recon.

## C — After env + masks

```bash
export HF_TOKEN=hf_...
export REPO_ROOT=~/3d-reconstruction
export MVSAM3D_ROOT=~/code/MV-SAM3D
SCENE=t04_v01_s00_r06_VaryingAltitudes_A01 \
  ./scripts/run_trinity_stage2_lora.sh --train-steps 50 --micro-n 8
```
