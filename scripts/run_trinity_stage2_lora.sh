#!/usr/bin/env bash
# Trinity job launcher (NO SLURM — lab self-schedules; see cluster doc).
#
# Workflow from the FALL 2026 Trinity doc:
#   1) Check monitor: http://trinity.vision.cs.cmu.edu:53240/  (VPN or SSH tunnel)
#   2) ssh to a free compute node (not the head node)
#   3) Copy working data to local /scratch or /ssd*  (train from local, not /data*)
#   4) Run this script on that node
#
# Do NOT leave continuous checkpoint spam on networked /data*.
# Leave 100–200GB free on /scratch. Unused /scratch files may be deleted after 7 days.

set -euo pipefail

SCS_ID="${SCS_ID:-tnishida}"
REPO_ROOT="${REPO_ROOT:-/home/${SCS_ID}/GSV-MSV/3d-reconstruction}"
# Long-term copy of WRIVA (your NAS quota — NOT Mehar's tree)
DATA_NAS="${DATA_NAS:-/data3/${SCS_ID}/WRIVA}"
# Local working copy on THIS compute node
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${SCS_ID}}"
SCENE="${SCENE:?set SCENE=...}"
MVSAM3D_ROOT="${MVSAM3D_ROOT:-/home/${SCS_ID}/code/MV-SAM3D}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/stage2_lora_wriva.yaml}"
CONDA_ENV="${CONDA_ENV:-mvsam3d}"

# Refuse running heavy work on head node (hostname usually just "trinity")
HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOST" == "trinity" ]] || [[ "$HOST" != trinity-* ]]; then
  echo "ERROR: run on a compute node (trinity-*), not the head node (host=$HOST)."
  echo "Check monitor, then: ssh trinity-X-Y"
  exit 1
fi

mkdir -p "$SCRATCH_ROOT/WRIVA" "$SCRATCH_ROOT/masks" "$SCRATCH_ROOT/outputs/stage2_lora"

# Sync scene from your NAS copy → local scratch (not from /data2/mehark)
if [[ ! -d "$DATA_NAS/$SCENE" ]]; then
  echo "ERROR: missing NAS scene $DATA_NAS/$SCENE"
  echo "First copy from Mehar once:"
  echo "  mkdir -p $DATA_NAS && rsync -a /data2/mehark/WRIVA/$SCENE/ $DATA_NAS/$SCENE/"
  exit 1
fi

echo "Syncing $SCENE → $SCRATCH_ROOT/WRIVA/$SCENE ..."
rsync -a --delete "$DATA_NAS/$SCENE/" "$SCRATCH_ROOT/WRIVA/$SCENE/"

# Optional: masks already on NAS
if [[ -d "/data3/${SCS_ID}/masks/$SCENE" ]]; then
  rsync -a "/data3/${SCS_ID}/masks/$SCENE/" "$SCRATCH_ROOT/masks/$SCENE/"
fi

cd "$REPO_ROOT"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

module load cuda 2>/dev/null || true

export HF_TOKEN="FAKESECRET_g2h3i4j5k6l7m8n9o0p1"
export HF_HOME="${HF_HOME:-/home/${SCS_ID}/.cache/huggingface}"
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
# Never write under Mehar
export STAGE2_READONLY_ROOTS="/data2/mehark:/data/mehark:/data3/mehark"

echo "host=$HOST  gpus=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi -L || true

python scripts/train_stage2_lora.py \
  --config "$CONFIG" \
  --layout wriva \
  --data-root "$SCRATCH_ROOT/WRIVA" \
  --scene "$SCENE" \
  --mask-root "$SCRATCH_ROOT/masks" \
  --mvsam3d-root "$MVSAM3D_ROOT" \
  --output-dir "$SCRATCH_ROOT/outputs/stage2_lora" \
  "$@"

echo "Training finished. Sync outputs back to NAS when done:"
echo "  rsync -a $SCRATCH_ROOT/outputs/stage2_lora/ /data3/${SCS_ID}/outputs/stage2_lora/"
