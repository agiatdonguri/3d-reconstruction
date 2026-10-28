#!/usr/bin/env bash
# Trinity launcher: full JHU APL WRIVA scene with MV-SAM3D (no SLURM: pick a free node on the monitor,
# ssh to it, run inside tmux). See README.md.
#
#   SCENE=<scene> ./run_trinity.sh                                   # every stage, SAM 3 masks
#   SCENE=<scene> STAGES=inspect ./run_trinity.sh                    # only check the data (minutes, CPU)
#   SCENE=<scene> STAGES=import_masks MASKS_IMPORT_DIR=/data3/tnishida/… ./run_trinity.sh   # Mehar's masks
#   SCENE=<scene> STAGES=mvsam3d,place,assemble EXTRA_SETS="max_objects=3" ./run_trinity.sh  # smoke test
#
# Reads the scene from YOUR NAS copy (/data3/$SCS_ID/WRIVA/<scene>, see copy_from_mehar.sh), works on this
# node's /scratch, and syncs the run folder back to /data3/$SCS_ID/outputs/full_jhuapl_mvsam3d after each stage.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCS_ID="${SCS_ID:-tnishida}"
SCENE="${SCENE:?set SCENE=<WRIVA scene folder name>}"
CONFIG="${CONFIG:-$HERE/configs/jhuapl_full.yaml}"
DATA_NAS="${DATA_NAS:-/data3/${SCS_ID}/WRIVA}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${SCS_ID}}"
OUT_NAS="${OUT_NAS:-/data3/${SCS_ID}/outputs/full_jhuapl_mvsam3d}"
RUN_NAME="${RUN_NAME:-$SCENE}"
STAGES="${STAGES:-inspect,prepare,masks,associate,mvsam3d,place,assemble}"
CONDA_ENV="${CONDA_ENV:-mvsam3d}"          # SAM 3D Objects / MV-SAM3D stack (../scripts/setup_conda_trinity.sh)
CONDA_ENV_SAM3="${CONDA_ENV_SAM3:-sam3}"   # SAM 3 needs a newer torch (./setup_sam3_env.sh)
MVSAM3D_ROOT="${MVSAM3D_ROOT:-/home/${SCS_ID}/code/MV-SAM3D}"
MASKS_IMPORT_DIR="${MASKS_IMPORT_DIR:-}"
GPUS="${GPUS:-auto}"                       # auto = GPUs that are idle right now; or e.g. GPUS=0,1,2,3
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"       # "idle" = less than this much memory in use
EXTRA_SETS="${EXTRA_SETS:-}"               # e.g. "max_objects=3 image_glob=*drone*"

HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOST" != trinity-* ]]; then
  echo "ERROR: run on a compute node (trinity-*), not $HOST. Check the monitor, then ssh trinity-X-Y."
  exit 1
fi
case "$DATA_NAS $SCRATCH_ROOT $OUT_NAS $MASKS_IMPORT_DIR" in *mehark*) echo "ERROR: copy Mehar's data to /data3/$SCS_ID first (copy_from_mehar.sh); runs never touch mehark paths"; exit 1 ;; esac
if [[ ! -d "$DATA_NAS/$SCENE" ]]; then
  echo "ERROR: $DATA_NAS/$SCENE missing. First: SCENE=$SCENE $HERE/copy_from_mehar.sh"
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Hugging Face downloads (SAM 3, SAM 3D) fail unless already cached."
fi

if [[ "$GPUS" == auto ]]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', *' -v lim="$GPU_FREE_MIB" '$2 < lim {printf "%s%s", sep, $1; sep=","}')"
fi
if [[ -z "$GPUS" ]]; then
  echo "ERROR: no idle GPU on $HOST (all use ≥ $GPU_FREE_MIB MiB). Pick another node or set GPUS=… explicitly."
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPUS="$(awk -F',' '{print NF}' <<< "$GPUS")"

WORK="$SCRATCH_ROOT/outputs/full_jhuapl_mvsam3d"
mkdir -p "$SCRATCH_ROOT/WRIVA" "$WORK/$RUN_NAME/logs" "$OUT_NAS"
LOG="$WORK/$RUN_NAME/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "host=$HOST  gpus=$GPUS ($NGPUS)  scene=$SCENE  run=$RUN_NAME  stages=$STAGES"
echo "log: $LOG"

echo "sync inputs $DATA_NAS/$SCENE → $SCRATCH_ROOT/WRIVA/$SCENE"
rsync -a --delete --exclude 'model/' --exclude 'sparse/gsplat/' --exclude 'metrics/' \
  --exclude 'sparse/map_anything_rescaled_sparse/' "$DATA_NAS/$SCENE/" "$SCRATCH_ROOT/WRIVA/$SCENE/"
if [[ -d "$OUT_NAS/$RUN_NAME" ]]; then   # resume on a node whose scratch lacks earlier results
  rsync -a --ignore-existing "$OUT_NAS/$RUN_NAME/" "$WORK/$RUN_NAME/"
fi

activate() {
  set +u   # conda's activate scripts reference unset variables
  if ! type conda 2>/dev/null | grep -q function; then
    local base=""
    command -v conda >/dev/null && base="$(conda info --base 2>/dev/null)"
    for f in "$base/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/mambaforge/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
      # shellcheck disable=SC1090
      [[ -f "$f" ]] && source "$f" && break
    done
  fi
  conda activate "$1"
  set -u
}
module load cuda 2>/dev/null || true
export STAGE2_READONLY_ROOTS="/data2/mehark:/data/mehark:/data3/mehark"
export HF_HOME="${HF_HOME:-/home/${SCS_ID}/.cache/huggingface}"
export PYTHONUNBUFFERED=1

SETS=(--set "wriva_root=$SCRATCH_ROOT/WRIVA" --set "work_root=$WORK" --set "run_name=$RUN_NAME"
      --set "mvsam3d_root=$MVSAM3D_ROOT")
[[ -n "$MASKS_IMPORT_DIR" ]] && SETS+=(--set "masks_import_dir=$MASKS_IMPORT_DIR")
for kv in $EXTRA_SETS; do SETS+=(--set "$kv"); done
ENTRY="$HERE/run.py"
cd "$HERE"

sync_back() { rsync -a --exclude '.claim_*' "$WORK/$RUN_NAME/" "$OUT_NAS/$RUN_NAME/"; }
# claims left by a killed run would block those buildings until they go stale (one launcher per run)
clear_claims() { find "$WORK/$RUN_NAME" -name '.claim_*' -delete 2>/dev/null || true; }
# One process per GPU. Rendezvous on 127.0.0.1 at a free port: no hostname lookup (--standalone hung on a
# machine whose localhost reverse-resolved badly), and no clash with other users on the default port 29500.
free_port() { python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
run_ranks() {
  torchrun --nnodes=1 --nproc_per_node="$NGPUS" --master_addr=127.0.0.1 --master_port="$(free_port)" \
    "$ENTRY" "$@" --config "$CONFIG" --scene "$SCENE" "${SETS[@]}"
}

IFS=',' read -ra STAGE_LIST <<< "$STAGES"
for stage in "${STAGE_LIST[@]}"; do
  echo ""
  echo "================= $stage  ($(date '+%F %T')) ================="
  case "$stage" in
    masks)
      activate "$CONDA_ENV_SAM3"
      run_ranks masks ;;
    mvsam3d)
      activate "$CONDA_ENV"
      if ! ls "$MVSAM3D_ROOT"/checkpoints/*/pipeline.yaml >/dev/null 2>&1; then
        echo "ERROR: no $MVSAM3D_ROOT/checkpoints/<tag>/pipeline.yaml. Clone MV-SAM3D and download SAM 3D Objects (README §1)."
        exit 1
      fi
      clear_claims
      run_ranks mvsam3d ;;
    place)
      activate "$CONDA_ENV"
      clear_claims
      run_ranks place ;;
    doctor)   # both envs: mvsam3d runs everything but `masks`, sam3 runs `masks`
      for env in "$CONDA_ENV" "$CONDA_ENV_SAM3"; do
        echo "--- conda env $env"
        if activate "$env" 2>/dev/null; then
          python "$ENTRY" doctor --config "$CONFIG" --scene "$SCENE" "${SETS[@]}" || true
        else
          echo "✗ conda env $env does not exist"
        fi
      done ;;
    inspect|prepare|import_masks|associate|assemble)
      activate "$CONDA_ENV"
      python "$ENTRY" "$stage" --config "$CONFIG" --scene "$SCENE" "${SETS[@]}" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
  sync_back
done

echo ""
echo "done. Results: $OUT_NAS/$RUN_NAME/  (scene/summary.json, scene/objects.csv, scene/*.glb)"
