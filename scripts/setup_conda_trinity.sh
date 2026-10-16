#!/usr/bin/env bash
# Trinity: bootstrap conda env for Stage-2 LoRA / MV-SAM3D (run on a GPU compute node).
# Mirrors Colab pins from notebooks/mvsam3d_stage2LoRA.ipynb (torch 2.5.1+cu121).
set -euo pipefail

SCS_ID="${SCS_ID:-tnishida}"
CODE="${CODE:-/home/${SCS_ID}/code}"
ENV_NAME="${ENV_NAME:-mvsam3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

mkdir -p "$CODE"
cd "$CODE"

echo "=== host=$(hostname) ==="
nvidia-smi -L || true
module load cuda 2>/dev/null || module load cuda-12.1 2>/dev/null || true

# ---- conda / mamba ----
if ! command -v conda >/dev/null 2>&1; then
  echo "Installing Miniconda into ~/miniconda3 ..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Newer Miniconda requires explicit ToS accept (non-interactive clusters)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi
conda activate "$ENV_NAME"
python -V

# Prefer mamba later if installs are slow: conda install -y -n base -c conda-forge mamba

echo "=== PyTorch 2.5.1+cu121 ==="
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28.post3
python -c "import torch; assert torch.__version__.startswith('2.5.1'); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "=== clone repos ==="
if [[ ! -d "$CODE/sam-3d-objects/.git" ]]; then
  git clone https://github.com/facebookresearch/sam-3d-objects.git "$CODE/sam-3d-objects"
fi
if [[ ! -d "$CODE/Depth-Anything-3/.git" ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$CODE/Depth-Anything-3"
fi
# Same remote as Colab notebook (devinli123/MV-SAM3D)
MVSAM3D_GIT="${MVSAM3D_GIT:-https://github.com/devinli123/MV-SAM3D.git}"
if [[ ! -d "$CODE/MV-SAM3D/.git" ]]; then
  git clone "$MVSAM3D_GIT" "$CODE/MV-SAM3D"
fi

echo "=== light LoRA + mask deps ==="
pip install "numpy==1.26.4" "scipy==1.13.1"
pip install -r /home/${SCS_ID}/3d-reconstruction/requirements-stage2-lora.txt
pip install segment-anything huggingface_hub

echo ""
echo "NEXT (manual — heavy SAM3D stack, same as Colab §4):"
echo "  1) export HF_TOKEN=hf_..."
echo "  2) huggingface-cli login --token \$HF_TOKEN"
echo "  3) Install sam-3d-objects + DA3 + flash-attn/kaolin/spconv like the notebook"
echo "     (or rsync your Colab compiled_pkgs backup to $CODE/)"
echo "  4) Masks: see docs/SETUP_ENV.md"
echo "Done base env: conda activate $ENV_NAME"
