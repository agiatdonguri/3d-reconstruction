#!/usr/bin/env bash
# Trinity: conda env `sam3` for the `masks` stage (text-prompted SAM 3 instance masks). Separate from the
# `mvsam3d` env because SAM 3 needs torch ≥ 2.7 / CUDA ≥ 12.6 while the SAM 3D stack is pinned to 2.5.1.
# Not needed if you use Mehar's masks (import_masks stage). Run on a compute node; needs Hugging Face
# access to facebook/sam3 (request it on the model page) and HF_TOKEN in the environment.
set -euo pipefail

SCS_ID="${SCS_ID:-tnishida}"
CODE="${CODE:-/home/${SCS_ID}/code}"
ENV_NAME="${ENV_NAME:-sam3}"

for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh"; do
  # shellcheck disable=SC1090
  [[ -f "$f" ]] && source "$f" && break
done
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

pip install -U pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
mkdir -p "$CODE"
if [[ ! -d "$CODE/sam3/.git" ]]; then
  git clone https://github.com/facebookresearch/sam3.git "$CODE/sam3"
fi
pip install -e "$CODE/sam3"
pip install numpy scipy pillow pyyaml
python -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sam3 ok')"

echo ""
echo "Next: export HF_TOKEN=… (an account with access to facebook/sam3), then"
echo "  SCENE=<scene> STAGES=masks $(dirname "$0")/run_trinity.sh"
