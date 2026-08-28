#!/usr/bin/env bash
# Interactive / debug launch (no SLURM). Run on an allocated GPU node.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUMM_DISABLE_JIT="${CUMM_DISABLE_JIT:-1}"
export SPCONV_DISABLE_JIT="${SPCONV_DISABLE_JIT:-1}"
exec python scripts/train_stage2_lora.py --config configs/stage2_lora_va_split.yaml "$@"
