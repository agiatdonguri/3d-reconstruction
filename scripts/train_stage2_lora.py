#!/usr/bin/env python3
"""CLI: train Stage-2 LoRA on Trinity (or any CUDA cluster node).

Examples
--------
  python scripts/train_stage2_lora.py --config configs/stage2_lora_va_split.yaml

  STAGE2_DATA_ROOT=/data/va_split \\
  STAGE2_MVSAM3D_ROOT=$HOME/code/MV-SAM3D \\
    python scripts/train_stage2_lora.py --config configs/stage2_lora_va_split.yaml \\
      --train-steps 200 --micro-n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python scripts/train_stage2_lora.py` without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage2_lora.config import load_config
from stage2_lora.train import run_training


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MV-SAM3D Stage-2 LoRA (cluster)")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--mvsam3d-root", type=str, default=None)
    p.add_argument("--pipeline-yaml", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--case-name", type=str, default=None)
    p.add_argument("--layout", type=str, default=None, choices=["legacy", "wriva", "va_split"])
    p.add_argument("--scene", type=str, default=None, help="WRIVA scene id")
    p.add_argument("--mask-root", type=str, default=None, help="sidecar SAM masks root")
    p.add_argument("--micro-n", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume-adapter", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "data_root": args.data_root,
        "mvsam3d_root": args.mvsam3d_root,
        "pipeline_yaml": args.pipeline_yaml,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "case_name": args.case_name,
        "layout": args.layout,
        "scene": args.scene,
        "mask_root": args.mask_root,
        "micro_n": args.micro_n,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "resume_adapter": args.resume_adapter,
        "log_every": args.log_every,
    }
    # default config next to package if none given
    cfg_path = args.config
    if cfg_path is None:
        default = _ROOT / "configs" / "stage2_lora_va_split.yaml"
        cfg_path = str(default) if default.exists() else None

    cfg = load_config(cfg_path, overrides=overrides)
    if not cfg.data_root:
        raise SystemExit("data_root required (YAML, --data-root, or STAGE2_DATA_ROOT)")
    if not cfg.mvsam3d_root:
        raise SystemExit("mvsam3d_root required (YAML, --mvsam3d-root, or STAGE2_MVSAM3D_ROOT)")

    print("Config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")

    run_dir = run_training(cfg)
    print("DONE", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
