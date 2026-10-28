#!/usr/bin/env python3
"""Full JHU APL WRIVA scene with MV-SAM3D (GT depth + poses) — one entry point for every stage.

  python run.py <stage> --config configs/jhuapl_full.yaml --scene <scene> [--set key=value ...]
  torchrun --nnodes=1 --nproc_per_node=<gpus> --master_addr=127.0.0.1 --master_port=<free port> \
      run.py {masks|mvsam3d|place} --config ...

Stages in order: inspect, prepare, masks (or import_masks), associate, mvsam3d, place, assemble.
Every stage reads the previous one's files under <work_root>/<run_name>/ and skips work already done.
See README.md (how to run) and HOW_IT_WORKS.md (what each stage does).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scene_mvsam3d.config import load_config  # noqa: E402
from scene_mvsam3d.dist import destroy_distributed, dist_info, init_distributed_mode  # noqa: E402
from scene_mvsam3d.stages import STAGES  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", required=True)
    p.add_argument("--scene", default="", help="shortcut for --set scene=…")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set + ([f"scene={args.scene}"] if args.scene else []))
    distributed = init_distributed_mode(backend="gloo")
    if dist_info().is_main:
        work = cfg.work_dir()
        work.mkdir(parents=True, exist_ok=True)
        (work / f"config_{args.stage}.json").write_text(json.dumps(cfg.to_dict(), indent=1))
        print(f"== {args.stage} | scene {cfg.scene} | work_dir {work} | ranks {dist_info().world_size}"
              f"{' (process group: gloo)' if distributed else ''}", flush=True)
    try:
        STAGES[args.stage](cfg)
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
