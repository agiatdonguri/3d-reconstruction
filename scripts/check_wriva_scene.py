#!/usr/bin/env python3
"""Preflight: print WRIVA scene summary (raw inputs + eval assets + mask coverage)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage2_lora.wriva import resolve_wriva_scene, summarize_scene  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wriva-root", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--mask-root", default="")
    args = p.parse_args(argv)
    paths = resolve_wriva_scene(
        args.wriva_root, args.scene, mask_root=args.mask_root or None
    )
    s = summarize_scene(paths)
    s["role_hint"] = {
        "reconstruct_from": [
            "images_dir",
            "colmap / sparse_nvm",
            "optional MoGe depth",
        ],
        "evaluate_only": ["model/", "gsplat/", "metrics/"],
        "masks": "sidecar via mask_root — not part of WRIVA dump",
    }
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
