#!/usr/bin/env python3
"""Generate (or copy) building masks for a WRIVA scene.

WRIVA has no SAM masks. This script writes a sidecar::

  <mask_out>/<scene>/*.png

Strategies
----------
1. --from-dir  Copy/rename existing Drive masks if stems already match RGB.
2. --full-frame  Write all-ones masks (debug only — bad for building recon).
3. --sam  (optional) Run a local SAM/SAM2 checkpoint if SAM_CKPT is set.

Does NOT hand-annotate. Prefer auto SAM for scenes you train; copy Drive
masks when ULTRRA overlap already exists.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage2_lora.safety import validate_mask_out  # noqa: E402
from stage2_lora.wriva import resolve_wriva_scene  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_rgb(images_dir: Path, max_n: int | None = None) -> list[Path]:
    paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )
    if max_n is not None and max_n > 0:
        paths = paths[: int(max_n)]
    return paths


def copy_from_dir(rgb_paths: list[Path], src_dir: Path, out_dir: Path) -> tuple[int, int]:
    src_dir = Path(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = miss = 0
    for ip in rgb_paths:
        cands = list(src_dir.glob(ip.stem + ".*"))
        if not cands:
            miss += 1
            continue
        src = cands[0]
        dst = out_dir / f"{ip.stem}.png"
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            from PIL import Image

            Image.open(src).save(dst)
        ok += 1
    return ok, miss


def write_full_frame(rgb_paths: list[Path], out_dir: Path) -> int:
    from PIL import Image
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    for ip in rgb_paths:
        im = Image.open(ip)
        w, h = im.size
        Image.fromarray(np.full((h, w), 255, np.uint8), mode="L").save(out_dir / f"{ip.stem}.png")
    return len(rgb_paths)


def run_sam(rgb_paths: list[Path], out_dir: Path, ckpt: Path, model_type: str) -> int:
    """Minimal SAM path; requires segment_anything + checkpoint on cluster."""
    import numpy as np
    from PIL import Image

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as e:
        raise SystemExit(
            "segment_anything not installed. Either pip install segment-anything "
            "or use --from-dir / --full-frame.\n"
            f"{e}"
        ) from e

    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.cuda()
    gen = SamAutomaticMaskGenerator(sam)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ip in rgb_paths:
        rgb = np.asarray(Image.open(ip).convert("RGB"))
        masks = gen.generate(rgb)
        if not masks:
            # fallback: full frame so pipeline can still run
            m = np.full(rgb.shape[:2], 255, np.uint8)
        else:
            # largest mask ≈ dominant object (building in aerial often large)
            best = max(masks, key=lambda d: float(d.get("area", 0)))
            m = (best["segmentation"].astype(np.uint8)) * 255
        Image.fromarray(m, mode="L").save(out_dir / f"{ip.stem}.png")
        print("mask", ip.name, "→", out_dir / f"{ip.stem}.png")
    return len(rgb_paths)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build SAM mask sidecar for WRIVA scene")
    p.add_argument(
        "--wriva-root",
        required=True,
        type=str,
        help="YOUR WRIVA copy, e.g. /data2/tnishida/WRIVA (not /data2/mehark)",
    )
    p.add_argument("--scene", required=True, type=str)
    p.add_argument(
        "--mask-out",
        required=True,
        type=str,
        help="under your tree, e.g. /data2/tnishida/masks",
    )
    p.add_argument("--from-dir", type=str, default="", help="copy existing masks (Drive dump)")
    p.add_argument("--full-frame", action="store_true", help="debug all-ones masks")
    p.add_argument("--sam", action="store_true", help="run segment-anything auto masks")
    p.add_argument("--sam-ckpt", type=str, default="", help="SAM .pth checkpoint")
    p.add_argument("--sam-type", type=str, default="vit_h")
    p.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="optional cap on #images (0=all). Use for smoke tests.",
    )
    args = p.parse_args(argv)

    # Never write into Mehar WRIVA
    mask_root = validate_mask_out(wriva_root=args.wriva_root, mask_out=args.mask_out)

    paths = resolve_wriva_scene(args.wriva_root, args.scene)
    images_dir = paths.require_images()
    rgb = list_rgb(images_dir, max_n=args.max_n or None)
    if not rgb:
        raise SystemExit(f"no images in {images_dir}")

    out = mask_root / args.scene
    print(f"scene={args.scene}  n_rgb={len(rgb)}  out={out}")

    if args.from_dir:
        ok, miss = copy_from_dir(rgb, Path(args.from_dir), out)
        print(f"copied={ok} missing_stems={miss}")
        if miss:
            print("WARNING: some RGB views have no mask — generate with --sam or fix stems")
        return 0 if ok else 1

    if args.sam:
        ckpt = Path(args.sam_ckpt)
        if not ckpt.is_file():
            raise SystemExit("--sam requires a valid --sam-ckpt")
        run_sam(rgb, out, ckpt, args.sam_type)
        return 0

    if args.full_frame:
        n = write_full_frame(rgb, out)
        print(f"wrote {n} full-frame masks (DEBUG ONLY)")
        return 0

    raise SystemExit("pick one of: --from-dir, --sam, or --full-frame (debug)")


if __name__ == "__main__":
    raise SystemExit(main())
