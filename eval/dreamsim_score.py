"""DreamSim evaluation helpers (prof-requested perceptual score).

Usage (after you have rendered novel/held-out views)::

  python -m eval.dreamsim_score \\
    --pred-dir outputs/renders/ours \\
    --ref-dir  /data2/mehark/WRIVA/<scene>/metrics/ref_images \\
    --out      outputs/eval/dreamsim.json

Pairing is by filename stem. Lower DreamSim distance = more similar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _list_images(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def pair_by_stem(pred_dir: Path, ref_dir: Path) -> list[tuple[Path, Path, str]]:
    preds = _list_images(Path(pred_dir))
    refs = _list_images(Path(ref_dir))
    common = sorted(set(preds) & set(refs))
    if not common:
        raise FileNotFoundError(
            f"no shared stems between {pred_dir} ({len(preds)} imgs) "
            f"and {ref_dir} ({len(refs)} imgs)"
        )
    return [(preds[s], refs[s], s) for s in common]


def compute_dreamsim(
    pairs: list[tuple[Path, Path, str]],
    *,
    device: str = "cuda",
) -> dict:
    """Return mean/per-image DreamSim distances. Requires `pip install dreamsim`."""
    try:
        from dreamsim import dreamsim
        from PIL import Image
    except ImportError as e:
        raise SystemExit(
            "dreamsim not installed. On the GPU node:\n"
            "  pip install dreamsim\n"
            f"original error: {e}"
        ) from e

    model, preprocess = dreamsim(pretrained=True, device=device)
    per = []
    for pred_p, ref_p, stem in pairs:
        img_a = preprocess(Image.open(pred_p).convert("RGB")).to(device)
        img_b = preprocess(Image.open(ref_p).convert("RGB")).to(device)
        # dreamsim returns distance (lower = more similar)
        d = float(model(img_a, img_b).item())
        per.append({"stem": stem, "dreamsim": d, "pred": str(pred_p), "ref": str(ref_p)})

    vals = [x["dreamsim"] for x in per]
    return {
        "n": len(vals),
        "dreamsim_mean": sum(vals) / max(len(vals), 1),
        "dreamsim_min": min(vals) if vals else None,
        "dreamsim_max": max(vals) if vals else None,
        "per_image": per,
        "note": "lower DreamSim distance = closer perceptual match",
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="DreamSim score: pred renders vs ref images")
    p.add_argument("--pred-dir", required=True, type=str)
    p.add_argument("--ref-dir", required=True, type=str)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    pairs = pair_by_stem(Path(args.pred_dir), Path(args.ref_dir))
    print(f"paired {len(pairs)} images by stem")
    report = compute_dreamsim(pairs, device=args.device)
    print(
        f"DreamSim mean={report['dreamsim_mean']:.6f} "
        f"(min={report['dreamsim_min']:.6f}, max={report['dreamsim_max']:.6f})"
    )
    if args.out:
        import sys
        from pathlib import Path as _P

        _root = _P(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from stage2_lora.safety import assert_writable_dir

        out = Path(args.out)
        assert_writable_dir(out.parent, label="dreamsim out dir", create=True)
        out.write_text(json.dumps(report, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
