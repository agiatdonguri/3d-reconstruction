"""Per-view instance masks: bit-packed storage, the SAM 3 text-prompt segmenter, and import of existing masks."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_") or "object"


def instances_path(work_dir: Path, prompt: str, stem: str) -> Path:
    return Path(work_dir) / "masks" / slug(prompt) / f"{stem}.npz"


def save_instances(path: Path, masks: np.ndarray, scores: np.ndarray, boxes: Optional[np.ndarray], prompt: str) -> None:
    masks = np.asarray(masks, bool)
    k, h, w = masks.shape if masks.ndim == 3 else (0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(masks.reshape(k, -1), axis=1) if k else np.zeros((0, 0), np.uint8),
        shape=np.array([k, h, w], np.int64),
        scores=np.asarray(scores, np.float32).reshape(-1),
        boxes=np.asarray(boxes if boxes is not None else np.zeros((k, 4)), np.float32).reshape(-1, 4),
        prompt=np.array(prompt),
    )


def load_instances(path: Path) -> dict:
    z = np.load(path)
    k, h, w = (int(v) for v in z["shape"])
    masks = (np.unpackbits(z["packed"], axis=1, count=h * w).reshape(k, h, w).astype(bool)
             if k else np.zeros((0, h, w), bool))
    return {"masks": masks, "scores": z["scores"], "boxes": z["boxes"], "prompt": str(z["prompt"])}


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to (width, height)."""
    if (mask.shape[1], mask.shape[0]) == tuple(size):
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


class Sam3Segmenter:
    """facebookresearch/sam3 image model with a text prompt → every matching instance."""

    def __init__(self, checkpoint: str = "", sam3_root: str = "", score_thresh: float = 0.4):
        if sam3_root and sam3_root not in sys.path:
            sys.path.insert(0, sam3_root)
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if checkpoint:
            model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
        else:
            model = build_sam3_image_model()   # facebook/sam3 from Hugging Face (gated: needs HF_TOKEN + access)
        self.processor = Sam3Processor(model, confidence_threshold=score_thresh)

    def __call__(self, image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        with torch.no_grad():
            state = self.processor.set_image(image)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks, scores, boxes = out["masks"], out["scores"], out.get("boxes")
        to_np = lambda t: t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32)
        masks, scores = to_np(masks), to_np(scores).reshape(-1)
        boxes = to_np(boxes).reshape(-1, 4) if boxes is not None else np.zeros((len(scores), 4), np.float32)
        if masks.ndim == 4:
            masks = masks[:, 0]
        w, h = image.size
        if masks.size == 0:
            return np.zeros((0, h, w), bool), scores[:0], boxes[:0]
        return np.stack([resize_mask(m > 0.5, (w, h)) for m in masks]), scores, boxes


# --------------------------------------------------------------------------------------------- import
MASK_EXTS = (".npz", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_mask_files(root: Path, pattern: str, stem: str) -> list[Path]:
    base = pattern.format(stem=stem)
    if (root / base).is_dir():
        return sorted(p for p in (root / base).iterdir() if p.suffix.lower() in MASK_EXTS)
    glob = base if ("*" in base or Path(base).suffix.lower() in MASK_EXTS) else base + ".*"
    return sorted(p for p in root.glob(glob) if p.suffix.lower() in MASK_EXTS)


def read_mask_array(path: Path) -> np.ndarray:
    """Whatever the file holds: (K,H,W) stack, (H,W) label map or (H,W) binary mask."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        z = np.load(path)
        if "packed" in z.files and "shape" in z.files:   # this pipeline's own format
            return load_instances(path)["masks"]
        return np.asarray(z["masks"] if "masks" in z.files else z[z.files[0]])
    if suffix == ".npy":
        return np.load(path)
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        return np.asarray(im)[..., -1] > 127
    if suffix in (".jpg", ".jpeg"):                      # lossy: only a binary mask survives
        return np.asarray(im.convert("L")) > 127
    if im.mode == "RGB":                                 # colour-coded instances, black = background
        rgb = np.asarray(im).astype(np.int64)
        return rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    return np.asarray(im)


def to_instances(a: np.ndarray, kind: str = "auto") -> np.ndarray:
    """(K,H,W) bool from a stack, a binary mask or a label map (0 = background)."""
    a = np.asarray(a)
    if a.ndim == 3:
        return a > (0.5 if a.dtype.kind == "f" else 0)
    if a.ndim != 2:
        raise ValueError(f"mask array with shape {a.shape}")
    if a.dtype == bool or a.dtype.kind == "f":
        return (a > 0.5)[None]
    values = np.unique(a)
    values = values[values != 0]
    if kind == "binary" or (kind == "auto" and set(values.tolist()) <= {1, 255}):
        return (a > 0)[None]
    return np.stack([a == v for v in values]) if len(values) else np.zeros((0, *a.shape), bool)


def _boxes(masks: np.ndarray) -> np.ndarray:
    out = np.zeros((len(masks), 4), np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out


def import_masks(cfg, scene, work_dir: Path) -> dict:
    root = Path(cfg.masks_import_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"masks_import_dir does not exist: {root}")
    missing, counts, sizes = [], [], set()
    for cam in scene.cameras:
        files = find_mask_files(root, cfg.masks_import_pattern, cam.stem)
        if not files:
            missing.append(cam.stem)
            continue
        if len(files) == 1:
            arr = read_mask_array(files[0])
            sizes.add(tuple(arr.shape[-2:]))
            inst = to_instances(arr, cfg.masks_import_kind)
        else:                                            # one file per instance
            parts = [read_mask_array(f) for f in files]
            sizes.update(tuple(p.shape[-2:]) for p in parts)
            inst = np.concatenate([to_instances(p, "binary") if p.ndim == 2 else to_instances(p) for p in parts])
        inst = np.stack([resize_mask(m, (cam.width, cam.height)) for m in inst]) if len(inst) else \
            np.zeros((0, cam.height, cam.width), bool)
        inst = inst[inst.reshape(len(inst), -1).mean(1) >= cfg.mask_min_area_frac] if len(inst) else inst
        save_instances(instances_path(work_dir, cfg.masks_import_prompt, cam.stem), inst,
                       np.ones(len(inst), np.float32), _boxes(inst), cfg.masks_import_prompt)
        counts.append(len(inst))
    return {"views": len(scene.cameras), "imported": len(counts), "missing": missing,
            "instances_total": int(sum(counts)), "instances_per_view_median": float(np.median(counts)) if counts else 0.0,
            "source_mask_sizes_hw": sorted(sizes)[:10], "prompt": cfg.masks_import_prompt, "source": str(root)}
