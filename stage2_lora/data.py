"""Aerial RGB + mask loading and MV-SAM3D preprocess helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image


def natkey(p: Path | str):
    s = Path(p).stem
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def list_image_paths(img_dir: Path, n: int) -> list[Path]:
    img_dir = Path(img_dir)
    assert img_dir.is_dir(), f"missing image dir: {img_dir}"
    paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=natkey,
    )[:n]
    if len(paths) < n:
        raise FileNotFoundError(f"need {n} images in {img_dir}, found {len(paths)}")
    return paths


def load_views(
    img_dir: Path,
    mask_dir: Path,
    n: int,
) -> tuple[list[Image.Image], list[Image.Image], list[Path]]:
    img_paths = list_image_paths(img_dir, n)
    mask_dir = Path(mask_dir)
    assert mask_dir.is_dir(), f"missing mask dir: {mask_dir}"

    images, masks = [], []
    for ip in img_paths:
        images.append(Image.open(ip).convert("RGB"))
        mp = mask_dir / f"{ip.stem}.png"
        if not mp.exists():
            cands = list(mask_dir.glob(ip.stem + "*"))
            if not cands:
                raise FileNotFoundError(f"no mask for {ip.name} under {mask_dir}")
            mp = cands[0]
        masks.append(Image.open(mp))
    return images, masks, img_paths


def as_alpha(mask) -> np.ndarray:
    """Force mask to HxWx1 uint8 in {0,255}."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[-1] == 4 else m[..., 0]
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        m = (m * 255).astype(np.uint8) if float(m.max()) <= 1.0 else m.astype(np.uint8)
    return m[..., None]


def preprocess_views(
    pipeline,
    view_images: Sequence,
    view_masks: Optional[Sequence] = None,
    view_pointmaps: Optional[Sequence] = None,
):
    """Same RGBA + ss/slat preprocess path as mvsam3d_stage2LoRA notebook."""
    n = len(view_images)
    if view_masks is None:
        view_masks = [None] * n
    if view_pointmaps is None:
        view_pointmaps = [None] * n

    ss_dicts, slat_dicts = [], []
    for image, mask, ext_pm in zip(view_images, view_masks, view_pointmaps):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image[..., :3]
        if mask is not None:
            rgba = np.concatenate([image.astype(np.uint8), as_alpha(mask)], axis=-1)
        else:
            alpha = np.full((*image.shape[:2], 1), 255, np.uint8)
            rgba = np.concatenate([image.astype(np.uint8), alpha], axis=-1)
        rgba_pil = Image.fromarray(rgba, mode="RGBA")

        if hasattr(pipeline, "compute_pointmap"):
            if ext_pm is not None:
                pmd = pipeline.compute_pointmap(
                    rgba_pil, pointmap=torch.from_numpy(np.asarray(ext_pm)).float()
                )
            else:
                pmd = pipeline.compute_pointmap(rgba_pil, pointmap=None)
            pointmap = pmd["pointmap"]
            ss_dicts.append(
                pipeline.preprocess_image(rgba_pil, pipeline.ss_preprocessor, pointmap=pointmap)
            )
            slat_dicts.append(pipeline.preprocess_image(rgba_pil, pipeline.slat_preprocessor))
        else:
            ss_dicts.append(pipeline.preprocess_image(rgba_pil, pipeline.ss_preprocessor))
            slat_dicts.append(pipeline.preprocess_image(rgba_pil, pipeline.slat_preprocessor))
    return ss_dicts, slat_dicts
