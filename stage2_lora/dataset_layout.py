"""Resolve image/mask directories for legacy va_split or Trinity WRIVA layouts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .wriva import WrivaPaths, resolve_wriva_scene, summarize_scene


@dataclass(frozen=True)
class ViewRoots:
    """Where Stage-2 LoRA reads RGB + SAM masks."""

    layout: str
    case_name: str
    images_dir: Path
    mask_dir: Path
    wriva: Optional[WrivaPaths] = None

    def summary(self) -> dict:
        if self.wriva is not None:
            return {"layout": self.layout, **summarize_scene(self.wriva)}
        return {
            "layout": self.layout,
            "case_name": self.case_name,
            "images_dir": str(self.images_dir),
            "mask_dir": str(self.mask_dir),
        }


def resolve_view_roots(cfg) -> ViewRoots:
    """
    layout:
      - \"legacy\" / \"va_split\": <data_root>/<aerial_subdir>/{images,masks}
      - \"wriva\": <data_root>/<scene>/sparse/undistorted_mitigated_images/0/
                masks from mask_root sidecar (Drive copy or generated)
    """
    layout = (getattr(cfg, "layout", "legacy") or "legacy").lower().strip()
    if layout in ("legacy", "va_split", "satellite"):
        images = cfg.img_dir()
        masks = cfg.mask_dir()
        return ViewRoots(
            layout="legacy",
            case_name=cfg.case_name,
            images_dir=images,
            mask_dir=masks,
        )

    if layout in ("wriva", "trinity"):
        scene = getattr(cfg, "scene", None) or cfg.case_name
        mask_root = getattr(cfg, "mask_root", "") or None
        paths = resolve_wriva_scene(
            cfg.data_root,
            scene,
            mask_root=mask_root,
            mask_subdir=getattr(cfg, "mask_subdir", "") or "",
            images_rel=getattr(cfg, "wriva_images_rel", "")
            or "sparse/undistorted_mitigated_images/0",
        )
        images = paths.require_images()
        masks = paths.require_masks()
        return ViewRoots(
            layout="wriva",
            case_name=paths.scene_name,
            images_dir=images,
            mask_dir=masks,
            wriva=paths,
        )

    raise ValueError(f"Unknown layout={layout!r} (use legacy|wriva)")
