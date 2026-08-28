"""WRIVA / Accenture scene layout on Trinity.

Peer layout (example)::

  /data2/mehark/WRIVA/<scene>/
    sparse/
      sparse_nvm/sparse.nvm
      metric_z_depth_mve/          # MoGe image_NNNN_zdepth.npy + _intrinsics.json
      undistorted_mitigated_images/0/
      depth_scale.json
      points3D.ply, images.txt, cameras.txt
      gsplat/...
      map_anything_rescaled_sparse/
    model/                         # *_acc_mesh.ply, ckpt, splat, …
    metrics/

SAM masks are NOT shipped with WRIVA. Keep a sidecar tree, e.g.::

  /data2/<you>/masks/<scene>/*.png
  # or copied Drive: SatelliteFuse/datasets/<case>/satellite/masks/

Image↔mask pairing is by stem (filename without extension).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WrivaPaths:
    """Resolved paths for one WRIVA scene."""

    scene_root: Path
    scene_name: str
    images_dir: Path
    mask_dir: Optional[Path]
    sparse_dir: Path
    model_dir: Path
    metrics_dir: Path
    sparse_nvm: Path
    depth_dir: Path
    depth_scale_json: Path
    colmap_images_txt: Path
    colmap_cameras_txt: Path
    points3d_ply: Path
    gsplat_dir: Path
    world_norm_npy: Path
    mapa_dir: Path

    def require_images(self) -> Path:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"WRIVA images missing: {self.images_dir}\n"
                "expected …/sparse/undistorted_mitigated_images/0/"
            )
        return self.images_dir

    def require_masks(self) -> Path:
        if self.mask_dir is None or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                "No SAM mask directory. WRIVA does not ship masks.\n"
                "Copy Drive masks (stem-matched) or generate them, then set "
                "mask_root / mask_subdir / STAGE2_MASK_ROOT.\n"
                f"scene={self.scene_name} tried={self.mask_dir}"
            )
        return self.mask_dir


def resolve_wriva_scene(
    wriva_root: str | Path,
    scene: str,
    *,
    mask_root: str | Path | None = None,
    mask_subdir: str = "",
    images_rel: str = "sparse/undistorted_mitigated_images/0",
) -> WrivaPaths:
    """Map Trinity WRIVA layout (+ optional mask sidecar) for one scene."""
    wriva_root = Path(wriva_root)
    scene_root = wriva_root / scene
    if not scene_root.is_dir():
        # allow passing scene_root directly as wriva_root when scene=""
        if scene in ("", ".") and wriva_root.is_dir():
            scene_root = wriva_root
            scene = wriva_root.name
        else:
            raise FileNotFoundError(f"WRIVA scene not found: {scene_root}")

    sparse = scene_root / "sparse"
    images_dir = scene_root / images_rel

    mask_dir: Optional[Path] = None
    if mask_root:
        mr = Path(mask_root)
        # Prefer <mask_root>/<scene>/… then flat <mask_root>/…
        candidates = []
        if mask_subdir:
            candidates += [
                mr / scene / mask_subdir,
                mr / mask_subdir,
                scene_root / mask_subdir,
            ]
        candidates += [
            mr / scene,
            mr,
            scene_root / "masks",
            sparse / "masks",
        ]
        for c in candidates:
            if c.is_dir():
                mask_dir = c
                break
        if mask_dir is None:
            mask_dir = candidates[0]  # best-effort path for error messages
    else:
        for c in (scene_root / "masks", sparse / "masks"):
            if c.is_dir():
                mask_dir = c
                break

    return WrivaPaths(
        scene_root=scene_root,
        scene_name=scene,
        images_dir=images_dir,
        mask_dir=mask_dir,
        sparse_dir=sparse,
        model_dir=scene_root / "model",
        metrics_dir=scene_root / "metrics",
        sparse_nvm=sparse / "sparse_nvm" / "sparse.nvm",
        depth_dir=sparse / "metric_z_depth_mve",
        depth_scale_json=sparse / "depth_scale.json",
        colmap_images_txt=sparse / "images.txt",
        colmap_cameras_txt=sparse / "cameras.txt",
        points3d_ply=sparse / "points3D.ply",
        gsplat_dir=sparse / "gsplat",
        world_norm_npy=sparse / "world_normalization_transform.npy",
        mapa_dir=sparse / "map_anything_rescaled_sparse",
    )


def summarize_scene(paths: WrivaPaths) -> dict:
    """Cheap existence checks for logging / SLURM preflight."""
    img_n = 0
    if paths.images_dir.is_dir():
        img_n = sum(
            1
            for p in paths.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    mask_n = 0
    if paths.mask_dir and paths.mask_dir.is_dir():
        mask_n = sum(
            1
            for p in paths.mask_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
    return {
        "scene": paths.scene_name,
        "scene_root": str(paths.scene_root),
        "images_dir": str(paths.images_dir),
        "n_images": img_n,
        "mask_dir": str(paths.mask_dir) if paths.mask_dir else None,
        "n_masks": mask_n,
        "has_sparse_nvm": paths.sparse_nvm.is_file(),
        "has_depth_dir": paths.depth_dir.is_dir(),
        "has_depth_scale": paths.depth_scale_json.is_file(),
        "has_colmap_txt": paths.colmap_images_txt.is_file()
        and paths.colmap_cameras_txt.is_file(),
        "has_points3d": paths.points3d_ply.is_file(),
        "has_gsplat": paths.gsplat_dir.is_dir(),
        "has_model": paths.model_dir.is_dir(),
    }
