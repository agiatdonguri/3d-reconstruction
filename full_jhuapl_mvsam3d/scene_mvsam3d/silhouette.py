"""Occlusion-aware silhouettes of a point-sampled mesh, and their IoU with the object's masks.

A mesh pixel counts only if the mesh is not hidden behind what the photo saw there (metric depth +
tolerance), so a building partly behind a tree is not penalized for the tree — SAM 3 masks only cover
the visible part too."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class EvalView:
    stem: str
    K: torch.Tensor        # 3×3 at (width, height)
    w2c: torch.Tensor      # 4×4
    width: int
    height: int
    mask: torch.Tensor     # (H, W) bool
    depth: torch.Tensor    # (H, W) metric depth seen in the photo, 0 = unknown


def build_eval_views(scene, assoc, obj_id: str, stems: list[str], max_side: int, device) -> list[EvalView]:
    views = []
    for stem in stems:
        cam = scene.by_stem[stem]
        s = min(1.0, max_side / max(cam.width, cam.height))
        w, h = max(1, round(cam.width * s)), max(1, round(cam.height * s))
        m = assoc.mask(obj_id, stem, (w, h))
        d = np.zeros((h, w), np.float32)
        if scene.depth_path(cam) is not None:
            d = np.asarray(Image.fromarray(scene.load_depth(cam)[0]).resize((w, h), Image.NEAREST), np.float32)
        views.append(EvalView(stem, torch.tensor(cam.scaled_K(w, h), dtype=torch.float32, device=device),
                              torch.tensor(cam.w2c, dtype=torch.float32, device=device), w, h,
                              torch.from_numpy(m.copy()).to(device), torch.from_numpy(d.copy()).to(device)))
    return views


def render_silhouette(pts: torch.Tensor, view: EvalView, occlusion_tol: float, dilate: int = 1) -> torch.Tensor:
    pc = pts @ view.w2c[:3, :3].T + view.w2c[:3, 3]
    z = pc[:, 2]
    front = z > 1e-6
    zs = torch.where(front, z, torch.ones_like(z))
    u = torch.round(view.K[0, 0] * pc[:, 0] / zs + view.K[0, 2]).long()
    v = torch.round(view.K[1, 1] * pc[:, 1] / zs + view.K[1, 2]).long()
    ok = front & (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
    zbuf = torch.full((view.height * view.width,), float("inf"), device=pts.device)
    if ok.any():
        zbuf.scatter_reduce_(0, v[ok] * view.width + u[ok], z[ok], reduce="amin")
    obs = view.depth.reshape(-1)
    hidden = (obs > 0) & (zbuf > obs * (1.0 + occlusion_tol))
    sil = (torch.isfinite(zbuf) & ~hidden).reshape(1, 1, view.height, view.width).float()
    if dilate > 0:   # closing: fills gaps between projected samples without growing the outline
        k = 2 * dilate + 1
        sil = F.max_pool2d(sil, k, stride=1, padding=dilate)
        sil = -F.max_pool2d(-sil, k, stride=1, padding=dilate)
    return sil[0, 0] > 0.5


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    return float((a & b).sum().item() / union) if union else 0.0


def mean_iou(pts: torch.Tensor, views: list[EvalView], occlusion_tol: float) -> tuple[float, list[float]]:
    per = [iou(render_silhouette(pts, v, occlusion_tol), v.mask) for v in views]
    return (float(np.mean(per)) if per else 0.0), per


def n_surface_samples(views: list[EvalView], base: int, cap: int = 600_000) -> int:
    """Enough points that the largest mask gets several samples per pixel (front and back surfaces)."""
    biggest = max((int(v.mask.sum().item()) for v in views), default=0)
    return int(min(cap, max(base, 8 * biggest)))
