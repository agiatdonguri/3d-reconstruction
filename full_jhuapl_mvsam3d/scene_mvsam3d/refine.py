"""Refinement in Mehar's parametrization: yaw, x/y (optionally z) translation, horizontal and vertical scale
about the object's base, with the ground normal as up — not a similarity transform, since single-view
SAM 3D came out 2–4× too tall with a good footprint. Objective: mean occlusion-aware silhouette IoU
against the object's SAM 3 masks in the other views. Coarse yaw and height scans, then Nelder–Mead."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from .geometry import orthonormal_frame
from .silhouette import EvalView, mean_iou

DOF_ORDER = ("yaw", "tx", "ty", "tz", "s_h", "s_v")
_STEP = {"yaw": 8.0, "tx": 0.03, "ty": 0.03, "tz": 0.03, "s_h": 0.2, "s_v": 0.25}


@dataclass
class Frame:
    origin: np.ndarray     # base centre of the object
    axes: np.ndarray       # rows e1, e2, up
    size: float            # horizontal extent; translations are fractions of it


def object_frame(points: np.ndarray, up: np.ndarray) -> Frame:
    axes = orthonormal_frame(up)
    c = points.mean(0)
    loc = (points - c) @ axes.T
    origin = c + axes[2] * np.percentile(loc[:, 2], 2)
    size = float(max(np.ptp(loc[:, 0]), np.ptp(loc[:, 1]), 1e-6))
    return Frame(origin, axes, size)


def _theta(x: np.ndarray, dofs: list[str]) -> dict[str, float]:
    t = {k: 0.0 for k in DOF_ORDER}
    t.update(dict(zip(dofs, (float(v) for v in x))))
    return t


def apply(points, theta: dict[str, float], frame: Frame):
    """Works on numpy arrays and torch tensors (same formula)."""
    is_t = torch.is_tensor(points)
    lib = torch if is_t else np
    A = torch.as_tensor(frame.axes, dtype=points.dtype, device=points.device) if is_t else frame.axes
    o = torch.as_tensor(frame.origin, dtype=points.dtype, device=points.device) if is_t else frame.origin
    loc = (points - o) @ A.T
    sh, sv = float(np.exp(theta["s_h"])), float(np.exp(theta["s_v"]))
    a, b, h = loc[:, 0] * sh, loc[:, 1] * sh, loc[:, 2] * sv
    c, s = float(np.cos(np.radians(theta["yaw"]))), float(np.sin(np.radians(theta["yaw"])))
    a2 = c * a - s * b + theta["tx"] * frame.size
    b2 = s * a + c * b + theta["ty"] * frame.size
    h2 = h + theta["tz"] * frame.size
    return o + lib.stack([a2, b2, h2], 1) @ A


def refine_object(pts: torch.Tensor, views: list[EvalView], frame: Frame, dofs: list[str], max_evals: int,
                  max_scale: float, occlusion_tol: float) -> dict:
    dofs = [d for d in DOF_ORDER if d in dofs]
    lim = float(np.log(max_scale))

    def loss(x):
        th = _theta(x, dofs)
        if abs(th["s_h"]) > lim or abs(th["s_v"]) > lim or max(abs(th["tx"]), abs(th["ty"]), abs(th["tz"])) > 0.5:
            return 2.0
        return 1.0 - mean_iou(apply(pts, th, frame), views, occlusion_tol)[0]

    x0 = np.zeros(len(dofs))
    best = loss(x0)
    if "yaw" in dofs:
        for yaw in np.arange(-180.0, 180.0, 15.0):
            x = x0.copy()
            x[dofs.index("yaw")] = yaw
            if (l := loss(x)) < best:
                best, x0 = l, x
    if "s_v" in dofs:
        for ls in np.log(2.0) * np.arange(-1.5, 1.51, 0.5):
            x = x0.copy()
            x[dofs.index("s_v")] = ls
            if (l := loss(x)) < best:
                best, x0 = l, x
    simplex = np.vstack([x0] + [x0 + np.eye(len(dofs))[i] * _STEP[d] for i, d in enumerate(dofs)])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxfev": max_evals, "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-4})
    x = res.x if res.fun <= best else x0
    th = _theta(x, dofs)
    return {"theta": th, "scale_h": float(np.exp(th["s_h"])), "scale_v": float(np.exp(th["s_v"])),
            "n_evals": int(res.nfev), "objective_iou": 1.0 - float(min(res.fun, best))}
