"""Small geometry helpers shared by the stages (numpy, OpenCV camera convention)."""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """Same matrix as pytorch3d.transforms.quaternion_to_matrix for a (w, x, y, z) quaternion."""
    w, x, y, z = (np.asarray(q, np.float64).reshape(-1)[:4] / np.linalg.norm(np.asarray(q).reshape(-1)[:4]))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def project(pts_world: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (u, v, z_cam); u, v are pixel coordinates with pixel centres at integers."""
    pc = transform_points(w2c, pts_world)
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    return u, v, z


def unproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """z-depth (H, W) → camera-space points (H, W, 3), x right / y down / z forward."""
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], -1)


def unproject_pixels(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return transform_points(c2w, pc)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a to unit vector b."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        k = np.cross(a, perp)
        k /= np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    k = axis / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def orthonormal_frame(up: np.ndarray) -> np.ndarray:
    """3×3 with rows (e1, e2, up): two horizontal axes and the up axis."""
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.stack([e1, e2, up])


def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int = 2000, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Dominant plane n·x + d = 0 → (unit normal, d, inlier mask), refined by least squares on the inliers."""
    rng = np.random.default_rng(seed)
    best_n, best_d, best_cnt = None, 0.0, -1
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -float(n @ p[0])
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    inl = np.abs(pts @ best_n + best_d) < thresh
    c = pts[inl].mean(0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    d = -float(n @ c)
    return n, d, np.abs(pts @ n + d) < thresh


def robust_log_mode(ratios: np.ndarray, bin_width: float = 0.01) -> float:
    """Mode of log(ratio) via a histogram, refined by the median of its neighbourhood (occlusion-robust)."""
    r = np.log(ratios[np.isfinite(ratios) & (ratios > 0)])
    if r.size == 0:
        return float("nan")
    lo, hi = np.percentile(r, [1, 99])
    if hi - lo < bin_width:
        return float(np.exp(np.median(r)))
    hist, edges = np.histogram(r, bins=max(10, int((hi - lo) / bin_width)), range=(lo, hi))
    k = int(np.argmax(hist))
    centre = 0.5 * (edges[k] + edges[k + 1])
    near = r[np.abs(r - centre) < 5 * bin_width]
    return float(np.exp(np.median(near)))
