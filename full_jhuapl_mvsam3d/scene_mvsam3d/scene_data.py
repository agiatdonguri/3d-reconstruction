"""One WRIVA scene: cameras (COLMAP), RGB, metric z-depth and its scale."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .colmap_io import Camera, find_colmap_dir, load_cameras, read_points3d
from .config import SceneConfig
from .geometry import project, robust_log_mode

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class SceneData:
    def __init__(self, cfg: SceneConfig, depth_scales: Optional[dict[str, float]] = None):
        self.cfg = cfg
        self.root = cfg.scene_root()
        if not self.root.is_dir():
            raise FileNotFoundError(f"scene not found: {self.root}")
        self.images_dir = self.root / cfg.images_rel
        self.sparse_dir = find_colmap_dir(self.root, cfg.colmap_rel)
        self.depth_dir = self.root / cfg.depth_rel
        if cfg.depth_kind not in ("z", "ray"):
            raise ValueError(f"depth_kind must be z or ray, got {cfg.depth_kind!r}")
        files = {p.name: p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        by_stem = {p.stem: p for p in files.values()}
        cams = []
        for cam in load_cameras(self.sparse_dir):
            path = files.get(Path(cam.name).name) or by_stem.get(cam.stem)
            if path is None or not fnmatch.fnmatch(path.name, cfg.image_glob):
                continue
            cams.append(cam)
        if cfg.max_images > 0:
            cams = cams[: cfg.max_images]
        if not cams:
            raise RuntimeError(f"no registered images under {self.images_dir} match image_glob={cfg.image_glob!r}")
        self._files = {c.stem: files.get(Path(c.name).name) or by_stem[c.stem] for c in cams}
        self.cameras: list[Camera] = cams
        self.by_stem = {c.stem: c for c in cams}
        self.depth_scales = depth_scales or {}

    # ---- RGB ----
    def image_path(self, cam: Camera) -> Path:
        return self._files[cam.stem]

    def load_image(self, cam: Camera, max_side: int = 0) -> np.ndarray:
        img = Image.open(self.image_path(cam)).convert("RGB")
        if img.size != (cam.width, cam.height):
            img = img.resize((cam.width, cam.height), Image.BICUBIC)
        if max_side and max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
        return np.asarray(img)

    # ---- depth ----
    def depth_path(self, cam: Camera) -> Optional[Path]:
        for pat in self.cfg.depth_patterns:
            p = self.depth_dir / pat.format(stem=cam.stem, name=cam.name)
            if p.is_file():
                return p
        return None

    def load_depth_raw(self, cam: Camera) -> np.ndarray:
        p = self.depth_path(cam)
        if p is None:
            raise FileNotFoundError(f"no depth for {cam.name} in {self.depth_dir} (patterns {self.cfg.depth_patterns})")
        d = np.load(p)
        if isinstance(d, np.lib.npyio.NpzFile):
            d = d[d.files[0]]
        d = np.asarray(d, np.float32).squeeze()
        if d.ndim != 2:
            raise ValueError(f"depth {p} has shape {d.shape}, expected (H, W)")
        d[~np.isfinite(d) | (d <= 0)] = 0.0
        if self.cfg.depth_kind == "ray":
            d *= ray_cosine(cam.scaled_K(d.shape[1], d.shape[0]), d.shape)
        return d

    def scale_for(self, cam: Camera) -> float:
        return float(self.depth_scales.get(cam.stem, self.depth_scales.get("__global__", 1.0)))

    def load_depth(self, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
        """Metric z-depth (H_d, W_d) and the intrinsics at that resolution (0 = no depth)."""
        d = self.load_depth_raw(cam) * self.scale_for(cam)
        return d, cam.scaled_K(d.shape[1], d.shape[0])


def ray_cosine(K: np.ndarray, hw: tuple[int, int], u: Optional[np.ndarray] = None,
               v: Optional[np.ndarray] = None) -> np.ndarray:
    """cos(angle between the pixel's ray and the optical axis) = z-depth / distance along the ray."""
    if u is None:
        v, u = np.mgrid[0:hw[0], 0:hw[1]].astype(np.float32)
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    return (1.0 / np.sqrt(1.0 + x * x + y * y)).astype(np.float32)


def parse_depth_scale_json(path: Path, stems: list[str]) -> dict[str, float]:
    """depth_scale.json in whatever shape it comes: a number, {"scale": s}, or {image: s}."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, (int, float)):
        return {"__global__": float(data)}
    if isinstance(data, dict):
        for key in ("scale", "depth_scale", "global_scale", "metric_scale"):
            if isinstance(data.get(key), (int, float)):
                return {"__global__": float(data[key])}
        per = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                per[Path(str(k)).stem] = float(v)
            elif isinstance(v, dict):
                for key in ("scale", "depth_scale"):
                    if isinstance(v.get(key), (int, float)):
                        per[Path(str(k)).stem] = float(v[key])
        if per and any(s in per for s in stems):
            return per
    raise ValueError(f"could not interpret {path}: {str(data)[:300]}")


def _cos_slope(res: np.ndarray, logcos: np.ndarray, min_points: int = 30) -> float:
    """Slope of log(z_colmap / z_file) against log(cos θ): 0 if the files are z-depth, 1 if they hold the
    distance along the ray (z = distance · cos θ). Two robust least-squares passes around the mode."""
    keep = np.abs(res) < 0.25
    slope = float("nan")
    for _ in range(2):
        if keep.sum() < min_points or np.std(logcos[keep]) < 0.01:
            return float("nan")
        A = np.stack([logcos[keep], np.ones(int(keep.sum()))], 1)
        (slope, icpt), *_ = np.linalg.lstsq(A, res[keep], rcond=None)
        r = res - (slope * logcos + icpt)
        keep = np.abs(r) < max(3 * 1.4826 * float(np.median(np.abs(r[keep]))), 0.02)
    return float(slope)


def calibrate_depth_scales(scene: SceneData, max_points: int = 20000, min_points: int = 30,
                           points: Optional[tuple] = None) -> dict[str, dict]:
    """Per image: the factor that makes the depth map agree with the COLMAP sparse points.

    With 2D–3D tracks (points3D.txt) the observed pixels are used directly; with only points3D.ply (WRIVA)
    every point is projected and the mode of z_colmap / z_depth is taken (points hidden behind a surface
    only push the ratio up, so the mode ignores them). Also measures `cos_slope` (see _cos_slope)."""
    by_id, xyz = points if points is not None else read_points3d(scene.sparse_dir)
    out = {}
    rng = np.random.default_rng(0)
    for cam in scene.cameras:
        if scene.depth_path(cam) is None:
            out[cam.stem] = {"scale": float("nan"), "n": 0, "source": "no depth"}
            continue
        d = scene.load_depth_raw(cam)
        hd, wd = d.shape
        sx, sy = wd / cam.width, hd / cam.height
        if by_id is not None and cam.point3d_ids is not None:
            ok = cam.point3d_ids >= 0
            ids, uv = cam.point3d_ids[ok], cam.points2d[ok]
            keep = np.array([i in by_id for i in ids], bool)
            if keep.sum() < min_points:
                out[cam.stem] = {"scale": float("nan"), "n": int(keep.sum()), "source": "tracks"}
                continue
            pts = np.stack([by_id[i] for i in ids[keep]])
            u, v = uv[keep, 0] - 0.5, uv[keep, 1] - 0.5
            z = (pts @ cam.w2c[:3, :3].T + cam.w2c[:3, 3])[:, 2]
            source = "tracks"
        elif xyz is not None:
            pts = xyz if len(xyz) <= max_points else xyz[rng.choice(len(xyz), max_points, replace=False)]
            u, v, z = project(pts, cam.w2c, cam.K)
            source = "projected points3D (mode)"
        else:
            out[cam.stem] = {"scale": float("nan"), "n": 0, "source": "no sparse points"}
            continue
        ud = np.round((u + 0.5) * sx - 0.5).astype(int)
        vd = np.round((v + 0.5) * sy - 0.5).astype(int)
        ok = (z > 0) & (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        zd = np.zeros_like(z)
        zd[ok] = d[vd[ok], ud[ok]]
        ok &= zd > 0
        if ok.sum() < min_points:
            out[cam.stem] = {"scale": float("nan"), "n": int(ok.sum()), "source": source}
            continue
        ratio = z[ok] / zd[ok]
        s = robust_log_mode(ratio) if source.startswith("projected") else float(np.median(ratio))
        res = np.log(ratio) - np.log(s)
        logcos = np.log(ray_cosine(cam.K, (cam.height, cam.width), u[ok], v[ok]))
        out[cam.stem] = {"scale": s, "n": int(ok.sum()), "log_mad": float(np.median(np.abs(res))),
                         "cos_slope": _cos_slope(res, logcos), "source": source}
    return out


def resolve_depth_scales(scene: SceneData, mode: str, calib: Optional[dict[str, dict]] = None) -> dict[str, float]:
    """Depth multipliers for depth_scale_mode = auto | json | none."""
    mode = mode.lower()
    if mode == "none":
        return {"__global__": 1.0}
    if mode == "json":
        p = scene.root / scene.cfg.depth_scale_json_rel
        return parse_depth_scale_json(p, [c.stem for c in scene.cameras])
    if mode != "auto":
        raise ValueError(f"depth_scale_mode must be auto | json | none, got {mode!r}")
    calib = calib if calib is not None else calibrate_depth_scales(scene)
    good = {k: v["scale"] for k, v in calib.items() if np.isfinite(v.get("scale", np.nan))}
    if not good:
        raise RuntimeError("depth_scale_mode=auto but no image has enough sparse points to calibrate against; "
                           "use depth_scale_mode=json or none")
    scales = dict(good)
    scales["__global__"] = float(np.median(list(good.values())))
    return scales
