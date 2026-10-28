"""COLMAP text model → cameras in the OpenCV convention (pixel centres at integer coordinates)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Camera:
    name: str            # image file name as listed in images.txt
    width: int
    height: int
    K: np.ndarray        # 3×3, OpenCV convention: pixel (u, v) has its centre at (u, v)
    w2c: np.ndarray      # 4×4 world → camera (x right, y down, z forward)
    point3d_ids: Optional[np.ndarray] = None   # observed sparse points (COLMAP ids), if listed
    points2d: Optional[np.ndarray] = None      # their pixel positions (COLMAP convention)

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    def scaled_K(self, width: int, height: int) -> np.ndarray:
        """Intrinsics for the same view resampled to width × height."""
        return scale_intrinsics(self.K, self.width, self.height, width, height)


def scale_intrinsics(K: np.ndarray, w0: int, h0: int, w1: int, h1: int) -> np.ndarray:
    sx, sy = w1 / w0, h1 / h0
    K1 = K.astype(np.float64).copy()
    K1[0, 0] *= sx
    K1[1, 1] *= sy
    K1[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K1[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K1


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _intrinsics(model: str, params: list[float]) -> tuple[np.ndarray, bool]:
    """K in COLMAP's convention (pixel centres at +0.5) and whether the model has distortion."""
    model = model.upper()
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = params[:3]
        fx = fy = f
        distorted = model != "SIMPLE_PINHOLE" and any(abs(p) > 1e-12 for p in params[3:])
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[:4]
        distorted = model != "PINHOLE" and any(abs(p) > 1e-12 for p in params[4:])
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64), distorted


def read_cameras_txt(path: str | Path) -> dict[int, tuple[str, int, int, list[float]]]:
    cams = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cams[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]), [float(p) for p in parts[4:]])
    return cams


def read_images_txt(path: str | Path) -> list[dict]:
    """images.txt alternates a pose line and a (possibly empty) 2D-observation line per image."""
    lines = [l for l in Path(path).read_text().splitlines() if not l.startswith("#")]
    out = []
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        parts = head.split()
        rec = {
            "image_id": int(parts[0]),
            "qvec": np.array([float(v) for v in parts[1:5]]),
            "tvec": np.array([float(v) for v in parts[5:8]]),
            "camera_id": int(parts[8]),
            "name": " ".join(parts[9:]),
        }
        obs = lines[i + 1].split() if i + 1 < len(lines) else []
        if obs and len(obs) % 3 == 0:
            arr = np.array(obs, np.float64).reshape(-1, 3)
            rec["points2d"] = arr[:, :2]
            rec["point3d_ids"] = arr[:, 2].astype(np.int64)
        out.append(rec)
        i += 2
    return out


def read_points3d(sparse_dir: str | Path) -> tuple[Optional[dict[int, np.ndarray]], Optional[np.ndarray]]:
    """(id → xyz from points3D.txt, or None; all xyz as an array from .txt or .ply, or None)."""
    sparse_dir = Path(sparse_dir)
    txt = sparse_dir / "points3D.txt"
    if txt.is_file():
        by_id = {}
        for line in txt.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            by_id[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
        xyz = np.stack(list(by_id.values())) if by_id else None
        return by_id, xyz
    ply = sparse_dir / "points3D.ply"
    if ply.is_file():
        import trimesh

        pc = trimesh.load(ply, process=False)
        return None, np.asarray(pc.vertices, np.float64)
    return None, None


COLMAP_DIR_CANDIDATES = ("sparse/sparse", "sparse", "sparse/0", "sparse/sparse/0")   # WRIVA: sparse/sparse


def find_colmap_dir(scene_root: str | Path, colmap_rel: str = "") -> Path:
    """The folder holding cameras.txt + images.txt: `colmap_rel` if given, else the first candidate that has them."""
    scene_root = Path(scene_root)
    cands = [colmap_rel] if colmap_rel else list(COLMAP_DIR_CANDIDATES)
    for rel in cands:
        d = scene_root / rel
        if (d / "cameras.txt").is_file() and (d / "images.txt").is_file():
            return d
    binary = [str(scene_root / rel) for rel in cands if (scene_root / rel / "cameras.bin").is_file()]
    hint = (f"; binary models in {binary}: in YOUR copy run `colmap model_converter --input_path <dir> "
            "--output_path <dir> --output_type TXT`") if binary else ""
    raise FileNotFoundError(f"no cameras.txt + images.txt under {scene_root} in {cands}{hint}")


def load_cameras(sparse_dir: str | Path) -> list[Camera]:
    sparse_dir = Path(sparse_dir)
    cam_models = read_cameras_txt(sparse_dir / "cameras.txt")
    cams = []
    n_distorted = 0
    for rec in read_images_txt(sparse_dir / "images.txt"):
        model, w, h, params = cam_models[rec["camera_id"]]
        K_colmap, distorted = _intrinsics(model, params)
        n_distorted += int(distorted)
        K = K_colmap.copy()
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        w2c = np.eye(4)
        w2c[:3, :3] = qvec_to_rotmat(rec["qvec"])
        w2c[:3, 3] = rec["tvec"]
        cams.append(Camera(rec["name"], w, h, K, w2c, rec.get("point3d_ids"), rec.get("points2d")))
    if n_distorted:
        print(f"⚠ {n_distorted} cameras have distortion parameters; they are ignored "
              "(the WRIVA images used here are already undistorted)")
    return sorted(cams, key=lambda c: c.name)
