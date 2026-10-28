"""MV-SAM3D output (canonical mesh + pose in the reference camera) → WRIVA world frame.

Same chain MV-SAM3D uses to overlay its result on the input pointmap (run_inference_weighted.py,
overlay_sam3d_on_pointmap / merge_glb_with_da3_aligned):
  canonical GLB vertices → Z-up→Y-up → scale, rotate (p @ R, PyTorch3D row-vector convention), translate
  → PyTorch3D camera (x left, y up) → OpenCV camera (diag(−1, −1, 1)) → world via the reference view's c2w.
The pose is metric because the reference view's pointmap is (WRIVA metric depth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import quat_wxyz_to_matrix, transform_points

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])


def load_pose(params_npz: str | Path) -> dict:
    z = np.load(params_npz)
    scale = np.asarray(z["scale"], np.float64).reshape(-1)
    return {
        "rotation": np.asarray(z["rotation"], np.float64).reshape(-1)[:4],
        "scale": np.repeat(scale[:1], 3) if scale.size == 1 else scale[:3],
        "translation": np.asarray(z["translation"], np.float64).reshape(-1)[:3],
    }


def canonical_to_camera_cv(verts: np.ndarray, pose: dict) -> np.ndarray:
    v = np.asarray(verts, np.float64) @ Z_UP_TO_Y_UP.T
    p3d = (v * pose["scale"]) @ quat_wxyz_to_matrix(pose["rotation"]) + pose["translation"]
    return p3d @ P3D_TO_CV.T


def canonical_to_world(verts: np.ndarray, pose: dict, w2c_ref: np.ndarray) -> np.ndarray:
    return transform_points(np.linalg.inv(w2c_ref), canonical_to_camera_cv(verts, pose))


def load_world_scene(result_glb: str | Path, pose: dict, w2c_ref: np.ndarray):
    """→ (trimesh.Scene with textures kept, one concatenated Trimesh for geometry checks)."""
    import trimesh

    loaded = trimesh.load(result_glb, process=False)
    geoms = list(loaded.geometry.items()) if isinstance(loaded, trimesh.Scene) else [("object", loaded)]
    scene, meshes = trimesh.Scene(), []
    for name, g in geoms:
        if not hasattr(g, "vertices") or len(g.vertices) == 0:
            continue
        g = g.copy()
        g.vertices = canonical_to_world(np.asarray(g.vertices), pose, w2c_ref)
        scene.add_geometry(g, node_name=str(name))
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            meshes.append(trimesh.Trimesh(g.vertices, g.faces, process=False))
    if not meshes:
        raise ValueError(f"{result_glb} has no triangle mesh")
    return scene, trimesh.util.concatenate(meshes)


def apply_to_scene(scene, fn):
    """Copy of a trimesh.Scene with fn applied to every geometry's vertices (world → world)."""
    import trimesh

    out = trimesh.Scene()
    for name, g in scene.geometry.items():
        g = g.copy()
        g.vertices = fn(np.asarray(g.vertices, np.float64))
        out.add_geometry(g, node_name=str(name))
    return out
