"""Per-object MV-SAM3D: write its input folder, run run_inference_weighted.py, collect the outputs.

Input layout MV-SAM3D expects (see its README / notebook/load_images_and_masks.py)::

  <obj_dir>/images/<stem>.png         selected views, reference view first in --image_names
  <obj_dir>/<obj_id>/<stem>.png       RGBA, alpha = this object's mask
  <obj_dir>/da3_output.npz            DA3-format geometry, here from WRIVA instead of Depth Anything 3:
                                      pointmaps_sam3d (N,3,H,W) camera-space XYZ (x right, y down, z fwd),
                                      extrinsics (N,4,4) world→camera, intrinsics (N,3,3) at the image size,
                                      image_files (N,) matched to images by stem
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .associate import Associations
from .config import SceneConfig
from .geometry import unproject_depth
from .scene_data import SceneData


def object_dir(work_dir: Path, obj_id: str) -> Path:
    """Folder name doubles as MV-SAM3D's dataset name, so it is made unique per run (parallel workers share
    <mvsam3d_root>/visualization)."""
    import hashlib

    tag = hashlib.sha1(str(work_dir.resolve()).encode()).hexdigest()[:8]
    return work_dir / "objects" / f"{tag}_{obj_id}"


def write_object_input(cfg: SceneConfig, scene: SceneData, assoc: Associations, obj_id: str, stems: list[str],
                       obj_dir: Path) -> None:
    img_dir, mask_dir = obj_dir / "images", obj_dir / obj_id
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    pointmaps, depths, w2cs, Ks = [], [], [], []
    depth_hw = None
    for stem in stems:
        cam = scene.by_stem[stem]
        rgb = scene.load_image(cam, cfg.image_max_side)
        h, w = rgb.shape[:2]
        Image.fromarray(rgb).save(img_dir / f"{stem}.png")
        m = assoc.mask(obj_id, stem, (w, h))
        rgba = np.concatenate([rgb, (m * 255).astype(np.uint8)[..., None]], -1)
        Image.fromarray(rgba, "RGBA").save(mask_dir / f"{stem}.png")
        d, _ = scene.load_depth(cam)
        if depth_hw is None:
            depth_hw = d.shape
        if d.shape != depth_hw:
            d = np.asarray(Image.fromarray(d).resize((depth_hw[1], depth_hw[0]), Image.NEAREST))
        Kd = cam.scaled_K(depth_hw[1], depth_hw[0])
        pointmaps.append(unproject_depth(d.astype(np.float64), Kd).transpose(2, 0, 1).astype(np.float32))
        depths.append(d.astype(np.float32))
        w2cs.append(cam.w2c.astype(np.float32))
        Ks.append(cam.scaled_K(w, h).astype(np.float32))
    np.savez(
        obj_dir / "da3_output.npz",
        depth=np.stack(depths), pointmaps=np.stack(pointmaps).transpose(0, 2, 3, 1),
        pointmaps_sam3d=np.stack(pointmaps), extrinsics=np.stack(w2cs), intrinsics=np.stack(Ks),
        image_files=np.array([f"{s}.png" for s in stems]),
    )
    (obj_dir / "views.json").write_text(json.dumps({"object": obj_id, "views": stems, "reference": stems[0]}))


def run_mvsam3d(cfg: SceneConfig, obj_dir: Path, obj_id: str, stems: list[str], gpu: str) -> dict:
    root = Path(cfg.mvsam3d_root).resolve()
    script = root / "run_inference_weighted.py"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found (set mvsam3d_root)")
    if not (root / "checkpoints" / cfg.model_tag / "pipeline.yaml").is_file():
        raise FileNotFoundError(f"{root}/checkpoints/{cfg.model_tag}/pipeline.yaml missing — download the "
                                "SAM 3D Objects checkpoints into the MV-SAM3D checkout first")
    cmd = [
        cfg.mvsam3d_python or sys.executable, str(script),
        "--input_path", str(obj_dir.resolve()), "--mask_prompt", obj_id, "--image_names", ",".join(stems),
        "--da3_output", str((obj_dir / "da3_output.npz").resolve()), "--model_tag", cfg.model_tag,
        "--seed", str(cfg.seed), "--stage1_steps", str(cfg.stage1_steps), "--stage2_steps", str(cfg.stage2_steps),
        "--decode_formats", cfg.decode_formats, "--stage2_weight_source", cfg.stage2_weight_source,
    ]
    if cfg.mvsam3d_pose_optimization:
        cmd.append("--run_pose_optimization")
    cmd += list(cfg.mvsam3d_extra_args)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUMM_DISABLE_JIT="1", SPCONV_DISABLE_JIT="1", PYTHONUNBUFFERED="1")
    log_path = obj_dir / "mvsam3d.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        # MV-SAM3D resolves notebook/ and checkpoints/ relative to the working directory
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=cfg.object_timeout_min * 60)
    viz = root / "visualization" / obj_dir.name / obj_id
    runs = sorted((p for p in viz.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime) if viz.is_dir() else []
    result = {"returncode": proc.returncode, "seconds": round(time.time() - t0, 1), "log": str(log_path),
              "cmd": cmd, "gpu": gpu}
    if proc.returncode != 0 or not runs:
        result["error"] = "MV-SAM3D failed" if proc.returncode else "no output folder under visualization/"
        result["log_tail"] = log_path.read_text()[-4000:]
        return result
    dest = obj_dir / "mvsam3d"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(runs[-1]), dest)
    shutil.rmtree(root / "visualization" / obj_dir.name, ignore_errors=True)
    for key, name in (("glb", "result.glb"), ("params", "params.npz"),
                      ("optimized_params", "pose_optimization/optimized_params.npz"), ("gaussians", "result.ply")):
        if (dest / name).is_file():
            result[key] = str(dest / name)
    if "glb" not in result or "params" not in result:
        result["error"] = f"result.glb / params.npz missing in {dest}"
    return result
