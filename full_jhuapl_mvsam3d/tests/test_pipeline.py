"""CPU tests on a synthetic WRIVA-style scene: two box buildings on a ground plane, eight aerial cameras,
COLMAP text files with 2D–3D tracks, z-depth stored at half scale (so `auto` must recover ×2), SAM-style
instance masks (one view merges both buildings into a single instance), and fake MV-SAM3D outputs built by
inverting the placement chain. The MV-SAM3D stage runs under real torchrun (2 CPU ranks, gloo).
Run from full_jhuapl_mvsam3d/:  python -m pytest tests -q"""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_mvsam3d import stages  # noqa: E402
from scene_mvsam3d.associate import Associations  # noqa: E402
from scene_mvsam3d.colmap_io import find_colmap_dir, load_cameras  # noqa: E402
from scene_mvsam3d.config import SceneConfig  # noqa: E402
from scene_mvsam3d.dist import claim  # noqa: E402
from scene_mvsam3d.geometry import quat_wxyz_to_matrix  # noqa: E402
from scene_mvsam3d.masks import instances_path, load_instances, save_instances  # noqa: E402
from scene_mvsam3d.mvsam3d_runner import object_dir, write_object_input  # noqa: E402
from scene_mvsam3d.placement import P3D_TO_CV, Z_UP_TO_Y_UP, canonical_to_world  # noqa: E402

W, H, F = 160, 120, 150.0
BOXES = {"A": (np.array([-10.0, -5, 0]), np.array([-2.0, 5, 6])), "B": (np.array([3.0, -4, 0]), np.array([12.0, 4, 10]))}
DEPTH_FILE_SCALE = 0.5          # files store true depth × 0.5


def look_at(center, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    f = (target - center) / np.linalg.norm(target - center)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R_c2w = np.stack([r, d, f], 1)
    w2c = np.eye(4)
    w2c[:3, :3] = R_c2w.T
    w2c[:3, 3] = -R_c2w.T @ center
    return w2c


def rot_to_qvec(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def raycast(w2c):
    """→ z-depth (H, W) and label image (−1 ground / nothing, 0 = A, 1 = B)."""
    K = np.array([[F, 0, W / 2 - 0.5], [0, F, H / 2 - 0.5], [0, 0, 1]])
    c2w = np.linalg.inv(w2c)
    v, u = np.mgrid[0:H, 0:W]
    dc = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, float)], -1).reshape(-1, 3)
    d = dc @ c2w[:3, :3].T
    o = c2w[:3, 3]
    t_best = np.full(len(d), np.inf)
    lab = np.full(len(d), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = -o[2] / d[:, 2]
        g = (d[:, 2] < 0) & (tg > 0)
        t_best[g] = tg[g]
        for k, (lo, hi) in enumerate(BOXES.values()):
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin = np.max(np.minimum(t1, t2), 1)
            tmax = np.min(np.maximum(t1, t2), 1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin < t_best)
            t_best[hit] = tmin[hit]
            lab[hit] = k
    z = np.where(np.isfinite(t_best), t_best * dc[:, 2], 0.0)
    return z.reshape(H, W), lab.reshape(H, W), K


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Laid out like /data2/mehark/WRIVA/<scene>/ (Mehar): sparse/{undistorted_mitigated_images/0,
    metric_z_depth_mve, depth_scale.json, sparse/ (COLMAP text + points3D.ply), sparse_nvm, map_anything_…}."""
    import trimesh
    from PIL import Image

    root = tmp_path_factory.mktemp("wriva")
    scene = root / "WRIVA" / "synth_scene"
    img_dir = scene / "sparse/undistorted_mitigated_images/0"
    depth_dir = scene / "sparse/metric_z_depth_mve"
    img_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    cams, labels, imgs_txt, pts_txt, xyz, pid = [], {}, [], [], [], 0
    rng = np.random.default_rng(0)
    for i in range(8):
        a = 2 * np.pi * i / 8
        w2c = look_at(np.array([45 * np.cos(a), 45 * np.sin(a), 32.0]))
        z, lab, K = raycast(w2c)
        stem = f"image_{i:04d}"
        rgb = np.full((H, W, 3), 90, np.uint8)
        rgb[lab == 0] = (200, 60, 60)
        rgb[lab == 1] = (60, 60, 200)
        Image.fromarray(rgb).save(img_dir / f"{stem}.jpg")
        np.save(depth_dir / f"{stem}_zdepth.npy", (z * DEPTH_FILE_SCALE).astype(np.float32))
        (depth_dir / f"{stem}_intrinsics.json").write_text(json.dumps({"fx": F, "fy": F, "cx": W / 2, "cy": H / 2}))
        labels[stem] = lab
        vv, uu = np.nonzero(z > 0)
        pick = rng.choice(len(vv), 60, replace=False)
        c2w = np.linalg.inv(w2c)
        obs = []
        for p in pick:
            zc = z[vv[p], uu[p]]
            Xc = np.array([(uu[p] - K[0, 2]) / F * zc, (vv[p] - K[1, 2]) / F * zc, zc])
            X = c2w[:3, :3] @ Xc + c2w[:3, 3]
            pts_txt.append(f"{pid} {X[0]} {X[1]} {X[2]} 128 128 128 0.5 {i + 1} {len(obs)}")
            xyz.append(X)
            obs.append(f"{uu[p] + 0.5} {vv[p] + 0.5} {pid}")
            pid += 1
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        imgs_txt.append(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {stem}.jpg")
        imgs_txt.append(" ".join(obs))
        cams.append((stem, w2c))
    cams_txt = f"1 PINHOLE {W} {H} {F} {F} {W / 2} {H / 2}\n"
    colmap = scene / "sparse/sparse"                        # COLMAP text + points3D.ply without track ids
    tracks = scene / "sparse/colmap_with_tracks"            # the same model with points3D.txt
    decoy = scene / "sparse/map_anything_rescaled_sparse"   # a separate camera set; must never be picked
    for d in (colmap, tracks, decoy, scene / "sparse/sparse_nvm"):
        d.mkdir(parents=True)
    for d in (colmap, tracks):
        (d / "cameras.txt").write_text(cams_txt)
        (d / "images.txt").write_text("\n".join(imgs_txt) + "\n")
    trimesh.PointCloud(np.array(xyz)).export(colmap / "points3D.ply")
    (tracks / "points3D.txt").write_text("\n".join(pts_txt) + "\n")
    (decoy / "cameras.txt").write_text(cams_txt)
    (decoy / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image_0000.jpg\n\n")
    (scene / "sparse/sparse_nvm/sparse.nvm").write_text("NVM_V3\n\n0\n\n0\n")
    (scene / "sparse/depth_scale.json").write_text(json.dumps({"scale": 1 / DEPTH_FILE_SCALE}))
    cfg = SceneConfig(wriva_root=str(root / "WRIVA"), scene="synth_scene", work_root=str(root / "out"),
                      lift_stride=2, min_object_voxels=20, views_per_object=4, eval_max_side=512,
                      surface_samples=40000, refine_max_evals=150, mvsam3d_root=str(root / "no_mvsam3d"))
    work = cfg.work_dir()
    for stem, lab in labels.items():
        inst = [lab == 0, lab == 1] if stem != "image_0000" else [(lab == 0) | (lab == 1)]
        inst = [m for m in inst if m.sum() > 20]
        save_instances(instances_path(work, "building", stem), np.stack(inst), np.ones(len(inst)), None, "building")
    return {"cfg": cfg, "labels": labels, "cams": dict(cams)}


def test_colmap_cameras_roundtrip(synth):
    colmap = find_colmap_dir(Path(synth["cfg"].wriva_root) / "synth_scene")
    assert colmap.parts[-2:] == ("sparse", "sparse")
    cams = load_cameras(colmap)
    assert len(cams) == 8
    for c in cams:
        np.testing.assert_allclose(c.w2c, synth["cams"][c.stem], atol=1e-9)
        assert abs(c.K[0, 2] - (W / 2 - 0.5)) < 1e-9


def test_inspect_recovers_depth_scale_from_points3d_ply(synth):
    rep = stages.stage_inspect(synth["cfg"])
    assert rep["colmap_dir"].endswith("sparse/sparse")
    assert rep["points3d"]["file"] == "points3D.ply" and not rep["points3d"]["has_track_ids"]
    assert rep["calibration_vs_colmap"]["source"].startswith("projected")
    assert abs(rep["calibration_vs_colmap"]["median_scale"] - 1 / DEPTH_FILE_SCALE) < 2e-3
    assert abs(rep["json_vs_calibration_median_ratio"] - 1) < 2e-3
    assert rep["depth_convention"]["verdict"].startswith("ok"), rep["depth_convention"]


def test_depth_scale_from_tracks(synth):
    from scene_mvsam3d.scene_data import SceneData, calibrate_depth_scales

    calib = calibrate_depth_scales(SceneData(dataclasses.replace(synth["cfg"], colmap_rel="sparse/colmap_with_tracks")))
    assert all(v["source"] == "tracks" for v in calib.values())
    assert abs(np.median([v["scale"] for v in calib.values()]) - 1 / DEPTH_FILE_SCALE) < 1e-3


def test_depth_along_the_ray_is_detected_and_converted(synth):
    import shutil

    from scene_mvsam3d.scene_data import ray_cosine

    cfg = synth["cfg"]
    src, dst = Path(cfg.wriva_root) / "synth_scene", Path(cfg.wriva_root) / "synth_scene_ray"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cos = ray_cosine(np.array([[F, 0, W / 2 - 0.5], [0, F, H / 2 - 0.5], [0, 0, 1]]), (H, W))
    for p in (src / "sparse/metric_z_depth_mve").glob("*_zdepth.npy"):
        z = np.load(p)
        np.save(dst / "sparse/metric_z_depth_mve" / p.name, np.where(z > 0, z / cos, 0).astype(np.float32))
    rep = stages.stage_inspect(dataclasses.replace(cfg, scene="synth_scene_ray"))
    assert "set depth_kind=ray" in rep["depth_convention"]["verdict"], rep["depth_convention"]
    assert "depth_kind=ray" in rep["recommendation"]
    rep = stages.stage_inspect(dataclasses.replace(cfg, scene="synth_scene_ray", depth_kind="ray"))
    assert rep["depth_convention"]["verdict"].startswith("ok"), rep["depth_convention"]
    assert abs(rep["calibration_vs_colmap"]["median_scale"] - 1 / DEPTH_FILE_SCALE) < 2e-3


def test_prepare_finds_ground_and_up(synth):
    info = stages.stage_prepare(synth["cfg"])
    assert np.dot(info["up"], [0, 0, 1]) > 0.999
    assert abs(info["ground_d"]) < 0.05
    scales = json.loads((synth["cfg"].work_dir() / "prepare/depth_scales.json").read_text())["scales"]
    assert abs(scales["__global__"] - 2.0) < 1e-3


def test_associate_two_buildings_and_split(synth):
    res = stages.stage_associate(synth["cfg"])
    objs = res["objects"]
    assert len(objs) == 2, [o["id"] for o in objs]
    assoc = Associations(synth["cfg"].work_dir())
    names = {}
    for o in objs:
        c = np.array(o["centroid"])
        names[o["id"]] = 0 if c[0] < 0 else 1
        assert 3 <= len(o["selected"]) <= 4
    for oid, k in names.items():
        for stem, lab in synth["labels"].items():
            gt = lab == k
            if gt.sum() < 50:
                continue
            m = assoc.mask(oid, stem)
            iou = (m & gt).sum() / (m | gt).sum()
            assert iou > 0.9, (oid, stem, iou)       # includes image_0000, where SAM merged both buildings


def _fake_mvsam3d(cfg, scene_cams, obj, world_box, rng):
    """Write result.glb + params.npz that placement must map back onto `world_box`."""
    import trimesh

    od = object_dir(cfg.work_dir(), obj["id"])
    out = od / "mvsam3d"
    out.mkdir(parents=True, exist_ok=True)
    w2c = scene_cams[obj["selected"][0]]
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    pose = {"rotation": q, "scale": np.array([3.0, 3.0, 3.0]), "translation": np.array([0.5, -0.2, 40.0])}
    p_cv = world_box.vertices @ w2c[:3, :3].T + w2c[:3, 3]
    p3d = p_cv @ np.linalg.inv(P3D_TO_CV).T
    v_rot = ((p3d - pose["translation"]) @ np.linalg.inv(quat_wxyz_to_matrix(q))) / pose["scale"]
    v_can = v_rot @ np.linalg.inv(Z_UP_TO_Y_UP.T)
    np.testing.assert_allclose(canonical_to_world(v_can, pose, w2c), world_box.vertices, atol=1e-6)
    trimesh.Trimesh(v_can, world_box.faces, process=False).export(out / "result.glb")
    np.savez(out / "params.npz", rotation=q[None], scale=pose["scale"][None], translation=pose["translation"][None])
    (od / "mvsam3d_result.json").write_text(json.dumps({"glb": str(out / "result.glb"),
                                                        "params": str(out / "params.npz"), "seconds": 0}))
    (od / "place_result.json").unlink(missing_ok=True)


def _box(lo, hi, height_scale=1.0, yaw_deg=0.0):
    import trimesh

    ext = (hi - lo) * np.array([1, 1, height_scale])
    T = np.eye(4)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (lo + hi) / 2 * np.array([1, 1, 0]) + np.array([0, 0, ext[2] / 2])
    return trimesh.creation.box(extents=ext, transform=T).subdivide().subdivide()


def test_object_input_pointmaps_land_on_the_building(synth):
    cfg = synth["cfg"]
    scene = stages._scene(cfg, cfg.work_dir())
    assoc = Associations(cfg.work_dir())
    obj = next(o for o in json.loads((cfg.work_dir() / "associate/objects.json").read_text())["objects"])
    od = object_dir(cfg.work_dir(), obj["id"])
    write_object_input(cfg, scene, assoc, obj["id"], obj["selected"], od)
    z = np.load(od / "da3_output.npz")
    assert z["pointmaps_sam3d"].shape == (len(obj["selected"]), 3, H, W)
    stem = obj["selected"][0]
    from PIL import Image

    alpha = np.asarray(Image.open(od / obj["id"] / f"{stem}.png"))[..., 3] > 127
    pm = z["pointmaps_sam3d"][0].transpose(1, 2, 0)[alpha]
    world = pm @ np.linalg.inv(z["extrinsics"][0])[:3, :3].T + np.linalg.inv(z["extrinsics"][0])[:3, 3]
    lo, hi = BOXES["A" if np.array(obj["centroid"])[0] < 0 else "B"]
    dist = np.linalg.norm(np.maximum(lo - world, 0) + np.maximum(world - hi, 0), axis=1)
    assert np.median(dist) < 0.05 and np.percentile(dist, 95) < 0.5


def test_place_exact_then_refine_a_too_tall_mesh(synth):
    cfg = synth["cfg"]
    rng = np.random.default_rng(1)
    objs = json.loads((cfg.work_dir() / "associate/objects.json").read_text())["objects"]
    for obj in objs:
        lo, hi = BOXES["A" if obj["centroid"][0] < 0 else "B"]
        _fake_mvsam3d(cfg, synth["cams"], obj, _box(lo, hi), rng)
    cfg.refine = False
    stages.stage_place(cfg)
    for obj in objs:
        r = json.loads((object_dir(cfg.work_dir(), obj["id"]) / "place_result.json").read_text())
        assert r["iou_all_placed"] > 0.9, r["per_view_placed"]
    for obj in objs:
        lo, hi = BOXES["A" if obj["centroid"][0] < 0 else "B"]
        _fake_mvsam3d(cfg, synth["cams"], obj, _box(lo, hi, height_scale=2.5, yaw_deg=12.0), rng)
    cfg.refine = True
    stages.stage_place(cfg)
    for obj in objs:
        r = json.loads((object_dir(cfg.work_dir(), obj["id"]) / "place_result.json").read_text())
        assert r["iou_all_placed"] < 0.8
        assert r["iou_all_refined"] > 0.88, (r["iou_all_placed"], r["iou_all_refined"], r["refine"])
        assert abs(r["refine"]["scale_v"] - 0.4) < 0.08, r["refine"]
        assert r["test_views"] and not set(r["test_views"]) & set(r["refine_views"] + r["input_views"])
        assert r["iou_test_refined"] > 0.88, r["iou_test_refined"]
    summary = stages.stage_assemble(cfg)
    assert summary["n_ok"] == 2 and summary["mean_iou_test_refined"] > 0.88
    assert (cfg.work_dir() / "scene/scene_objects_refined.glb").is_file()


FAKE_MVSAM3D = r'''
import argparse, sys
from pathlib import Path
import numpy as np, trimesh
p = argparse.ArgumentParser()
for a in ("input_path", "mask_prompt", "image_names", "da3_output", "model_tag", "seed", "stage1_steps",
          "stage2_steps", "decode_formats", "stage2_weight_source"):
    p.add_argument("--" + a)
a = p.parse_args()
assert Path("checkpoints/hf/pipeline.yaml").is_file(), "not started from the MV-SAM3D root"
inp = Path(a.input_path)
names = a.image_names.split(",")
z = np.load(a.da3_output)
assert [Path(str(f)).stem for f in z["image_files"]] == names
for n in names:
    assert (inp / "images" / f"{n}.png").is_file() and (inp / a.mask_prompt / f"{n}.png").is_file()
out = Path("visualization") / inp.name / a.mask_prompt / f"{inp.name}_{a.mask_prompt}_mv_20260101"
out.mkdir(parents=True)
trimesh.creation.box().export(out / "result.glb")
np.savez(out / "params.npz", rotation=np.array([[1.0, 0, 0, 0]]), scale=np.ones((1, 3)), translation=np.array([[0, 0, 40.0]]))
print("fake MV-SAM3D ok", file=sys.stderr)
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mvsam3d_stage_under_torchrun_two_ranks(synth, tmp_path):
    pytest.importorskip("torch.distributed")
    cfg = dataclasses.replace(synth["cfg"])
    root = tmp_path / "MV-SAM3D"
    (root / "checkpoints/hf").mkdir(parents=True)
    (root / "checkpoints/hf/pipeline.yaml").write_text("fake: true\n")
    (root / "run_inference_weighted.py").write_text(FAKE_MVSAM3D)
    cfg.mvsam3d_root = str(root)
    objs = json.loads((cfg.work_dir() / "associate/objects.json").read_text())["objects"]
    for obj in objs:
        (object_dir(cfg.work_dir(), obj["id"]) / "mvsam3d_result.json").unlink(missing_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg.to_dict()))          # JSON is valid YAML
    # same torchrun flags as run_trinity.sh
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
           "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
           str(ROOT / "run.py"), "mvsam3d", "--config", str(cfg_path)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path,
                         start_new_session=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="3,5", OMP_NUM_THREADS="1"))
    try:
        stdout, stderr = p.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)                     # torchrun and both ranks
        stdout, stderr = p.communicate()
        pytest.fail(f"torchrun timed out:\n{stdout[-3000:]}\n{stderr[-3000:]}")
    assert p.returncode == 0, stdout[-3000:] + stderr[-3000:]
    assert "process group: gloo" in stdout and "across 2 rank(s)" in stdout, stdout
    gpus = set()
    for obj in objs:
        od = object_dir(cfg.work_dir(), obj["id"])
        res = json.loads((od / "mvsam3d_result.json").read_text())
        assert "error" not in res, res
        assert Path(res["glb"]).is_file() and Path(res["params"]).is_file()
        assert Path(res["glb"]).parent == od / "mvsam3d"
        gpus.add(res["gpu"])
    assert gpus <= {"3", "5"}
    summary = json.loads((cfg.work_dir() / "mvsam3d_summary.json").read_text())
    assert sorted(summary["done"]) == sorted(o["id"] for o in objs) and not summary["failed"]
    assert len(summary["per_rank"]) == 2
    assert not (root / "visualization").exists() or not any((root / "visualization").iterdir())


@pytest.mark.parametrize("layout", ["label_png", "per_instance_png"])
def test_import_masks(synth, tmp_path, layout):
    from PIL import Image

    src = tmp_path / "mehar_masks"
    src.mkdir()
    for stem, lab in synth["labels"].items():
        if layout == "label_png":
            Image.fromarray((lab + 1).astype(np.uint8)).save(src / f"{stem}.png")   # 0 bg, 1 = A, 2 = B
        else:
            for k in (0, 1):
                Image.fromarray((lab == k).astype(np.uint8) * 255).save(src / f"{stem}_{k}.png")
    cfg = dataclasses.replace(synth["cfg"], run_name=f"import_{layout}", masks_import_dir=str(src),
                              masks_import_pattern="{stem}" if layout == "label_png" else "{stem}_*")
    stages.stage_import_masks(cfg)
    stats = json.loads((cfg.work_dir() / "import_masks_summary.json").read_text())
    assert stats["imported"] == 8 and not stats["missing"]
    for stem, lab in synth["labels"].items():
        got = load_instances(instances_path(cfg.work_dir(), "building", stem))["masks"]
        want = [lab == k for k in (0, 1) if (lab == k).mean() >= cfg.mask_min_area_frac]
        assert len(got) == len(want), stem
        for g, w in zip(got, want):
            assert (g == w).all()


def test_config_file_loads_with_overrides():
    from scene_mvsam3d.config import load_config

    cfg = load_config(ROOT / "configs/jhuapl_full.yaml",
                      ["scene=jhuapl", "object_timeout_min=60.5", "prompts=building,tree", "refine=false",
                       "masks_import_pattern={stem}_*", "max_objects=3"])
    assert cfg.scene == "jhuapl" and cfg.object_timeout_min == 60.5 and cfg.prompts == ["building", "tree"]
    assert cfg.refine is False and cfg.masks_import_pattern == "{stem}_*" and cfg.max_objects == 3
    assert cfg.image_glob == "*" and cfg.refine_dofs == ["yaw", "tx", "ty", "s_h", "s_v"]
    assert cfg.work_dir() == Path("/scratch/tnishida/outputs/full_jhuapl_mvsam3d/jhuapl")


def test_claim_is_exclusive_and_stale_claims_expire(tmp_path):
    p = tmp_path / "obj/.claim"
    assert claim(p, stale_after_s=3600)
    assert not claim(p, stale_after_s=3600)
    import os

    os.utime(p, (0, 0))
    assert claim(p, stale_after_s=60)
