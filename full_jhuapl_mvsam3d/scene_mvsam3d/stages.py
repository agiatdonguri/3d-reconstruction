"""The pipeline stages. Each reads the previous stage's files under <work_dir> and can be rerun on its own."""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .associate import Associations, run_association, select_views
from .colmap_io import find_colmap_dir, read_points3d
from .config import SceneConfig
from .dist import all_gather_object, claim, dist_info, physical_gpu, print_all
from .geometry import fit_plane_ransac, unproject_pixels
from .masks import Sam3Segmenter, import_masks, instances_path, save_instances
from .safety import assert_writable_dir
from .scene_data import SceneData, calibrate_depth_scales, parse_depth_scale_json, resolve_depth_scales


def _log(msg: str) -> None:
    d = dist_info()
    print_all(f"[rank {d.rank}/{d.world_size}] {msg}" if d.world_size > 1 else msg, flush=True)


def _summarize(name: str, work_dir: Path, mine: dict) -> None:
    """Gather every rank's results (all ranks must reach this) and let rank 0 write the stage summary."""
    every = all_gather_object(mine)
    if not dist_info().is_main:
        return
    merged = {k: sum((r.get(k, []) for r in every), []) for k in sorted({k for r in every for k in r})}
    merged["per_rank"] = [{k: len(v) for k, v in r.items()} for r in every]
    (work_dir / f"{name}_summary.json").write_text(json.dumps(merged, indent=1, default=float))
    print(f"== {name}: " + ", ".join(f"{len(v)} {k}" for k, v in merged.items() if k != "per_rank")
          + f" across {len(every)} rank(s) → {work_dir / f'{name}_summary.json'}", flush=True)


def _work_dir(cfg: SceneConfig) -> Path:
    return assert_writable_dir(cfg.work_dir(), label="work_dir")


def _scene(cfg: SceneConfig, work_dir: Path) -> SceneData:
    p = work_dir / "prepare" / "depth_scales.json"
    scales = json.loads(p.read_text())["scales"] if p.is_file() else None
    return SceneData(cfg, scales)


def _scene_info(work_dir: Path) -> dict:
    p = work_dir / "prepare" / "scene.json"
    return json.loads(p.read_text()) if p.is_file() else {}


# --------------------------------------------------------------------------------------------- inspect
def stage_doctor(cfg: SceneConfig) -> list[dict]:
    """Can this environment run the pipeline? Run it in both conda envs (mvsam3d and sam3)."""
    import importlib
    import os
    import subprocess
    import sys

    checks: list[dict] = []

    def check(name: str, fn, needed_for: str = "all") -> None:
        try:
            ok, detail = True, fn() or ""
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        checks.append({"ok": ok, "check": name, "needed_for": needed_for, "detail": str(detail)[:300]})

    def must(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)

    root = cfg.scene_root()
    check("scene folder", lambda: must(root))
    check("images", lambda: f"{len(list(Path(must(root / cfg.images_rel)).iterdir()))} files")
    check("COLMAP poses (cameras.txt + images.txt)", lambda: str(find_colmap_dir(root, cfg.colmap_rel)))
    check("depth", lambda: f"{len(list(Path(must(root / cfg.depth_rel)).glob('*.npy')))} .npy files")
    check("registered images with depth",
          lambda: (lambda s: f"{sum(s.depth_path(c) is not None for c in s.cameras)} / {len(s.cameras)}")(SceneData(cfg)))
    check("work_dir writable", lambda: str(_work_dir(cfg)))

    def gpus() -> str:
        import torch

        n = torch.cuda.device_count()
        names = {torch.cuda.get_device_name(i) for i in range(n)}
        return f"torch {torch.__version__}, {n} GPU(s) visible {sorted(names)}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
    check("torch + GPUs", gpus)
    for mod in ("numpy", "scipy", "PIL", "yaml", "trimesh"):
        check(f"python package {mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", "ok"))
    check("matplotlib (figures only)", lambda: importlib.import_module("matplotlib").__version__, "figures")

    mv = Path(cfg.mvsam3d_root)
    check("MV-SAM3D run_inference_weighted.py", lambda: must(mv / "run_inference_weighted.py"), "mvsam3d")
    check(f"MV-SAM3D checkpoints/{cfg.model_tag}/pipeline.yaml",
          lambda: must(mv / "checkpoints" / cfg.model_tag / "pipeline.yaml"), "mvsam3d")

    def mv_import() -> str:
        py = cfg.mvsam3d_python or sys.executable
        code = "import sys; sys.path.append('notebook'); from inference import Inference; print('ok')"
        r = subprocess.run([py, "-c", code], cwd=must(mv), capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, CUMM_DISABLE_JIT="1", SPCONV_DISABLE_JIT="1"))
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "failed")
        return f"{py}: notebook/inference.py imports (sam3d_objects, pytorch3d, kaolin, spconv …)"
    check("MV-SAM3D imports", mv_import, "mvsam3d")

    def sam3() -> str:
        if cfg.sam3_root and cfg.sam3_root not in sys.path:
            sys.path.insert(0, cfg.sam3_root)
        importlib.import_module("sam3.model_builder")
        return "sam3 importable"
    check("SAM 3", sam3, "masks")

    def hf_token() -> str:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("not set: Hugging Face downloads (facebook/sam3, facebook/sam-3d-objects) need it unless cached")
        return "set"
    check("HF_TOKEN in environment", hf_token, "downloads")
    if cfg.masks_import_dir:
        check("masks_import_dir", lambda: must(Path(cfg.masks_import_dir)), "import_masks")

    width = max(len(c["check"]) for c in checks)
    for c in checks:
        print(f"{'✓' if c['ok'] else '✗'} {c['check']:<{width}}  [{c['needed_for']}]  {c['detail']}")
    out = _work_dir(cfg) / "doctor"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"doctor_{Path(sys.prefix).name}.json").write_text(json.dumps(checks, indent=1))
    return checks


def stage_inspect(cfg: SceneConfig) -> dict:
    """Cheap, CPU only: what the scene contains, and which depth-scale mode makes depth agree with COLMAP."""
    work_dir = _work_dir(cfg)
    scene = SceneData(cfg)
    rep: dict = {"scene": str(scene.root), "colmap_dir": str(scene.sparse_dir),
                 "n_registered_images": len(scene.cameras)}
    points = read_points3d(scene.sparse_dir)
    rep["points3d"] = {"file": "points3D.txt" if points[0] is not None else
                       ("points3D.ply" if points[1] is not None else None),
                       "n_points": 0 if points[1] is None else len(points[1]),
                       "has_track_ids": points[0] is not None}
    sizes = sorted({(c.width, c.height) for c in scene.cameras})
    rep["image_sizes"] = sizes[:5]
    rep["fx_range"] = [float(min(c.K[0, 0] for c in scene.cameras)), float(max(c.K[0, 0] for c in scene.cameras))]
    have = [c for c in scene.cameras if scene.depth_path(c) is not None]
    rep["depth_dir"] = str(scene.depth_dir)
    rep["n_with_depth"] = len(have)
    rep["missing_depth_examples"] = [c.name for c in scene.cameras if scene.depth_path(c) is None][:5]
    if have:
        probe = have[:: max(1, len(have) // 5)][:5]
        rep["depth_examples"] = []
        for c in probe:
            d = scene.load_depth_raw(c)
            v = d[d > 0]
            ex = {"image": c.name, "file": scene.depth_path(c).name, "shape": list(d.shape),
                  "valid_frac": float((d > 0).mean()),
                  "p5_p50_p95": [float(x) for x in np.percentile(v, [5, 50, 95])] if v.size else None,
                  "camera_height_above_scene_center": float(np.linalg.norm(c.center))}
            intr = scene.depth_dir / f"{c.stem}_intrinsics.json"
            if intr.is_file():
                ex["intrinsics_json"] = json.loads(intr.read_text())
                ex["colmap_fx_at_depth_res"] = float(c.scaled_K(d.shape[1], d.shape[0])[0, 0])
            rep["depth_examples"].append(ex)
    calib_sub = SceneData(cfg)
    calib_sub.cameras = have[:: max(1, len(have) // 40)][:40]
    calib = calibrate_depth_scales(calib_sub, points=points)
    s = np.array([v["scale"] for v in calib.values() if np.isfinite(v.get("scale", np.nan))])
    rep["calibration_vs_colmap"] = {
        "n_images": int(s.size), "median_scale": float(np.median(s)) if s.size else None,
        "p10_p90": [float(x) for x in np.percentile(s, [10, 90])] if s.size else None,
        "source": next(iter(calib.values()), {}).get("source"),
        "meaning": "multiply the depth files by this to match the COLMAP poses (1.0 = already consistent)",
    }
    slopes = np.array([v.get("cos_slope", np.nan) for v in calib.values()], float)
    slopes = slopes[np.isfinite(slopes)]
    m = float(np.median(slopes)) if slopes.size else None
    if m is None:
        verdict = "could not tell (too few points, or too narrow a field of view)"
    elif abs(m) < 0.3:
        verdict = f"ok: consistent with depth_kind={cfg.depth_kind}"
    elif cfg.depth_kind == "z" and abs(m - 1) < 0.3:
        verdict = "the files hold distance along the ray, not z-depth: set depth_kind=ray"
    elif cfg.depth_kind == "ray" and abs(m + 1) < 0.3:
        verdict = "the files are z-depth: set depth_kind=z"
    else:
        verdict = f"unclear (slope {m:.2f}): look at the depth files before going on"
    rep["depth_convention"] = {"depth_kind": cfg.depth_kind, "median_cos_slope": m, "n_images": int(slopes.size),
                               "verdict": verdict,
                               "meaning": "slope of log(COLMAP z / file depth) vs log cos(angle off the optical "
                                          "axis): 0 = matches depth_kind, ±1 = the other convention"}
    js = scene.root / cfg.depth_scale_json_rel
    if js.is_file():
        rep["depth_scale_json_preview"] = js.read_text()[:400]
        try:
            parsed = parse_depth_scale_json(js, [c.stem for c in scene.cameras])
            vals = [parsed.get(c.stem, parsed.get("__global__")) for c in calib_sub.cameras]
            rep["depth_scale_json_parsed"] = {"n": len(parsed), "example": dict(list(parsed.items())[:3])}
            if s.size and all(v is not None for v in vals):
                rep["json_vs_calibration_median_ratio"] = float(np.median(
                    [calib[c.stem]["scale"] / v for c, v in zip(calib_sub.cameras, vals)
                     if np.isfinite(calib[c.stem]["scale"])]))
        except Exception as e:
            rep["depth_scale_json_parsed"] = f"could not parse: {e}"
    med = rep["calibration_vs_colmap"]["median_scale"]
    if med is None:
        rep["recommendation"] = "no sparse points to calibrate against — check depth_scale.json, use json or none"
    elif abs(med - 1) < 0.02:
        rep["recommendation"] = "depth files already agree with COLMAP: depth_scale_mode=none (auto is equivalent)"
    elif abs(rep.get("json_vs_calibration_median_ratio", 0) - 1) < 0.03:
        rep["recommendation"] = "depth_scale.json makes depth agree with COLMAP: depth_scale_mode=json (or auto)"
    else:
        rep["recommendation"] = f"depth is off by ×{med:.3f} vs COLMAP: depth_scale_mode=auto"
    if not verdict.startswith("ok"):
        rep["recommendation"] += f"; depth convention: {verdict}"
    out = work_dir / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return rep


# --------------------------------------------------------------------------------------------- prepare
def stage_prepare(cfg: SceneConfig, max_points: int = 400_000) -> dict:
    """Depth scales, ground plane / up axis, scene extent, background point cloud."""
    import trimesh

    work_dir = _work_dir(cfg)
    out = work_dir / "prepare"
    out.mkdir(parents=True, exist_ok=True)
    scene = SceneData(cfg)
    calib = calibrate_depth_scales(scene) if cfg.depth_scale_mode == "auto" else None
    scales = resolve_depth_scales(scene, cfg.depth_scale_mode, calib)
    (out / "depth_scales.json").write_text(json.dumps({"mode": cfg.depth_scale_mode, "scales": scales,
                                                       "calibration": calib}, indent=1, default=float))
    scene.depth_scales = scales
    rng = np.random.default_rng(0)
    pts, cols = [], []
    per_view = max(2000, max_points // max(1, len(scene.cameras)))
    for cam in scene.cameras:
        if scene.depth_path(cam) is None:
            continue
        d, Kd = scene.load_depth(cam)
        vv, uu = np.nonzero(d > 0)
        if vv.size == 0:
            continue
        pick = rng.choice(vv.size, min(per_view, vv.size), replace=False)
        vv, uu = vv[pick], uu[pick]
        pts.append(unproject_pixels(uu.astype(float), vv.astype(float), d[vv, uu], Kd, cam.c2w))
        rgb = scene.load_image(cam, max_side=max(d.shape))
        ry = np.clip(np.round((vv + 0.5) * rgb.shape[0] / d.shape[0] - 0.5).astype(int), 0, rgb.shape[0] - 1)
        rx = np.clip(np.round((uu + 0.5) * rgb.shape[1] / d.shape[1] - 0.5).astype(int), 0, rgb.shape[1] - 1)
        cols.append(rgb[ry, rx])
    if not pts:
        raise RuntimeError("no depth could be lifted — run `inspect` and check depth_rel / depth_patterns")
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    extent = float(np.linalg.norm(hi - lo))
    n, dd, inl = fit_plane_ransac(pts, thresh=0.005 * extent)
    cams = np.array([c.center for c in scene.cameras])
    if np.median(cams @ n + dd) < 0:
        n, dd = -n, -dd
    info = {
        "up": n.tolist(), "ground_d": float(dd), "ground_inlier_frac": float(inl.mean()),
        "origin": np.median(pts, 0).tolist(), "extent": extent, "bbox_p2": lo.tolist(), "bbox_p98": hi.tolist(),
        "auto_voxel": float(np.clip(extent / 400.0, 0.05, 2.0)), "n_images": len(scene.cameras),
        "camera_height_above_ground_median": float(np.median(cams @ n + dd)),
    }
    (out / "scene.json").write_text(json.dumps(info, indent=1))
    keys = np.floor((pts - lo) / info["auto_voxel"]).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    trimesh.PointCloud(pts[first], colors=cols[first]).export(out / "background_points.ply")
    print(json.dumps(info, indent=1))
    return info


# --------------------------------------------------------------------------------------------- masks
def stage_masks(cfg: SceneConfig) -> None:
    """SAM 3 instances for every prompt in every view (torchrun: views split across ranks)."""
    import torch
    from PIL import Image

    work_dir = _work_dir(cfg)
    d = dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(d.local_rank % torch.cuda.device_count())
    scene = SceneData(cfg)
    mine = scene.cameras[d.rank::d.world_size]
    todo = [(c, p) for c in mine for p in cfg.prompts if not instances_path(work_dir, p, c.stem).is_file()]
    _log(f"masks: {len(todo)} (view, prompt) pairs to do, {len(mine) * len(cfg.prompts) - len(todo)} cached")
    done, failed = [], []
    seg = Sam3Segmenter(cfg.sam3_checkpoint, cfg.sam3_root, cfg.sam3_score_thresh) if todo else None
    t0 = time.time()
    for i, (cam, prompt) in enumerate(todo):
        try:
            img = Image.fromarray(scene.load_image(cam))
            masks, scores, boxes = seg(img, prompt)
            keep = (scores >= cfg.sam3_score_thresh) & (masks.reshape(len(masks), -1).mean(1) >= cfg.mask_min_area_frac)
            save_instances(instances_path(work_dir, prompt, cam.stem), masks[keep], scores[keep], boxes[keep], prompt)
            done.append(f"{prompt}|{cam.stem}")
        except Exception as e:
            failed.append(f"{prompt}|{cam.stem}: {type(e).__name__}: {e}")
            _log(f"  ✗ {cam.name} '{prompt}': {e}")
            continue
        if i % 10 == 0:
            _log(f"  {i + 1}/{len(todo)} {cam.name} '{prompt}': {int(keep.sum())} instances "
                 f"[{time.time() - t0:.0f}s]")
    _summarize("masks", work_dir, {"done": done, "failed": failed})


def stage_import_masks(cfg: SceneConfig) -> None:
    """Use existing instance masks (e.g. Mehar's SAM 3 "building" masks) instead of running SAM 3."""
    work_dir = _work_dir(cfg)
    if not cfg.masks_import_dir:
        raise ValueError("import_masks needs masks_import_dir (e.g. --set masks_import_dir=/data3/tnishida/…)")
    if cfg.masks_import_prompt not in cfg.prompts:
        raise ValueError(f"masks_import_prompt {cfg.masks_import_prompt!r} must be one of prompts {cfg.prompts} "
                         "(associate only reads masks of the configured prompts)")
    stats = import_masks(cfg, SceneData(cfg), work_dir)
    (work_dir / "import_masks_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in stats.items()}, indent=1))


# --------------------------------------------------------------------------------------------- associate
def stage_associate(cfg: SceneConfig) -> dict:
    work_dir = _work_dir(cfg)
    scene = _scene(cfg, work_dir)
    info = _scene_info(work_dir)
    result = run_association(cfg, scene, work_dir, info)
    assoc = Associations(work_dir)
    for obj in result["objects"]:
        stats = select_views(cfg, scene, assoc, obj["id"])
        obj["views"] = stats
        obj["selected"] = [s["view"] for s in sorted((s for s in stats if "selected_rank" in s),
                                                     key=lambda s: s["selected_rank"])]
        print(f"  {obj['id']}: {obj['n_voxels']} voxels, masks in {len(stats)} views, "
              f"MV-SAM3D views {obj['selected']}")
    (work_dir / "associate" / "objects.json").write_text(json.dumps(result, indent=1))
    try:
        _figure_topdown(result, info, work_dir / "associate" / "objects_topdown.png")
        _figure_view_overlays(scene, assoc, result, work_dir / "associate")
    except ImportError as e:
        print(f"⚠ association figures skipped ({e}); objects.json is complete")
    return result


def _figure_topdown(result: dict, info: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .geometry import orthonormal_frame

    axes = orthonormal_frame(np.asarray(info.get("up", [0, 0, 1.0])))
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")
    for i, obj in enumerate(result["objects"]):
        c = np.load(path.parent / f"voxels_{obj['id']}.npy")
        p = c @ axes[:2].T
        ax.scatter(p[:, 0], p[:, 1], s=1, color=cmap(i % 20))
        m = p.mean(0)
        ax.text(m[0], m[1], obj["id"], fontsize=7)
    ax.set_aspect("equal")
    ax.set_title("objects from multi-view association (top-down, ground frame)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _figure_view_overlays(scene: SceneData, assoc: Associations, result: dict, out_dir: Path, n: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {}
    for a in assoc.assignments:
        counts.setdefault(a["view"], set()).add(a["object"])
    stems = sorted(counts, key=lambda s: -len(counts[s]))[:n]
    cmap = plt.get_cmap("tab20")
    for stem in stems:
        cam = scene.by_stem[stem]
        rgb = scene.load_image(cam, 1024).astype(np.float32) / 255
        h, w = rgb.shape[:2]
        over = rgb.copy()
        for obj in result["objects"]:
            if obj["index"] in counts[stem]:
                m = assoc.mask(obj["id"], stem, (w, h))
                over[m] = 0.45 * over[m] + 0.55 * np.array(cmap(obj["index"] % 20)[:3])
        plt.imsave(out_dir / f"view_{stem}.png", np.clip(over, 0, 1))


# --------------------------------------------------------------------------------------------- mvsam3d
def _objects(work_dir: Path) -> list[dict]:
    return json.loads((work_dir / "associate" / "objects.json").read_text())["objects"]


def stage_mvsam3d(cfg: SceneConfig) -> None:
    """One MV-SAM3D run per object; ranks claim objects from a shared folder (biggest first)."""
    from .mvsam3d_runner import object_dir, run_mvsam3d, write_object_input

    work_dir = _work_dir(cfg)
    d = dist_info()
    gpu = physical_gpu(d.local_rank)
    scene = _scene(cfg, work_dir)
    assoc = Associations(work_dir)
    objs = sorted((o for o in _objects(work_dir) if o.get("selected")), key=lambda o: -o["n_voxels"])
    if cfg.max_objects > 0:
        objs = objs[: cfg.max_objects]
    done, failed = [], []
    for obj in objs:
        od = object_dir(work_dir, obj["id"])
        res_path = od / "mvsam3d_result.json"
        if res_path.is_file() and "error" not in json.loads(res_path.read_text()):
            continue
        if not claim(od / ".claim_mvsam3d", stale_after_s=cfg.object_timeout_min * 60 + 600):
            continue
        _log(f"mvsam3d {obj['id']} on GPU {gpu}: views {obj['selected']}")
        try:
            write_object_input(cfg, scene, assoc, obj["id"], obj["selected"], od)
            res = run_mvsam3d(cfg, od, obj["id"], obj["selected"], gpu)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        res_path.write_text(json.dumps(res, indent=1, default=str))
        (od / ".claim_mvsam3d").unlink(missing_ok=True)
        if "error" in res:
            failed.append(f"{obj['id']}: {res['error']} (log: {res.get('log', '—')})")
            _log(f"  ✗ {failed[-1]}")
        else:
            done.append(obj["id"])
            _log(f"  ✓ {obj['id']} in {res['seconds']:.0f}s")
    _summarize("mvsam3d", work_dir, {"done": done, "failed": failed})


# --------------------------------------------------------------------------------------------- place
def stage_place(cfg: SceneConfig) -> None:
    """World placement of every finished object, held-out IoU, then the optional refinement."""
    import torch

    from .mvsam3d_runner import object_dir

    work_dir = _work_dir(cfg)
    d = dist_info()
    device = torch.device(f"cuda:{d.local_rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    scene = _scene(cfg, work_dir)
    assoc = Associations(work_dir)
    up = np.asarray(_scene_info(work_dir).get("up", [0, 0, 1.0]))
    done, failed = [], []
    for obj in _objects(work_dir):
        od = object_dir(work_dir, obj["id"])
        rp = od / "mvsam3d_result.json"
        if not rp.is_file() or (od / "place_result.json").is_file():
            continue
        res = json.loads(rp.read_text())
        if "error" in res or not claim(od / ".claim_place", stale_after_s=3600):
            continue
        try:
            out = _place_one(cfg, scene, assoc, obj, od, res, up, device)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        (od / "place_result.json").write_text(json.dumps(out, indent=1, default=float))
        (od / ".claim_place").unlink(missing_ok=True)
        if "error" in out:
            failed.append(f"{obj['id']}: {out['error']}")
            _log(f"  ✗ place {failed[-1]}")
        else:
            done.append(obj["id"])
            _log(f"  ✓ {obj['id']}: IoU on test views {out['iou_test_placed']:.3f} placed"
                 + (f" → {out['iou_test_refined']:.3f} refined" if "iou_test_refined" in out else "")
                 + f" (all views {out['iou_all_placed']:.3f}"
                 + (f" → {out['iou_all_refined']:.3f})" if "iou_all_refined" in out else ")"))
    _summarize("place", work_dir, {"done": done, "failed": failed})


def _place_one(cfg: SceneConfig, scene: SceneData, assoc: Associations, obj: dict, od: Path, res: dict,
               up: np.ndarray, device) -> dict:
    import torch
    import trimesh

    from .placement import apply_to_scene, load_pose, load_world_scene
    from .refine import apply, object_frame, refine_object
    from .silhouette import build_eval_views, mean_iou, n_surface_samples

    stems_in = obj["selected"]
    ref = scene.by_stem[stems_in[0]]
    params = res.get("optimized_params") if cfg.mvsam3d_pose_optimization and res.get("optimized_params") else res["params"]
    pose = load_pose(params)
    world_scene, mesh = load_world_scene(res["glb"], pose, ref.w2c)
    world_scene.export(od / "placed.glb")
    views_all = assoc.views_of(obj["id"])
    # heldout: not given to MV-SAM3D. test: every 3rd of those, also never used by the refinement.
    heldout = [s for s in views_all if s not in stems_in]
    test = heldout[::3]
    ev_all = build_eval_views(scene, assoc, obj["id"], views_all, cfg.eval_max_side, device)
    by = {v.stem: v for v in ev_all}
    n = n_surface_samples(ev_all, cfg.surface_samples)
    try:
        samples = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    except TypeError:   # trimesh < 3.21 has no seed argument
        samples = trimesh.sample.sample_surface(mesh, n)[0]
    pts_np = np.asarray(samples, np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float32, device=device)

    def score(p):
        a, per = mean_iou(p, ev_all, cfg.occlusion_tol)
        pv = dict(zip(views_all, per))
        mean = lambda stems: float(np.mean([pv[s] for s in stems if s in pv])) if stems else float("nan")
        return {"all": a, "heldout": mean(heldout), "test": mean(test), "input": mean(stems_in)}, pv

    m, pv = score(pts)
    out = {"object": obj["id"], "prompt": obj["prompt"], "params": params, "n_views": len(views_all),
           "input_views": stems_in, "n_heldout": len(heldout), "test_views": test,
           **{f"iou_{k}_placed": v for k, v in m.items()}, "per_view_placed": pv,
           "extent_world": (mesh.bounds[1] - mesh.bounds[0]).tolist()}
    if cfg.refine:
        frame = object_frame(pts_np, up)
        fit = [s for s in views_all if s not in test]
        opt_stems = sorted(fit, key=lambda s: -by[s].mask.sum().item())[: cfg.refine_max_views]
        r = refine_object(pts, [by[s] for s in opt_stems], frame, list(cfg.refine_dofs), cfg.refine_max_evals,
                          cfg.refine_max_scale, cfg.occlusion_tol)
        th = r["theta"]
        m2, pv2 = score(apply(pts, th, frame))
        apply_to_scene(world_scene, lambda v: apply(v, th, frame)).export(od / "refined.glb")
        out.update({"refine": r, "refine_views": opt_stems, **{f"iou_{k}_refined": v for k, v in m2.items()},
                    "per_view_refined": pv2,
                    "frame": {"origin": frame.origin.tolist(), "axes": frame.axes.tolist(), "size": frame.size}})
    return out


# --------------------------------------------------------------------------------------------- assemble
def stage_assemble(cfg: SceneConfig) -> dict:
    """Merge every placed / refined object into scene GLBs and write the report."""
    import trimesh

    from .mvsam3d_runner import object_dir

    work_dir = _work_dir(cfg)
    out = work_dir / "scene"
    out.mkdir(parents=True, exist_ok=True)
    rows, placed, refined = [], trimesh.Scene(), trimesh.Scene()
    for obj in _objects(work_dir):
        od = object_dir(work_dir, obj["id"])
        mv = json.loads((od / "mvsam3d_result.json").read_text()) if (od / "mvsam3d_result.json").is_file() else {}
        pr = json.loads((od / "place_result.json").read_text()) if (od / "place_result.json").is_file() else {}
        status = "not run" if not mv else ("mvsam3d failed" if "error" in mv else
                                           ("place failed" if "error" in pr else ("ok" if pr else "not placed")))
        rows.append({
            "object": obj["id"], "prompt": obj["prompt"], "status": status, "n_mask_views": len(obj.get("views", [])),
            "input_views": "|".join(obj.get("selected", [])), "mvsam3d_s": mv.get("seconds"),
            "n_test_views": len(pr.get("test_views", [])),
            "iou_test_placed": pr.get("iou_test_placed"), "iou_test_refined": pr.get("iou_test_refined"),
            "iou_heldout_placed": pr.get("iou_heldout_placed"),
            "iou_all_placed": pr.get("iou_all_placed"), "iou_all_refined": pr.get("iou_all_refined"),
            "scale_h": pr.get("refine", {}).get("scale_h"), "scale_v": pr.get("refine", {}).get("scale_v"),
            "yaw_deg": pr.get("refine", {}).get("theta", {}).get("yaw"),
        })
        for name, sc in (("placed.glb", placed), ("refined.glb", refined)):
            if (od / name).is_file():
                g = trimesh.load(od / name, process=False)
                for gname, geom in (g.geometry.items() if isinstance(g, trimesh.Scene) else [("mesh", g)]):
                    sc.add_geometry(geom, node_name=f"{obj['id']}/{gname}")
    if placed.geometry:
        placed.export(out / "scene_objects_placed.glb")
    if refined.geometry:
        refined.export(out / "scene_objects_refined.glb")
    bg = work_dir / "prepare" / "background_points.ply"
    best = refined if refined.geometry else placed
    if bg.is_file() and best.geometry:
        full = best.copy()
        full.add_geometry(trimesh.load(bg, process=False), node_name="background_points")
        full.export(out / "scene_with_background.glb")
    with open(out / "objects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["object"])
        wr.writeheader()
        wr.writerows(rows)

    def avg(k):
        v = [r[k] for r in rows if isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        return float(np.mean(v)) if v else None

    summary = {"n_objects": len(rows), "n_ok": sum(r["status"] == "ok" for r in rows),
               "mean_iou_test_placed": avg("iou_test_placed"), "mean_iou_test_refined": avg("iou_test_refined"),
               "mean_iou_heldout_placed": avg("iou_heldout_placed"),
               "mean_iou_all_placed": avg("iou_all_placed"), "mean_iou_all_refined": avg("iou_all_refined"),
               "status_counts": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}},
               "meaning": {"test": "views used neither by MV-SAM3D nor by the refinement (the fair number)",
                           "heldout": "views not given to MV-SAM3D", "all": "every view that sees the object",
                           "placed": "MV-SAM3D's own pose, no fitting", "refined": "after the yaw/x/y/scale fit"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


STAGES = {
    "doctor": stage_doctor, "inspect": stage_inspect, "prepare": stage_prepare, "masks": stage_masks,
    "import_masks": stage_import_masks,
    "associate": stage_associate, "mvsam3d": stage_mvsam3d, "place": stage_place, "assemble": stage_assemble,
}
