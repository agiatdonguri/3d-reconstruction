"""Multi-view instance association in 3D.

SAM 3 returns per-view instances with no identity across views (every building is just "building"). With
metric depth and the COLMAP poses every mask pixel has a world position, so:
1. lift each instance's pixels to world points and voxelize them;
2. keep voxels that at least `min_view_support` views agree on (single-view mistakes drop out);
3. split the kept voxels into connected components — one per object (prompts are kept apart);
4. give each instance to the component most of its points land in, splitting instances that cover two
   objects (SAM sometimes merges adjacent buildings into one mask).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import SceneConfig
from .geometry import project, unproject_pixels
from .masks import instances_path, load_instances, resize_mask, slug
from .scene_data import SceneData

_OFF = 1 << 20
_HALF_NEIGHBOURS = np.array([d for d in np.ndindex(3, 3, 3) if d > (1, 1, 1)], np.int64) - 1


def encode(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def decode(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.int64)
    return np.stack([(c >> 42) & 0x1FFFFF, (c >> 21) & 0x1FFFFF, c & 0x1FFFFF], 1) - _OFF


@dataclass
class Lifted:
    prompt: str
    view: int
    inst: int
    pts: Optional[np.ndarray]   # world position of every lifted sample (dropped once voxelized)
    rows: np.ndarray            # sample-grid row / col of every lifted sample
    cols: np.ndarray
    grid_hw: tuple[int, int]
    depth_hw: tuple[int, int]
    area_frac: float
    score: float
    spacing: float              # metres between neighbouring samples on the surface (depth × stride / f)
    codes: Optional[np.ndarray] = None


def lift_view_instances(scene: SceneData, cam_idx: int, prompt: str, masks: np.ndarray, scores: np.ndarray,
                        stride: int) -> list[Lifted]:
    cam = scene.cameras[cam_idx]
    depth, Kd = scene.load_depth(cam)
    hd, wd = depth.shape
    rows, cols = np.arange(0, hd, stride), np.arange(0, wd, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    z = depth[rr, cc]
    valid = z > 0
    h_img, w_img = masks.shape[1:]
    vi = np.clip(np.round((rr + 0.5) * h_img / hd - 0.5).astype(int), 0, h_img - 1)
    ui = np.clip(np.round((cc + 0.5) * w_img / wd - 0.5).astype(int), 0, w_img - 1)
    out = []
    for k in range(len(masks)):
        inside = masks[k][vi, ui] & valid
        if inside.sum() < 8:
            continue
        gi, gj = np.nonzero(inside)
        zs = z[gi, gj]
        pts = unproject_pixels(cc[gi, gj].astype(np.float64), rr[gi, gj].astype(np.float64), zs, Kd, cam.c2w)
        out.append(Lifted(prompt, cam_idx, k, pts.astype(np.float32), gi.astype(np.int32), gj.astype(np.int32),
                          rr.shape, (hd, wd), float(masks[k].mean()), float(scores[k]) if k < len(scores) else 1.0,
                          float(np.median(zs)) * stride / float(np.sqrt(Kd[0, 0] * Kd[1, 1]))))
    return out


def voxelize(lifted: list[Lifted], origin: np.ndarray, voxel: float) -> None:
    for l in lifted:
        l.codes = encode(np.floor((l.pts.astype(np.float64) - origin) / voxel).astype(np.int64))
        l.pts = None


def cluster_voxels(lifted: list[Lifted], min_support: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (kept voxel codes sorted, component label per kept voxel, (code_idx, view) support pairs)."""
    per = [np.unique(l.codes) for l in lifted]
    pairs = (np.concatenate([np.stack([u, np.full(len(u), l.view)], 1) for u, l in zip(per, lifted)])
             if lifted else np.zeros((0, 2), np.int64))
    pairs = np.unique(pairs, axis=0)
    codes, support = np.unique(pairs[:, 0], return_counts=True)
    kept = codes[support >= min_support]
    if kept.size == 0:
        return kept, np.zeros(0, np.int64), np.zeros((0, 2), np.int64)
    k = decode(kept)
    src, dst = [], []
    for d in _HALF_NEIGHBOURS:
        nb = encode(k + d)
        j = np.searchsorted(kept, nb)
        j = np.minimum(j, len(kept) - 1)
        found = kept[j] == nb
        src.append(np.nonzero(found)[0])
        dst.append(j[found])
    src, dst = np.concatenate(src), np.concatenate(dst)
    graph = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(len(kept), len(kept)))
    _, labels = connected_components(graph, directed=False)
    idx = np.searchsorted(kept, pairs[:, 0])
    idx = np.minimum(idx, len(kept) - 1)
    ok = kept[idx] == pairs[:, 0]
    return kept, labels, np.stack([idx[ok], pairs[ok, 1]], 1)


def lookup_labels(codes: np.ndarray, kept: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if kept.size == 0:
        return np.full(len(codes), -1, np.int64)
    j = np.minimum(np.searchsorted(kept, codes), len(kept) - 1)
    return np.where(kept[j] == codes, labels[j], -1)


def fill_from_nearest(grid: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Fill the −1 cells of `grid` that are inside the instance with the nearest labelled cell."""
    unknown = grid < 0
    if not (unknown & inside).any() or not (~unknown).any():
        return grid
    _, (ri, ci) = ndimage.distance_transform_edt(unknown, return_indices=True)
    filled = grid.copy()
    sel = unknown & inside
    filled[sel] = grid[ri[sel], ci[sel]]
    return filled


def split_labels(scene: SceneData, cam_idx: int, mask: np.ndarray, origin: np.ndarray, voxel: float,
                 kept: np.ndarray, obj_label: np.ndarray, members: list[int]) -> np.ndarray:
    """Object label of every depth pixel of a merged instance, from its own 3D point (full depth resolution,
    so the split follows the depth discontinuity between the objects)."""
    cam = scene.cameras[cam_idx]
    depth, Kd = scene.load_depth(cam)
    hd, wd = depth.shape
    h, w = mask.shape
    rr, cc = np.mgrid[0:hd, 0:wd]
    vi = np.clip(np.round((rr + 0.5) * h / hd - 0.5).astype(int), 0, h - 1)
    ui = np.clip(np.round((cc + 0.5) * w / wd - 0.5).astype(int), 0, w - 1)
    in_mask = mask[vi, ui]
    lifted = in_mask & (depth > 0)
    pts = unproject_pixels(cc[lifted].astype(np.float64), rr[lifted].astype(np.float64), depth[lifted], Kd, cam.c2w)
    lab = lookup_labels(encode(np.floor((pts - origin) / voxel).astype(np.int64)), kept, obj_label)
    grid = np.full((hd, wd), -1, np.int64)
    grid[lifted] = np.where(np.isin(lab, members), lab, -1)
    return fill_from_nearest(grid, ndimage.binary_dilation(in_mask, iterations=1))


def run_association(cfg: SceneConfig, scene: SceneData, work_dir: Path, scene_info: Optional[dict] = None) -> dict:
    out_dir = work_dir / "associate"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_info = scene_info or {}
    origin = np.asarray(scene_info.get("origin", np.mean([c.center for c in scene.cameras], 0)), np.float64)
    base_voxel = float(cfg.voxel_size or scene_info.get("auto_voxel", 0.0))
    if base_voxel <= 0:
        raise ValueError("voxel_size is 0 and prepare/scene.json has no auto_voxel — run `prepare` first")

    objects, assignments, split_grids, voxels_used = [], [], {}, {}
    for prompt in cfg.prompts:
        lifted = []
        for ci, cam in enumerate(scene.cameras):
            p = instances_path(work_dir, prompt, cam.stem)
            if not p.is_file() or scene.depth_path(cam) is None:
                continue
            inst = load_instances(p)
            lifted += lift_view_instances(scene, ci, prompt, inst["masks"], inst["scores"], cfg.lift_stride)
        if not lifted:
            print(f"[{prompt}] no instances with depth — run `masks` first")
            continue
        # voxels must hold several samples per view, or views never meet in the same voxel and objects
        # fall apart into fragments
        spacing = float(np.median([l.spacing for l in lifted]))
        voxel = max(base_voxel, 2.5 * spacing)
        voxels_used[prompt] = voxel
        voxelize(lifted, origin, voxel)
        print(f"[{prompt}] lifted {len(lifted)} instances from {len({l.view for l in lifted})} views; "
              f"sample spacing {spacing:.3f} → voxel {voxel:.3f}")
        kept, labels, support = cluster_voxels(lifted, cfg.min_view_support)
        if kept.size == 0:
            print(f"[{prompt}] no voxel is seen by {cfg.min_view_support}+ views — nothing to associate")
            continue
        sizes = np.bincount(labels)
        comp_views = np.unique(np.stack([labels[support[:, 0]], support[:, 1]], 1), axis=0)
        n_views = np.bincount(comp_views[:, 0], minlength=len(sizes))
        good = np.nonzero((sizes >= cfg.min_object_voxels) & (n_views >= cfg.min_object_views))[0]
        good = good[np.argsort(-sizes[good])]
        remap = np.full(len(sizes), -1, np.int64)
        base = len(objects)
        remap[good] = base + np.arange(len(good))
        obj_label = remap[labels]
        for j, comp in enumerate(good):
            vox = decode(kept[labels == comp])
            centres = (vox + 0.5) * voxel + origin
            objects.append({
                "index": base + j, "id": f"{slug(prompt)}_{j:03d}", "prompt": prompt,
                "n_voxels": int(sizes[comp]), "n_support_views": int(n_views[comp]),
                "centroid": centres.mean(0).tolist(), "bbox_min": centres.min(0).tolist(),
                "bbox_max": centres.max(0).tolist(),
            })
            np.save(out_dir / f"voxels_{objects[-1]['id']}.npy", centres.astype(np.float32))
        print(f"[{prompt}] {len(good)} objects (of {len(sizes)} components; "
              f"≥{cfg.min_object_voxels} voxels, ≥{cfg.min_object_views} views)")

        for l in lifted:
            lab = lookup_labels(l.codes, kept, obj_label)
            counts = np.bincount(lab[lab >= 0], minlength=len(objects)) if (lab >= 0).any() else np.zeros(len(objects))
            frac = counts / max(len(lab), 1)
            order = np.argsort(-frac)
            if frac[order[0]] < cfg.assign_min_frac:
                continue
            stem = scene.cameras[l.view].stem
            members = [int(order[0])] + [int(o) for o in order[1:]
                                         if frac[o] >= cfg.split_min_frac and counts[o] >= cfg.split_min_samples]
            if len(members) == 1:
                assignments.append({"prompt": prompt, "view": stem, "inst": l.inst, "object": members[0],
                                    "split": False, "frac": float(frac[members[0]])})
                continue
            mask = load_instances(instances_path(work_dir, prompt, stem))["masks"][l.inst]
            grid = split_labels(scene, l.view, mask, origin, voxel, kept, obj_label, members)
            key = f"{slug(prompt)}|{stem}|{l.inst}"
            split_grids[key] = grid.astype(np.int16)
            for o in members:
                assignments.append({"prompt": prompt, "view": stem, "inst": l.inst, "object": o, "split": True,
                                    "frac": float(frac[o]), "grid_key": key, "depth_hw": list(l.depth_hw),
                                    "stride": 1})

    (out_dir / "assignments.json").write_text(json.dumps(assignments))
    np.savez_compressed(out_dir / "split_grids.npz", **{k.replace("|", "__"): v for k, v in split_grids.items()})
    result = {"voxel": voxels_used, "origin": origin.tolist(), "objects": objects}
    (out_dir / "objects.json").write_text(json.dumps(result, indent=1))
    return result


class Associations:
    """Reads the association stage and rebuilds any object's mask in any view, at any size."""

    def __init__(self, work_dir: Path, cache_views: int = 8):
        self.work_dir = Path(work_dir)
        d = self.work_dir / "associate"
        self.info = json.loads((d / "objects.json").read_text())
        self.objects = {o["id"]: o for o in self.info["objects"]}
        self.by_index = {o["index"]: o for o in self.info["objects"]}
        self.assignments = json.loads((d / "assignments.json").read_text())
        z = np.load(d / "split_grids.npz")
        self.split_grids = {k.replace("__", "|"): z[k] for k in z.files}
        self._cache: OrderedDict = OrderedDict()
        self._cache_n = cache_views

    def views_of(self, obj_id: str) -> list[str]:
        idx = self.objects[obj_id]["index"]
        return sorted({a["view"] for a in self.assignments if a["object"] == idx})

    def _instances(self, prompt: str, stem: str) -> np.ndarray:
        key = (prompt, stem)
        if key not in self._cache:
            self._cache[key] = load_instances(instances_path(self.work_dir, prompt, stem))["masks"]
            while len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def mask(self, obj_id: str, stem: str, size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Boolean mask of the object in that view at (width, height) (default: the image size)."""
        obj = self.objects[obj_id]
        out = None
        for a in self.assignments:
            if a["object"] != obj["index"] or a["view"] != stem:
                continue
            inst = self._instances(a["prompt"], stem)[a["inst"]]
            if a["split"]:
                grid = self.split_grids[a["grid_key"]]
                hd, wd = a["depth_hw"]
                h, w = inst.shape
                s = a["stride"]
                gy = np.clip(np.round(((np.arange(h) + 0.5) * hd / h - 0.5) / s).astype(int), 0, grid.shape[0] - 1)
                gx = np.clip(np.round(((np.arange(w) + 0.5) * wd / w - 0.5) / s).astype(int), 0, grid.shape[1] - 1)
                inst = inst & (grid[gy][:, gx] == obj["index"])
            out = inst.copy() if out is None else (out | inst)
        if out is None:
            raise KeyError(f"{obj_id} has no mask in view {stem}")
        return resize_mask(out, size) if size is not None else out


def select_views(cfg: SceneConfig, scene: SceneData, assoc: Associations, obj_id: str,
                 small_side: int = 256) -> list[dict]:
    """Score every view of the object; the chosen MV-SAM3D views come back ordered, reference first."""
    centres = np.load(assoc.work_dir / "associate" / f"voxels_{obj_id}.npy").astype(np.float64)
    if len(centres) > 4000:
        centres = centres[np.random.default_rng(0).choice(len(centres), 4000, replace=False)]
    centroid = centres.mean(0)
    stats = []
    for stem in assoc.views_of(obj_id):
        cam = scene.by_stem[stem]
        s = small_side / max(cam.width, cam.height)
        w, h = max(1, round(cam.width * s)), max(1, round(cam.height * s))
        m = assoc.mask(obj_id, stem, (w, h))
        u, v, z = project(centres, cam.w2c, cam.scaled_K(w, h))
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inimg = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        hit = np.zeros(len(centres), bool)
        hit[inimg] = m[vi[inimg], ui[inimg]]
        band = max(1, round(0.01 * max(w, h)))
        border = bool(m[:band].any() or m[-band:].any() or m[:, :band].any() or m[:, -band:].any())
        coverage = float(hit.mean())
        area = float(m.mean())
        d = cam.center - centroid
        stats.append({"view": stem, "coverage": coverage, "in_image": float(inimg.mean()), "area_frac": area,
                      "truncated": border, "score": coverage * np.sqrt(area) * (0.5 if border else 1.0),
                      "dir": (d / (np.linalg.norm(d) + 1e-12)).tolist()})
    cands = sorted([s for s in stats if s["coverage"] >= cfg.min_view_coverage], key=lambda s: -s["score"])
    chosen: list[dict] = []
    untrunc = [s for s in cands if not s["truncated"]]
    if cands:
        chosen.append(untrunc[0] if untrunc else cands[0])
    while len(chosen) < cfg.views_per_object:
        rest = [s for s in cands if s not in chosen]
        if not rest:
            break
        dirs = np.array([c["dir"] for c in chosen])

        def gain(s):
            ang = np.degrees(np.arccos(np.clip(dirs @ np.array(s["dir"]), -1, 1))).min()
            return s["score"] * (0.5 + min(ang, 90.0) / 90.0)

        chosen.append(max(rest, key=gain))
    for i, s in enumerate(chosen):
        s["selected_rank"] = i
    return stats
