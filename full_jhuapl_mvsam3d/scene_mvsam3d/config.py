"""Config for the WRIVA whole-scene MV-SAM3D pipeline: YAML file + `--set key=value` overrides."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the YAML root must be a mapping")
    return data


@dataclass
class SceneConfig:
    # ---- inputs (read-only; copy from Mehar first, see README "One-time setup") ----
    wriva_root: str = "/scratch/tnishida/WRIVA"
    scene: str = ""
    images_rel: str = "sparse/undistorted_mitigated_images/0"
    colmap_rel: str = ""                            # cameras.txt, images.txt, points3D.(txt|ply); "" → auto:
                                                    # sparse/sparse (WRIVA), sparse, sparse/0
    depth_rel: str = "sparse/metric_z_depth_mve"    # MoGe z-depth per image (pseudo ground truth)
    depth_kind: str = "z"                           # z: distance along the optical axis; ray: distance from the
                                                    # camera centre (MVE's convention). `inspect` measures which.
    depth_patterns: list[str] = field(default_factory=lambda: ["{stem}_zdepth.npy", "{stem}.npy", "{stem}_depth.npy"])
    depth_scale_json_rel: str = "sparse/depth_scale.json"
    # auto = per-image scale fitted to the COLMAP sparse points (falls back to the scene median);
    # json = multiply by depth_scale.json; none = use the files as they are. `inspect` reports all three.
    depth_scale_mode: str = "auto"
    image_glob: str = "*"                           # fnmatch on the image file name, e.g. "*drone*"
    max_images: int = 0                             # 0 = all

    # ---- outputs (refused under /data*/mehark) ----
    work_root: str = "/scratch/tnishida/outputs/full_jhuapl_mvsam3d"
    run_name: str = ""                              # empty → scene name

    # ---- SAM 3 instance masks ----
    prompts: list[str] = field(default_factory=lambda: ["building"])
    sam3_root: str = ""                             # checkout of facebookresearch/sam3 if not pip-installed
    sam3_checkpoint: str = ""                       # empty → download facebook/sam3 from Hugging Face
    sam3_score_thresh: float = 0.4
    mask_min_area_frac: float = 0.0005              # drop instances smaller than this fraction of the image
    # import_masks stage (instead of `masks`): existing per-view instance masks, e.g. Mehar's SAM 3 output.
    # Per view, `masks_import_pattern` (with {stem} = image name without extension, globs allowed) is looked
    # up under masks_import_dir as: a folder of one-mask-per-file images; one file (.npz with `masks` (K,H,W),
    # .npy (K,H,W) or (H,W) label map, .png label map / single binary mask); or several files (one per instance).
    masks_import_dir: str = ""
    masks_import_pattern: str = "{stem}"
    masks_import_prompt: str = "building"           # the prompt the imported masks stand for
    masks_import_kind: str = "auto"                 # auto | label (0 = background, k = instance k) | binary

    # ---- multi-view association (3D, from metric depth + poses) ----
    lift_stride: int = 4                            # lift every n-th depth pixel
    voxel_size: float = 0.0                         # metres; 0 → scene extent / 400
    min_view_support: int = 2                       # a voxel must be seen in this many views
    min_object_voxels: int = 30
    min_object_views: int = 3
    assign_min_frac: float = 0.3                    # an instance joins an object if this much of it lands there
    # …and is split when another object (confirmed by other views) holds this share and sample count of it,
    # e.g. SAM merging a partly hidden building into the one in front
    split_min_frac: float = 0.05
    split_min_samples: int = 30

    # ---- per-object MV-SAM3D input ----
    views_per_object: int = 8
    min_view_coverage: float = 0.15                 # fraction of the object's voxels visible and inside the mask
    image_max_side: int = 1600

    # ---- MV-SAM3D ----
    mvsam3d_root: str = "/home/tnishida/code/MV-SAM3D"
    mvsam3d_python: str = ""                        # empty → this interpreter
    model_tag: str = "hf"                           # <mvsam3d_root>/checkpoints/<tag>/pipeline.yaml
    seed: int = 42
    stage1_steps: int = 50
    stage2_steps: int = 25
    decode_formats: str = "gaussian,mesh"
    stage2_weight_source: str = "entropy"           # entropy | visibility | mixed
    mvsam3d_pose_optimization: bool = False         # MV-SAM3D's own pose optimization (`--run_pose_optimization`)
    mvsam3d_extra_args: list[str] = field(default_factory=list)
    object_timeout_min: float = 45.0
    max_objects: int = 0                            # 0 = all; else only the N biggest (smoke test)

    # ---- placement, refinement, evaluation ----
    refine: bool = True
    refine_dofs: list[str] = field(default_factory=lambda: ["yaw", "tx", "ty", "s_h", "s_v"])
    refine_max_views: int = 12
    refine_max_evals: int = 400
    refine_max_scale: float = 4.0                   # bound on s_h / s_v (Mehar saw 2–4× too tall)
    eval_max_side: int = 512                        # silhouettes are compared at this resolution
    occlusion_tol: float = 0.05                     # mesh pixels this far (fraction of depth) behind the photo are hidden
    surface_samples: int = 60000

    def scene_root(self) -> Path:
        return Path(self.wriva_root) / self.scene

    def work_dir(self) -> Path:
        return Path(self.work_root) / (self.run_name or self.scene)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneConfig":
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(d) - set(known))
        if unknown:
            raise KeyError(f"unknown config keys: {unknown}")
        cfg = cls(**d)
        for f in fields(cls):   # YAML reads `45` as int; keep float fields float so --set 60.5 parses
            v = getattr(cfg, f.name)
            if f.type == "float" and isinstance(v, int) and not isinstance(v, bool):
                setattr(cfg, f.name, float(v))
        return cfg


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_config(yaml_path: Optional[str | Path] = None, sets: Optional[list[str]] = None) -> SceneConfig:
    d: dict[str, Any] = {}
    if yaml_path:
        d.update(load_yaml(yaml_path))
    cfg = SceneConfig.from_dict(d)
    for item in sets or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key = key.strip()
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config key {key!r}")
        setattr(cfg, key, _coerce(value.strip(), getattr(cfg, key)))
    if not cfg.scene:
        raise ValueError("config needs `scene` (the WRIVA scene folder name)")
    return cfg
