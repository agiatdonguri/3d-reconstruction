"""Config dataclass + YAML / CLI loading for Stage-2 LoRA."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Stage2LoRAConfig:
    # data
    # layout: "legacy" = SatelliteFuse va_split; "wriva" = Trinity WRIVA/<scene>/
    layout: str = "legacy"
    case_name: str = "va_split"
    scene: str = ""  # WRIVA scene id; empty → case_name
    data_root: str = ""
    aerial_subdir: str = "satellite"
    img_subdir: str = "images"
    mask_subdir: str = "masks"
    # Sidecar SAM masks (WRIVA has none). e.g. /data2/$USER/masks
    mask_root: str = ""
    wriva_images_rel: str = "sparse/undistorted_mitigated_images/0"

    # MV-SAM3D / checkpoints (cluster install)
    mvsam3d_root: str = ""
    pipeline_yaml: str = ""  # empty → <mvsam3d_root>/checkpoints/hf/pipeline.yaml

    # micro-batch / teacher
    micro_n: int = 8
    seed: int = 0
    stage1_steps: int = 25
    stage2_teacher_steps: int = 25

    # LoRA
    train_steps: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(
        default_factory=lambda: ["to_qkv", "to_q", "to_kv", "to_out"]
    )
    log_every: int = 10

    # I/O
    output_dir: str = "./outputs/stage2_lora"
    run_name: str = ""  # empty → auto timestamp
    resume_adapter: str = ""  # optional existing peft adapter to continue

    # runtime
    device: str = "cuda"
    compile_pipeline: bool = False

    def aerial_dir(self) -> Path:
        return Path(self.data_root) / self.aerial_subdir

    def img_dir(self) -> Path:
        return self.aerial_dir() / self.img_subdir

    def mask_dir(self) -> Path:
        return self.aerial_dir() / self.mask_subdir

    def resolve_pipeline_yaml(self) -> Path:
        if self.pipeline_yaml:
            p = Path(self.pipeline_yaml)
            if p.exists():
                return p
        root = Path(self.mvsam3d_root)
        cand = root / "checkpoints" / "hf" / "pipeline.yaml"
        if cand.exists():
            return cand
        raise FileNotFoundError(
            f"pipeline.yaml not found (set pipeline_yaml or mvsam3d_root). tried={cand}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stage2LoRAConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        # minimal fallback: only flat key: value lines (no nested YAML)
        data = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip().strip("'\"")
            if v.lower() in ("true", "false"):
                data[k] = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        data[k] = float(v)
                    else:
                        data[k] = int(v)
                except ValueError:
                    if v.startswith("[") and v.endswith("]"):
                        inner = v[1:-1].strip()
                        data[k] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                    else:
                        data[k] = v
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def apply_env_overrides(cfg: Stage2LoRAConfig) -> Stage2LoRAConfig:
    """Optional env overrides for cluster schedules (no code edits)."""
    mapping = {
        "STAGE2_DATA_ROOT": "data_root",
        "STAGE2_MVSAM3D_ROOT": "mvsam3d_root",
        "STAGE2_PIPELINE_YAML": "pipeline_yaml",
        "STAGE2_OUTPUT_DIR": "output_dir",
        "STAGE2_RUN_NAME": "run_name",
        "STAGE2_MICRO_N": "micro_n",
        "STAGE2_TRAIN_STEPS": "train_steps",
        "STAGE2_LR": "lr",
        "STAGE2_LORA_R": "lora_r",
        "STAGE2_LORA_ALPHA": "lora_alpha",
        "STAGE2_SEED": "seed",
        "STAGE2_LAYOUT": "layout",
        "STAGE2_SCENE": "scene",
        "STAGE2_MASK_ROOT": "mask_root",
    }
    d = cfg.to_dict()
    for env_k, field_k in mapping.items():
        if env_k not in os.environ or not os.environ[env_k]:
            continue
        raw = os.environ[env_k]
        cur = d[field_k]
        if isinstance(cur, bool):
            d[field_k] = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            d[field_k] = int(raw)
        elif isinstance(cur, float):
            d[field_k] = float(raw)
        else:
            d[field_k] = raw
    return Stage2LoRAConfig.from_dict(d)


def load_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Stage2LoRAConfig:
    d: dict[str, Any] = {}
    if yaml_path:
        d.update(load_yaml(yaml_path))
    if overrides:
        d.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Stage2LoRAConfig.from_dict(d)
    return apply_env_overrides(cfg)
