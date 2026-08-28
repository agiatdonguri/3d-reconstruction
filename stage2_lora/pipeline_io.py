"""Load MV-SAM3D Inference pipeline + Stage-1/2 teacher sampling."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _find_inference_py(mvsam3d_root: Path) -> Path:
    cand = mvsam3d_root / "notebook" / "inference.py"
    if cand.exists():
        return cand
    # fallback: sam-3d-objects sibling
    alt = mvsam3d_root.parent / "sam-3d-objects" / "notebook" / "inference.py"
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"notebook/inference.py not found under {mvsam3d_root} (or sibling sam-3d-objects)"
    )


def load_inference(mvsam3d_root: str | Path, pipeline_yaml: str | Path, compile: bool = False):
    """Return (Inference instance, pipeline, device)."""
    mvsam3d_root = Path(mvsam3d_root).resolve()
    pipeline_yaml = Path(pipeline_yaml).resolve()
    assert pipeline_yaml.exists(), pipeline_yaml

    # CUDA JIT env (same spirit as Colab / sam_baneus)
    os.environ.setdefault("CUMM_DISABLE_JIT", "1")
    os.environ.setdefault("SPCONV_DISABLE_JIT", "1")

    inf_py = _find_inference_py(mvsam3d_root)
    nb_dir = str(inf_py.parent)
    if nb_dir not in sys.path:
        sys.path.insert(0, nb_dir)
    if str(mvsam3d_root) not in sys.path:
        sys.path.insert(0, str(mvsam3d_root))

    try:
        from inference import Inference  # type: ignore
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location("mvsam3d_notebook_inference", inf_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["mvsam3d_notebook_inference"] = mod
        spec.loader.exec_module(mod)
        Inference = mod.Inference  # type: ignore

    print(f"MVSAM3D_ROOT = {mvsam3d_root}")
    print(f"inference.py = {inf_py}")
    print(f"PIPELINE_YAML = {pipeline_yaml}")
    print("Loading Inference pipeline…")
    inference = Inference(str(pipeline_yaml), compile=compile)
    pipeline = inference._pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-2 LoRA requires a CUDA GPU")
    return inference, pipeline, device


def sample_stage1_coords(pipeline, ss_dicts, steps: int = 25, mode: str = "multidiffusion"):
    with torch.no_grad():
        ss_ret = pipeline.sample_sparse_structure_multi_view(
            ss_dicts, inference_steps=int(steps), mode=mode
        )
    coords = ss_ret["coords"]
    print("coords:", tuple(coords.shape), "dtype=", getattr(coords, "dtype", None))
    return coords


def teacher_sample_slat(
    pipeline,
    slat_gen,
    slat_dicts,
    coords,
    device: torch.device,
    teacher_steps: int = 25,
    mode: str = "multidiffusion",
) -> tuple[torch.Tensor, tuple, dict]:
    """Base-weight Stage-2 teacher → x1 + multi-view cond (then collapse to view-0 for FM)."""
    from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view

    cfg_mod = slat_gen.reverse_fn
    slat_gen.no_shortcut = True
    cfg_mod.strength = float(getattr(pipeline, "slat_cfg_strength", 0.0) or 0.0)

    # spconv requires int32 indices
    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy().astype(np.int32, copy=False)
    else:
        coords_np = np.asarray(coords, dtype=np.int32)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.dtype, enabled=(device.type == "cuda")):
            cond_args, cond_kwargs = pipeline.get_multi_view_condition_input(
                pipeline.condition_embedders["slat_condition_embedder"],
                slat_dicts,
                pipeline.slat_condition_input_mapping,
            )
            cond_args = tuple(cond_args) + (coords_np,)
            latent_shape = (slat_dicts[0]["image"].shape[0], coords_np.shape[0], 8)
            prev_steps = slat_gen.inference_steps
            slat_gen.inference_steps = int(teacher_steps)
            with inject_generator_multi_view(
                slat_gen,
                num_views=len(slat_dicts),
                num_steps=slat_gen.inference_steps,
                mode=mode,
            ):
                teacher = slat_gen(latent_shape, device, *cond_args, **cond_kwargs)
            slat_gen.inference_steps = prev_steps
            x1 = teacher[0].detach().float()
            if x1.dim() == 2:
                x1 = x1.unsqueeze(0)

    cond_args_train = []
    for a in cond_args[:-1]:
        cond_args_train.append(a.detach().to(device) if torch.is_tensor(a) else a)
    cond_args_train.append(cond_args[-1])
    cond_args_train = tuple(cond_args_train)
    cond_kwargs_train = {
        k: (v.detach() if torch.is_tensor(v) else v) for k, v in cond_kwargs.items()
    }
    x1 = x1.to(device)

    # FM.loss calls reverse_fn once — use view-0 tokens only
    cond0 = cond_args_train[0]
    if torch.is_tensor(cond0) and cond0.shape[0] == len(slat_dicts):
        cond_single = cond0[0]
        if cond_single.dim() == 2:
            cond_single = cond_single.unsqueeze(0)
        cond_args_fm = (cond_single,) + cond_args_train[1:]
        print(f"train cond: multi-view {tuple(cond0.shape)} → view0 {tuple(cond_single.shape)}")
    else:
        cond_args_fm = cond_args_train
        print("train cond: using as-is", type(cond0), getattr(cond0, "shape", None))

    print(f"x1 shape={tuple(x1.shape)}")
    return x1, cond_args_fm, cond_kwargs_train
