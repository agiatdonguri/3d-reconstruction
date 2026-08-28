"""Flow-matching LoRA train loop + artifact export."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch


def _make_run_dir(output_dir: Path, run_name: str, case_name: str) -> Path:
    from .safety import assert_writable_dir

    output_dir = assert_writable_dir(output_dir, label="output_dir", create=True)
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name / case_name
    return assert_writable_dir(run_dir, label="run_dir", create=True)


def train_flow_matching(
    slat_gen,
    peft_backbone,
    x1: torch.Tensor,
    cond_args_fm: tuple,
    cond_kwargs_train: dict,
    pipeline,
    *,
    train_steps: int,
    lr: float,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    log_every: int = 10,
) -> list[float]:
    from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching

    device = x1.device
    opt = torch.optim.AdamW(
        [p for p in slat_gen.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    history: list[float] = []
    slat_gen.train()
    peft_backbone.train()

    print(f"Training {train_steps} steps | lr={lr} | device={device}")
    for step in range(int(train_steps)):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=pipeline.dtype, enabled=(device.type == "cuda")):
            loss, _details = FlowMatching.loss(
                slat_gen, x1, *cond_args_fm, **cond_kwargs_train
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in slat_gen.parameters() if p.requires_grad], float(grad_clip)
        )
        opt.step()

        lv = float(loss.detach().float().cpu())
        history.append(lv)
        if step % int(log_every) == 0 or step == int(train_steps) - 1:
            print(f"step {step:4d}/{train_steps}  flow_matching_loss={lv:.6f}", flush=True)
    return history


def save_artifacts(
    run_dir: Path,
    peft_backbone,
    history: Sequence[float],
    meta: dict[str, Any],
    plot: bool = True,
) -> Path:
    from .safety import assert_writable_dir, safe_rmtree

    run_dir = assert_writable_dir(run_dir, label="run_dir", create=True)

    adapter_dir = run_dir / "lora_adapter"
    if adapter_dir.exists():
        safe_rmtree(adapter_dir, must_be_under=run_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_backbone.save_pretrained(str(adapter_dir))

    hist = np.asarray(history, dtype=np.float64)
    np.savez(run_dir / "loss_history.npz", loss=hist)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if plot and len(hist) > 0:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hist, lw=1.5)
            ax.set_xlabel("step")
            ax.set_ylabel("flow-matching loss")
            ax.set_title(
                f"Stage-2 LoRA (N={meta.get('micro_n')}, r={meta.get('lora_r')})"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            curve = run_dir / "stage2_lora_loss_curve.png"
            fig.savefig(curve, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print("Saved", curve)
        except Exception as e:
            print("plot skipped:", e)

    print("loss first→last:", round(float(hist[0]), 6), "→", round(float(hist[-1]), 6))
    print("adapter →", adapter_dir)
    return adapter_dir


def run_training(cfg) -> Path:
    """Full Stage-2 LoRA PoC: load → micro-batch → teacher → LoRA → FM train → save."""
    from .config import Stage2LoRAConfig
    from .data import load_views, preprocess_views
    from .dataset_layout import resolve_view_roots
    from .lora import inject_lora
    from .pipeline_io import load_inference, sample_stage1_coords, teacher_sample_slat
    from .safety import validate_train_io

    assert isinstance(cfg, Stage2LoRAConfig)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # Refuse writing into /data2/mehark (or other STAGE2_READONLY_ROOTS)
    validate_train_io(
        data_root=cfg.data_root,
        output_dir=cfg.output_dir,
        mask_root=cfg.mask_root or None,
    )

    roots = resolve_view_roots(cfg)
    case = roots.case_name
    print("DATA:", json.dumps(roots.summary(), indent=2))

    run_dir = _make_run_dir(Path(cfg.output_dir), cfg.run_name, case)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print("RUN_DIR =", run_dir)

    pipeline_yaml = cfg.resolve_pipeline_yaml()
    _inference, pipeline, device = load_inference(
        cfg.mvsam3d_root, pipeline_yaml, compile=cfg.compile_pipeline
    )
    slat_gen = pipeline.models["slat_generator"]
    print("slat_generator:", type(slat_gen).__name__)
    print("reverse_fn:", type(slat_gen.reverse_fn).__name__)
    print("backbone:", type(slat_gen.reverse_fn.backbone).__name__)

    images, masks, img_paths = load_views(roots.images_dir, roots.mask_dir, cfg.micro_n)
    print("stems:", [p.stem for p in img_paths])

    print(f"Preprocessing {cfg.micro_n} views…")
    ss_dicts, slat_dicts = preprocess_views(pipeline, images, masks)

    print("Stage-1 multi-view → coords…")
    coords = sample_stage1_coords(pipeline, ss_dicts, steps=cfg.stage1_steps)

    print("Teacher Stage-2 (base weights) → x1…")
    x1, cond_args_fm, cond_kwargs_train = teacher_sample_slat(
        pipeline,
        slat_gen,
        slat_dicts,
        coords,
        device,
        teacher_steps=cfg.stage2_teacher_steps,
    )

    peft_backbone, trainable, total = inject_lora(
        slat_gen,
        targets=cfg.lora_targets,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        resume_adapter=cfg.resume_adapter or None,
    )

    history = train_flow_matching(
        slat_gen,
        peft_backbone,
        x1,
        cond_args_fm,
        cond_kwargs_train,
        pipeline,
        train_steps=cfg.train_steps,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        log_every=cfg.log_every,
    )

    meta = {
        **cfg.to_dict(),
        "stems": [p.stem for p in img_paths],
        "loss_first": float(history[0]) if history else None,
        "loss_last": float(history[-1]) if history else None,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "run_dir": str(run_dir),
        "pipeline_yaml": str(pipeline_yaml),
    }
    save_artifacts(run_dir, peft_backbone, history, meta, plot=True)
    return run_dir
