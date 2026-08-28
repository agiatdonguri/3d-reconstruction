"""PEFT LoRA injection into Stage-2 SLAT backbone."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def inject_lora(
    slat_gen,
    targets: Sequence[str],
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    resume_adapter: str | None = None,
):
    """Freeze base weights; attach LoRA to reverse_fn.backbone. Returns peft backbone."""
    from peft import LoraConfig, PeftModel, get_peft_model

    cfg_mod = slat_gen.reverse_fn
    backbone = cfg_mod.backbone

    for p in slat_gen.parameters():
        p.requires_grad = False

    if resume_adapter:
        adapter = Path(resume_adapter)
        assert adapter.exists(), adapter
        print(f"Loading existing LoRA adapter from {adapter}")
        peft_backbone = PeftModel.from_pretrained(backbone, str(adapter))
        for p in peft_backbone.parameters():
            if "lora_" in (getattr(p, "name", "") or ""):
                p.requires_grad = True
        # PeftModel: enable train on adapters
        peft_backbone.train()
        for n, p in peft_backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    else:
        hits = [n for n, _ in backbone.named_modules() if n.split(".")[-1] in set(targets)]
        print(f"LoRA name hits: {len(hits)}  e.g. {hits[:8]}")
        if not hits:
            raise RuntimeError(f"No modules match LORA targets={list(targets)}")
        lora_cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            target_modules=list(targets),
            lora_dropout=float(dropout),
            bias="none",
        )
        peft_backbone = get_peft_model(backbone, lora_cfg)

    cfg_mod.backbone = peft_backbone
    cfg_mod.strength = 0.0
    slat_gen.no_shortcut = True

    trainable = sum(p.numel() for p in slat_gen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in slat_gen.parameters())
    print(
        f"slat_generator trainable: {trainable:,} / {total:,}  "
        f"({100.0 * trainable / max(total, 1):.4f}%)"
    )
    if hasattr(peft_backbone, "print_trainable_parameters"):
        peft_backbone.print_trainable_parameters()
    print("CFG strength = 0 for FM training")
    return peft_backbone, trainable, total
