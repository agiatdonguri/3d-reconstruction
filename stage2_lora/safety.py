"""Cluster safety: never write into *shared* peer trees (e.g. Mehar).

Intended workflow
-----------------
1. Copy WRIVA scenes once into **your** tree, e.g. `/data2/tnishida/WRIVA/…`
2. Point all jobs at that copy (data_root, masks, outputs).
3. Never point write paths at `/data2/mehark/…`.

Policy
------
- `/data2/mehark` is **read-only** (copy source only).
- Your own copy under `/data2/tnishida` (or `$HOME`) is fully writable —
  including `<scene>/masks/` next to the RGB you copied.
- Destructive ops (`rmtree`) only allowed under an already-validated run dir.
"""
from __future__ import annotations

import os
from pathlib import Path


# Shared datasets we must never modify. Extend via STAGE2_READONLY_ROOTS (":"-separated).
_DEFAULT_READONLY = (
    "/data2/mehark",
    "/data/mehark",
    "/data3/mehark",
)


def readonly_roots() -> list[Path]:
    roots = list(_DEFAULT_READONLY)
    extra = os.environ.get("STAGE2_READONLY_ROOTS", "")
    if extra:
        roots.extend(p for p in extra.split(":") if p.strip())
    out: list[Path] = []
    for r in roots:
        try:
            out.append(Path(r).expanduser().resolve())
        except OSError:
            out.append(Path(r).expanduser())
    return out


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def is_under(child: Path, parent: Path) -> bool:
    try:
        child = child.resolve()
        parent = parent.resolve()
    except OSError:
        return False
    return child == parent or parent in child.parents


def assert_readable(path: str | Path, *, label: str = "path") -> Path:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    return p


def assert_not_readonly_target(path: str | Path, *, label: str = "write path") -> Path:
    """Refuse any write whose destination sits under a protected root."""
    p = _resolve(path)
    for root in readonly_roots():
        if is_under(p, root):
            raise PermissionError(
                f"REFUSING write: {label}={p} is under protected read-only root {root}.\n"
                "Copy data to your own path first (e.g. /data2/tnishida/WRIVA), "
                "then write masks/outputs only under that tree — never under /data2/mehark/."
            )
    return p


def assert_writable_dir(
    path: str | Path,
    *,
    label: str = "output",
    create: bool = True,
) -> Path:
    p = assert_not_readonly_target(path, label=label)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    # probe write access without leaving junk if possible
    probe = p / ".stage2_write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise PermissionError(f"cannot write under {label}={p}: {e}") from e
    return p


def safe_rmtree(path: str | Path, *, must_be_under: str | Path) -> None:
    """Delete only if `path` is strictly inside `must_be_under` (not equal)."""
    import shutil

    target = _resolve(path)
    parent = _resolve(must_be_under)
    assert_not_readonly_target(target, label="rmtree target")
    if not is_under(target, parent) or target == parent:
        raise PermissionError(
            f"REFUSING rmtree: {target} is not strictly under allowed parent {parent}"
        )
    if target.exists():
        shutil.rmtree(target)


def validate_train_io(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    mask_root: str | Path | None = None,
) -> dict[str, Path]:
    """Preflight for Stage-2 LoRA: read data_root, write only output (+ optional mask check)."""
    data = assert_readable(data_root, label="data_root")
    # reading under mehark is fine; writing is not — we only mkdir output
    out = assert_writable_dir(output_dir, label="output_dir", create=True)
    result = {"data_root": data, "output_dir": out}
    if mask_root:
        # masks may not exist yet; if path is under readonly, refuse even existence checks for writes later
        mr = Path(mask_root).expanduser()
        # resolve parent chain carefully
        try:
            mr_res = mr.resolve()
        except OSError:
            mr_res = mr
        assert_not_readonly_target(mr_res if mr_res.exists() else mr, label="mask_root")
        result["mask_root"] = mr_res
    return result


def validate_mask_out(*, wriva_root: str | Path, mask_out: str | Path) -> Path:
    """Allow masks under *your* WRIVA copy; refuse Mehar / other readonly roots.

    `wriva_root` is only used for a warning if it still points at a readonly tree
    (you should copy first). Mask writes are gated solely by readonly_roots().
    """
    wriva = Path(wriva_root).expanduser()
    try:
        wriva_res = wriva.resolve()
    except OSError:
        wriva_res = wriva
    for root in readonly_roots():
        if is_under(wriva_res, root):
            print(
                f"WARNING: wriva_root={wriva_res} is under {root}. "
                "Prefer a personal copy under /data2/tnishida/… before training."
            )
            break
    return assert_writable_dir(mask_out, label="mask_out", create=True)
