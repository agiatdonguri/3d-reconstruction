"""Never write into shared peer trees (Mehar's WRIVA copy). Reading from them is fine; copy first anyway."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_READONLY = ("/data2/mehark", "/data/mehark", "/data3/mehark")


def readonly_roots() -> list[Path]:
    extra = [p for p in os.environ.get("STAGE2_READONLY_ROOTS", "").split(":") if p.strip()]
    return [Path(r).expanduser().resolve() for r in (*DEFAULT_READONLY, *extra)]


def assert_writable_dir(path: str | Path, *, label: str = "output") -> Path:
    p = Path(path).expanduser().resolve()
    for root in readonly_roots():
        if p == root or root in p.parents:
            raise PermissionError(f"REFUSING to write {label}={p}: it is under the protected root {root}. "
                                  "Copy the data to /data3/tnishida/… and write there.")
    p.mkdir(parents=True, exist_ok=True)
    probe = p / f".write_probe.{os.getpid()}"   # per process: parallel ranks share the directory
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise PermissionError(f"cannot write under {label}={p}: {e}") from e
    return p
