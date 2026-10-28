"""Multi-GPU helpers, named and structured like map-anything's mapanything/utils/train_tools.py
(init_distributed_mode, setup_for_distributed, get_rank / get_world_size / is_main_process, barrier), and
launched the same way: torchrun sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT, and
init_process_group reads them through init_method="env://".

Unlike DDP *training*, nothing is trained here, so there is no DistributedDataParallel wrapper, no
DistributedSampler and no gradient all-reduce: each rank is one GPU worker that runs whole buildings
(MV-SAM3D inference, then refinement) on its own. Ranks only meet at barriers and to gather a summary, so
the process group uses the CPU `gloo` backend (the MV-SAM3D runs are subprocesses that own the GPU) with
a long timeout, because the last barrier waits for the slowest building.

Work split: instead of a static DistributedSampler shard, buildings are claimed from a shared folder
(atomic file create), so a GPU that finishes early takes the next building and a rerun skips finished
ones. Building costs vary several-fold, which a static 1/N split would turn into idle GPUs."""
from __future__ import annotations

import builtins
import datetime
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def dist_info() -> DistInfo:
    """From the torchrun environment; valid before (or without) init_distributed_mode."""
    return DistInfo(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
                    int(os.environ.get("LOCAL_RANK", 0)))


def is_dist_avail_and_initialized() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_for_distributed(is_master: bool) -> None:
    """Silence print() on ranks other than 0; print(..., force=True) still prints."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    print.dist_patched = True
    builtins.print = print


def print_all(*args, **kwargs) -> None:
    """Print from every rank (per-building progress)."""
    if getattr(builtins.print, "dist_patched", False):
        kwargs["force"] = True
    builtins.print(*args, **kwargs)


def init_distributed_mode(backend: str = "gloo", timeout_hours: float = 24.0) -> bool:
    """Join the torchrun process group if launched by torchrun; returns whether we are distributed."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2:
        return False
    import torch
    import torch.distributed as dist

    info = dist_info()
    if backend == "nccl":
        torch.cuda.set_device(info.local_rank)   # before init_process_group, so NCCL binds this rank's GPU
    dist.init_process_group(backend=backend, init_method="env://", world_size=info.world_size, rank=info.rank,
                            timeout=datetime.timedelta(hours=timeout_hours))
    dist.barrier()
    setup_for_distributed(info.is_main)
    return True


def barrier() -> None:
    if is_dist_avail_and_initialized():
        import torch.distributed as dist

        dist.barrier()


def all_gather_object(obj) -> list:
    """Every rank's `obj` on every rank (every rank must call this, or the job hangs)."""
    if not is_dist_avail_and_initialized():
        return [obj]
    import torch.distributed as dist

    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def destroy_distributed() -> None:
    if is_dist_avail_and_initialized():
        import torch.distributed as dist

        dist.destroy_process_group()


def physical_gpu(local_rank: int) -> str:
    """The GPU id this rank hands to its MV-SAM3D subprocess via CUDA_VISIBLE_DEVICES."""
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return visible[local_rank % len(visible)] if visible else str(local_rank)


def claim(path: Path, stale_after_s: float) -> bool:
    """True if this process now owns `path` (a claim file). Claims older than stale_after_s are taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time(), "rank": dist_info().rank}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                json.dump(owner, f)
            return True
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < stale_after_s:
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return False
