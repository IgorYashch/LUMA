"""Shared test utilities for the LUMA test-suite.

Single source of truth for hardware-detection flags, pytest skip markers,
PRNG helpers, shared models, and the distributed test harness used by
``test_fsdp.py`` and ``test_mixed_precision.py``.
"""

from __future__ import annotations

import os
import random
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# =====================================================================
#  Hardware detection
# =====================================================================

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    pass

CUDA_AND_TRITON: bool = torch.cuda.is_available() and _TRITON_AVAILABLE
MULTI_GPU: bool = torch.cuda.is_available() and torch.cuda.device_count() >= 2
GPU_BF16: bool = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
)

_FSDP2_AVAILABLE = False
_FSDP2_MP_AVAILABLE = False
try:
    from torch.distributed._composable.fsdp import fully_shard  # noqa: F401

    _FSDP2_AVAILABLE = True

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy  # noqa: F401

    _FSDP2_MP_AVAILABLE = True
except ImportError:
    pass

# =====================================================================
#  Pytest skip markers
# =====================================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)

requires_cuda_triton = pytest.mark.skipif(
    not CUDA_AND_TRITON,
    reason="Requires CUDA + Triton",
)

requires_multi_gpu = pytest.mark.skipif(
    not MULTI_GPU,
    reason="Requires >= 2 CUDA GPUs",
)

requires_fsdp2 = pytest.mark.skipif(
    not (MULTI_GPU and _FSDP2_AVAILABLE),
    reason="Requires >= 2 CUDA GPUs + PyTorch FSDP2 (torch >= 2.4)",
)

requires_fsdp2_triton = pytest.mark.skipif(
    not (MULTI_GPU and _FSDP2_AVAILABLE and CUDA_AND_TRITON),
    reason="Requires >= 2 CUDA GPUs + FSDP2 + Triton",
)

requires_fsdp2_mp = pytest.mark.skipif(
    not (MULTI_GPU and _FSDP2_MP_AVAILABLE),
    reason="Requires >= 2 CUDA GPUs + FSDP2 MixedPrecisionPolicy",
)

# =====================================================================
#  PRNG helpers
# =====================================================================


def seed_all(seed: int = 42):
    """Reset all PRNGs (Python, PyTorch, CUDA) to a known state."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
#  Shared models
# =====================================================================


class SimpleMLP(nn.Module):
    """Tiny MLP reused across distributed / mixed-precision tests."""

    def __init__(self, d_in=64, d_hidden=128, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# =====================================================================
#  Distributed test harness
# =====================================================================


def find_free_port() -> int:
    """Grab an ephemeral port that is (momentarily) free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_workers(fn, world_size: int = 2):
    """Launch *fn(rank, world_size)* on *world_size* GPU processes."""
    port = find_free_port()
    mp.spawn(
        _worker_entry,
        args=(world_size, port, fn),
        nprocs=world_size,
        join=True,
    )


def _worker_entry(rank: int, world_size: int, port: int, fn):
    """Per-process bootstrap: init NCCL, run test, tear down."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), f".triton_test_rank{rank}_{os.getpid()}",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    seed_all(42)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()
