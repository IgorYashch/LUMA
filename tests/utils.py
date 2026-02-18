"""Shared test utilities for the LUMA test-suite."""

import random

import pytest
import torch

# ── Triton + CUDA availability (single source of truth) ─────────────────────
_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    pass

CUDA_AND_TRITON = torch.cuda.is_available() and _TRITON_AVAILABLE
MULTI_GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
GPU_BF16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
)

requires_cuda_triton = pytest.mark.skipif(
    not CUDA_AND_TRITON,
    reason="Requires CUDA + Triton",
)

requires_multi_gpu = pytest.mark.skipif(
    not MULTI_GPU,
    reason="Requires >= 2 CUDA GPUs",
)


def seed_all(seed: int = 42):
    """Reset all PRNGs (Python, PyTorch, CUDA) to a known state."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
