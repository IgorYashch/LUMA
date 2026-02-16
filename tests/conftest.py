"""Shared fixtures for the LUMA test-suite."""

import pytest
import torch

from utils import CUDA_AND_TRITON, seed_all  # noqa: F401


def _available_devices() -> list[str]:
    """Return device strings for every backend we can test on."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _available_backends() -> list[str]:
    """Return optimizer backends available in this environment."""
    backends = ["pytorch"]
    if CUDA_AND_TRITON:
        backends.append("triton")
    return backends


@pytest.fixture(autouse=True)
def seed_rng():
    """Fix all PRNGs for full reproducibility across all tests."""
    seed_all(42)


@pytest.fixture(params=_available_devices(), ids=lambda d: d)
def device(request):
    """Parametrised device fixture — tests run once per available device."""
    return torch.device(request.param)


@pytest.fixture(params=_available_backends(), ids=lambda b: b)
def backend(request, device):
    """Parametrised backend fixture — skips invalid device+backend combos."""
    if request.param == "triton" and device.type != "cuda":
        pytest.skip("Triton requires CUDA")
    return request.param
