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


def _valid_device_backend_configs() -> list[tuple[str, str]]:
    """Return only valid (device, backend) pairs — no impossible combos."""
    configs = [("cpu", "pytorch")]
    if torch.cuda.is_available():
        configs.append(("cuda", "pytorch"))
    if CUDA_AND_TRITON:
        configs.append(("cuda", "triton"))
    return configs


@pytest.fixture(autouse=True)
def seed_rng():
    """Fix all PRNGs for full reproducibility across all tests."""
    seed_all(42)


@pytest.fixture(params=_available_devices(), ids=lambda d: d)
def device(request):
    """Parametrised device fixture — tests run once per available device."""
    return torch.device(request.param)


@pytest.fixture(
    params=_valid_device_backend_configs(),
    ids=lambda c: f"{c[0]}-{c[1]}",
)
def device_backend(request):
    """Yields only valid (device, backend) pairs — zero skips."""
    return torch.device(request.param[0]), request.param[1]
