"""Shared fixtures for the LUMA test-suite."""

import pytest
import torch


def _available_devices() -> list[str]:
    """Return device strings for every backend we can test on."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(autouse=True)
def seed_rng():
    """Fix the global PRNG for reproducibility across all tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(params=_available_devices(), ids=lambda d: d)
def device(request):
    """Parametrised device fixture — tests run once per available device."""
    return torch.device(request.param)
