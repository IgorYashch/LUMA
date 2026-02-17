import pytest
import torch

from utils import CUDA_AND_TRITON, seed_all  # noqa: F401

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_CONFIGS = (
    [("cpu", "pytorch")]
    + ([("cuda", "pytorch")] if torch.cuda.is_available() else [])
    + ([("cuda", "triton")] if CUDA_AND_TRITON else [])
)


@pytest.fixture(autouse=True)
def _seed():
    seed_all(42)


@pytest.fixture(params=_DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=_CONFIGS, ids=lambda c: f"{c[0]}-{c[1]}")
def device_backend(request):
    return torch.device(request.param[0]), request.param[1]
