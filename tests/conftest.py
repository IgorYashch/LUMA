import pytest
import torch

from utils import CUDA_AND_TRITON, GPU_BF16, seed_all  # noqa: F401

# ── Parametrize lists (names mirror the fixtures they feed) ──────────────────

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_DEVICE_BACKENDS = (
    [("cpu", "pytorch")]
    + ([("cuda", "pytorch")] if torch.cuda.is_available() else [])
    + ([("cuda", "triton")] if CUDA_AND_TRITON else [])
)

_AUTOCAST_CFGS = (
    [("cpu", torch.bfloat16)]
    + ([("cuda", torch.float16)] if torch.cuda.is_available() else [])
    + ([("cuda", torch.bfloat16)] if GPU_BF16 else [])
)

_DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _seed():
    seed_all(42)


@pytest.fixture(params=_DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=_DEVICE_BACKENDS, ids=lambda c: f"{c[0]}-{c[1]}")
def device_backend(request):
    return torch.device(request.param[0]), request.param[1]


@pytest.fixture(
    params=_AUTOCAST_CFGS,
    ids=lambda c: f"{c[0]}-{_DTYPE_NAMES[c[1]]}",
)
def autocast_cfg(request):
    """Return ``(device_type, amp_dtype)`` for autocast tests."""
    return request.param
