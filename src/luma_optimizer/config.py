"""Kernel tuning configuration for the LUMA optimizer.

Default values are chosen to work correctly across NVIDIA GPU generations
from Volta (V100) through Blackwell (B200).  The helper
:func:`get_kernel_config` refines them per-architecture when a CUDA device
is available.

Quantization grid constants (``K_M``, ``K_W``, ``SCALE_FLOOR_M``) are
defined here so that *functional.py* and *triton_kernel.py* share a single
source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# =====================================================================
#  Quantization grid constants
# =====================================================================
K_M: int = (1 << 15) - 1          # 32 767  — signed-int16 magnitude
K_W: int = (1 << 16) - 1          # 65 535  — unsigned-int16 range
SCALE_FLOOR_M: float = 1e-9       # minimum delayed momentum scale

# =====================================================================
#  Triton kernel launch defaults
# =====================================================================
DEFAULT_BLOCK_SIZE: int = 1024
DEFAULT_NUM_WARPS: int = 4
DEFAULT_NUM_STAGES: int = 3


def get_kernel_config(device: "torch.device | None" = None) -> dict[str, int]:
    """Return Triton launch parameters tuned for the target GPU.

    Returns a dict with keys ``BLOCK_SIZE``, ``num_warps``, ``num_stages``.

    Architecture heuristics
    -----------------------
    ============  ========  ===========  ===========  ======================
    GPU family    SM major  num_warps    num_stages   rationale
    ============  ========  ===========  ===========  ======================
    Volta  V100   7         4            1            no async-copy HW
    Turing T4     7         4            1            same as Volta
    Ampere A100   8         4            3            good balance
    Hopper H100   9         8            3            higher HBM3 BW
    Blackwell B200 10+      8            3            even higher HBM3e BW
    ============  ========  ===========  ===========  ======================

    ``BLOCK_SIZE = 1024`` is kept constant: 1024 elements × ~24 B traffic
    ≈ 24 KB per block, which fits L1/smem on every generation and gives a
    good ratio of useful work to launch overhead.
    """
    import torch as _torch

    cfg: dict[str, int] = {
        "BLOCK_SIZE": DEFAULT_BLOCK_SIZE,
        "num_warps": DEFAULT_NUM_WARPS,
        "num_stages": DEFAULT_NUM_STAGES,
    }

    if device is None or device.type != "cuda" or not _torch.cuda.is_available():
        return cfg

    major, _ = _torch.cuda.get_device_capability(device)
    if major >= 9:          # Hopper / Blackwell  — saturate higher BW
        cfg["num_warps"] = 8
    elif major < 8:         # Volta / Turing      — no software pipelining
        cfg["num_stages"] = 1

    return cfg
