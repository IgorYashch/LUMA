"""LUMA — Log-space Unbiased Momentum AdamW.

Memory-efficient drop-in replacement for AdamW that compresses optimizer
states to 4 bytes/param via preconditioner-domain quantization with
unbiased log-to-linear stochastic rounding.
"""

from importlib.metadata import PackageNotFoundError, version

from luma_optimizer.config import get_kernel_config
from luma_optimizer.optimizer import LUMA

__all__ = ["LUMA", "get_kernel_config"]

try:
    __version__ = version("luma-optimizer")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
