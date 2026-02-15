"""LUMA — Log-space Unbiased Momentum Adam.

Memory-efficient drop-in replacement for AdamW that compresses optimizer
states to 4 bytes/param via preconditioner-domain quantization with
unbiased log-to-linear stochastic rounding.
"""

from luma_optimizer.config import get_kernel_config
from luma_optimizer.optimizer import LUMA

__all__ = ["LUMA", "get_kernel_config"]
__version__ = "0.1.0"
