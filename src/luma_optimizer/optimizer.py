"""LUMA optimizer — ``torch.optim.Optimizer`` interface.

Usage::

    from luma_optimizer import LUMA

    opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)
    for data, target in loader:
        opt.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        opt.step()

The optimizer stores two int16 tensors per parameter (momentum ``Q_m``
and preconditioner ``Q_w``), giving a 4-byte/param state footprint —
exactly half of FP32 AdamW (8 bytes/param).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from . import functional as F

# Triton back-end is optional (CUDA-only)
try:
    from .triton_kernel import luma_triton_step  # noqa: F401

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    _TRITON_AVAILABLE = False


class LUMA(Optimizer):
    r"""LUMA: Log-space Unbiased Momentum Adam.

    Memory-efficient drop-in replacement for AdamW.  Compresses optimizer
    states to **4 bytes / param** (two int16 tensors) via
    preconditioner-domain quantisation with unbiased stochastic rounding.

    Parameters
    ----------
    params :
        Iterable of parameters or parameter-group dicts.
    lr : float
        Learning rate (default ``1e-3``).
    betas : tuple[float, float]
        EMA coefficients ``(β₁, β₂)`` (default ``(0.9, 0.999)``).
    eps : float
        Denominator term for numerical stability (default ``1e-8``).
    weight_decay : float
        Decoupled weight-decay coefficient (default ``0.01``).
    backend : str
        ``"auto"`` | ``"triton"`` | ``"pytorch"``.
        *auto* selects Triton when CUDA + Triton are available.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        backend: str = "auto",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        if backend == "auto":
            self._use_triton = _TRITON_AVAILABLE and torch.cuda.is_available()
        elif backend == "triton":
            if not _TRITON_AVAILABLE:
                raise RuntimeError(
                    "Triton is not available. Install with: pip install triton"
                )
            self._use_triton = True
        else:
            self._use_triton = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LUMA does not support sparse gradients")

                state = self.state[p]

                # ── initialise on first encounter ────────────────────
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1
                t = state["step"]

                if t == 1:
                    # First step: pure FP32 + seed delayed scales
                    Q_m, Q_w, S_m, S_v = F.luma_init_step(
                        p, grad, beta1, beta2, eps, lr, wd,
                    )
                    state["Q_m"] = Q_m
                    state["Q_w"] = Q_w
                    state["S_m"] = S_m
                    state["S_v"] = S_v
                else:
                    # Quantised step (Triton or PyTorch fallback)
                    if self._use_triton and p.is_cuda:
                        S_m, S_v = luma_triton_step(
                            p,
                            grad,
                            state["Q_m"],
                            state["Q_w"],
                            state["S_m"],
                            state["S_v"],
                            t,
                            beta1,
                            beta2,
                            eps,
                            lr,
                            wd,
                        )
                        state["S_m"] = S_m
                        state["S_v"] = S_v
                        # Q_m / Q_w modified in-place by the kernel
                    else:
                        Q_m, Q_w, S_m, S_v = F.luma_update_step(
                            p,
                            grad,
                            state["Q_m"],
                            state["Q_w"],
                            state["S_m"],
                            state["S_v"],
                            t,
                            beta1,
                            beta2,
                            eps,
                            lr,
                            wd,
                        )
                        state["Q_m"] = Q_m
                        state["Q_w"] = Q_w
                        state["S_m"] = S_m
                        state["S_v"] = S_v

        return loss
