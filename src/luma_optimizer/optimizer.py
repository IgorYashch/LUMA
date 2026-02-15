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

The Triton backend is fully asynchronous (zero CPU-GPU sync) and
compatible with external ``torch.cuda.CUDAGraph`` capture.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from . import functional as F
from .config import get_kernel_config

# Triton back-end is optional (CUDA-only)
try:
    from .triton_kernel import luma_triton_step  # noqa: F401

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    _TRITON_AVAILABLE = False


class LUMA(Optimizer):
    r"""LUMA: Log-space Unbiased Momentum AdamW.

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

        self._next_param_id = 0

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
    #  Checkpoint: portable save / load
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a backend-agnostic state dict for checkpointing.

        Triton-internal GPU buffers (``_old_scalars``, ``_new_grid``, etc.)
        are replaced with portable ``S_m`` / ``S_v`` floats so that
        checkpoints are interchangeable between Triton and PyTorch backends.
        """
        raw = super().state_dict()
        clean_state: dict = {}
        for idx, s in raw["state"].items():
            cs: dict = {}
            for k, v in s.items():
                if k.startswith("_"):
                    continue  # skip internal Triton buffers
                cs[k] = v
            # Triton path: extract S_m / S_v from GPU buffer
            if "_new_grid" in s:
                ng = s["_new_grid"]
                cs["S_m"] = ng[0].item()
                cs["S_v"] = ng[1].item()
            clean_state[idx] = cs
        raw["state"] = clean_state
        return raw

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state dict and reconstruct backend-specific buffers.

        Handles cross-backend loading: a checkpoint saved with the PyTorch
        backend can be loaded into a Triton-backed optimiser and vice versa.
        """
        super().load_state_dict(state_dict)

        max_param_id = -1
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue

                # Track highest param_id for _next_param_id
                pid = state.get("param_id", -1)
                if pid > max_param_id:
                    max_param_id = pid

                # Move quantised tensors to parameter's device if needed
                for key in ("Q_m", "Q_w"):
                    if key in state and state[key].device != p.device:
                        state[key] = state[key].to(p.device)

                # Reconstruct Triton buffers if missing
                if (
                    self._use_triton
                    and p.is_cuda
                    and "_old_scalars" not in state
                    and "S_m" in state
                ):
                    self._init_triton_buffers(
                        state, p,
                        state["S_m"], state["S_v"],
                        beta1, beta2,
                        step=state.get("step", 1),
                    )
                    # S_m/S_v now live inside _new_grid[0:2]; drop stale floats
                    state.pop("S_m", None)
                    state.pop("S_v", None)

        self._next_param_id = max_param_id + 1

    # ------------------------------------------------------------------
    #  Export dequantised state → AdamW format
    # ------------------------------------------------------------------

    def export_adamw_state(self) -> dict:
        """Export dequantised optimizer states as an AdamW-compatible state dict.

        Decodes the int16 quantised momentum and preconditioner back to
        FP32 ``exp_avg`` and ``exp_avg_sq`` tensors that can be loaded
        directly into ``torch.optim.AdamW``::

            adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
            adamw.load_state_dict(luma_opt.export_adamw_state())
        """
        # Build parameter index mapping (same order as param_groups)
        param_to_idx: dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_to_idx[id(p)] = idx
                idx += 1

        adamw_state: dict = {}
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue

                pidx = param_to_idx[id(p)]
                step = state["step"]

                # Get S_m, S_v (from GPU buffer or Python float)
                if "_new_grid" in state:
                    S_m = state["_new_grid"][0].item()
                    S_v = state["_new_grid"][1].item()
                else:
                    S_m = state["S_m"]
                    S_v = state["S_v"]

                # Decode quantised states → FP32
                w_min, delta_m, _, delta_w, _ = F._precompute(S_m, S_v, eps)
                exp_avg = F._decode_momentum(state["Q_m"], delta_m)
                w = F._decode_preconditioner(state["Q_w"], w_min, delta_w)
                exp_avg_sq = (1.0 / w - eps).square()

                adamw_state[pidx] = {
                    "step": torch.tensor(float(step)),
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                }

        # Build param_groups (same hyper-params, integer param refs)
        param_groups = []
        for group in self.param_groups:
            pg = {k: v for k, v in group.items() if k != "params"}
            pg["params"] = [param_to_idx[id(p)] for p in group["params"]]
            param_groups.append(pg)

        return {"state": adamw_state, "param_groups": param_groups}

    # ------------------------------------------------------------------
    #  Internal: buffer allocation
    # ------------------------------------------------------------------

    def _init_triton_buffers(
        self,
        state: dict,
        p: torch.Tensor,
        S_m: float,
        S_v: float,
        beta1: float,
        beta2: float,
        step: int = 1,
    ) -> None:
        """Allocate pre-allocated GPU buffers for the Triton pipeline.

        Called once per parameter at step 1 (or after ``load_state_dict``).
        Buffer addresses are fixed across steps.
        """
        kcfg = get_kernel_config(p.device)
        block_size = kcfg["BLOCK_SIZE"]
        num_blocks = (p.numel() + block_size - 1) // block_size

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        # old_scalars: [w_min, delta_m, delta_w, eta_t, bc2, bc1, step, reserved]
        state["_old_scalars"] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, bc2, bc1, float(step), 0.0],
            device=p.device,
            dtype=torch.float32,
        )
        # new_grid: [S_m, S_v, delta_m, z_m, w_min, delta_w, z_w, w_max]
        state["_new_grid"] = torch.zeros(
            8, device=p.device, dtype=torch.float32,
        )
        state["_new_grid"][0] = S_m
        state["_new_grid"][1] = S_v
        # Block-max reduction buffers
        state["_s_m_block"] = torch.empty(
            num_blocks, device=p.device, dtype=torch.float32,
        )
        state["_s_v_block"] = torch.empty(
            num_blocks, device=p.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    #  Main entry point
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
                    state["param_id"] = self._next_param_id
                    self._next_param_id += 1

                state["step"] += 1
                t = state["step"]

                if t == 1:
                    # First step: pure FP32 + seed delayed scales
                    Q_m, Q_w, S_m, S_v = F.luma_init_step(
                        p, grad, beta1, beta2, eps, lr, wd,
                    )
                    state["Q_m"] = Q_m
                    state["Q_w"] = Q_w

                    if self._use_triton and p.is_cuda:
                        self._init_triton_buffers(
                            state, p, S_m, S_v, beta1, beta2,
                        )
                    else:
                        state["S_m"] = S_m
                        state["S_v"] = S_v
                else:
                    # Quantised step — Q_m / Q_w modified in-place
                    if self._use_triton and p.is_cuda:
                        luma_triton_step(
                            p, grad,
                            state["Q_m"], state["Q_w"],
                            state["_old_scalars"], state["_new_grid"],
                            state["_s_m_block"], state["_s_v_block"],
                            beta1, beta2, eps, lr, wd,
                            state["param_id"],
                        )
                        # S_m/S_v live in _new_grid[0:2] — no CPU sync
                    else:
                        S_m, S_v = F.luma_update_step(
                            p, grad,
                            state["Q_m"], state["Q_w"],
                            state["S_m"], state["S_v"],
                            t, beta1, beta2, eps, lr, wd,
                        )
                        state["S_m"] = S_m
                        state["S_v"] = S_v

        return loss
