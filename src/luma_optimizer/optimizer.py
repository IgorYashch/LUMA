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
from .config import SCALE_FLOOR_M, get_kernel_config

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

        self._base_seed: int = int(torch.randint(0x7FFFFFFF, ()).item())

        self._pt_gen: torch.Generator | None = None

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
            # Triton path: extract S_m / S_v and true step from GPU
            if "_new_grid" in s:
                ng = s["_new_grid"]
                cs["S_m"] = ng[0].item()
                cs["S_v"] = ng[1].item()
            if "_step_tensor" in s:
                cs["step"] = int(s["_step_tensor"].item())
            clean_state[idx] = cs
        raw["state"] = clean_state
        raw["_base_seed"] = self._base_seed
        return raw

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state dict and reconstruct backend-specific buffers.

        Handles cross-backend loading: a checkpoint saved with the PyTorch
        backend can be loaded into a Triton-backed optimiser and vice versa.
        """
        self._base_seed = state_dict.get("_base_seed", self._base_seed)
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
                param = self._unwrap_tensor(p)
                if (
                    self._use_triton
                    and param.is_cuda
                    and "_old_scalars" not in state
                    and "S_m" in state
                ):
                    self._init_triton_buffers(
                        state, param,
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

                # True step lives in _step_tensor on Triton path
                if "_step_tensor" in state:
                    step = int(state["_step_tensor"].item())
                else:
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
    #  Import from AdamW
    # ------------------------------------------------------------------

    def import_adamw_state(self, adamw_sd: dict) -> None:
        """Import optimizer state from a ``torch.optim.AdamW`` state dict.

        Quantises FP32 ``exp_avg`` and ``exp_avg_sq`` tensors into LUMA's
        int16 format, preserving the step counter so that bias-correction
        continues seamlessly from where AdamW left off.

        Usage::

            adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
            # ... train with AdamW ...
            luma = LUMA(model.parameters(), lr=1e-3)
            luma.import_adamw_state(adamw.state_dict())
            # ... continue training with LUMA ...

        Note
        ----
        Hyper-parameters (``lr``, ``betas``, ``eps``, ``weight_decay``) are
        **not** imported from the AdamW state dict — set them when
        constructing the LUMA optimiser.
        """
        # Build flat param list in param_group order, with group reference
        all_params: list[torch.Tensor] = []
        param_group_map: dict[int, dict] = {}
        for group in self.param_groups:
            for p in group["params"]:
                param_group_map[len(all_params)] = group
                all_params.append(p)

        self._next_param_id = 0

        for idx_key, adamw_state in adamw_sd["state"].items():
            idx = int(idx_key) if isinstance(idx_key, str) else idx_key
            if idx >= len(all_params):
                raise ValueError(
                    f"AdamW state has index {idx} but LUMA only has "
                    f"{len(all_params)} parameters"
                )

            p = all_params[idx]
            group = param_group_map[idx]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            # Extract step
            step_val = adamw_state["step"]
            step = int(
                step_val.item() if isinstance(step_val, torch.Tensor) else step_val
            )

            # Extract FP32 states
            param = self._unwrap_tensor(p)
            device = param.device if param is not None else p.device
            m = adamw_state["exp_avg"].to(device).float()
            v = adamw_state["exp_avg_sq"].to(device).float().clamp(min=eps * eps)

            # Compute delayed scales
            S_m: float = max(m.abs().max().item(), SCALE_FLOOR_M)
            S_v: float = max(v.max().item(), eps * eps)

            # Assign param_id first so we can derive a deterministic seed
            param_id = self._next_param_id
            self._next_param_id += 1

            # Quantise onto LUMA grid (deterministic stochastic rounding)
            gen = self._get_generator(
                device, F._make_step_seed(self._base_seed, 0, param_id),
            )

            w_min, delta_m, z_m, delta_w, z_w = F._precompute(S_m, S_v, eps)
            Q_m = F._quantize_momentum(m, S_m, delta_m, z_m, generator=gen)
            Q_w = F._quantize_preconditioner(v, eps, w_min, delta_w, z_w, generator=gen)

            state = self.state[p]
            state["step"] = step
            state["param_id"] = param_id
            state["Q_m"] = Q_m
            state["Q_w"] = Q_w

            if self._use_triton and param is not None and param.is_cuda:
                self._init_triton_buffers(
                    state, param, S_m, S_v, beta1, beta2, step=step,
                )
            else:
                state["S_m"] = S_m
                state["S_v"] = S_v

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

        # old_scalars: [w_min, delta_m, delta_w, eta_t, bc2, bc1, reserved, reserved]
        state["_old_scalars"] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, bc2, bc1, 0.0, 0.0],
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
        # Step counter as int32 on GPU — survives CUDA Graph capture
        # (scalar Python args are baked at capture time) and avoids
        # the float32 precision ceiling at 2²⁴.
        state["_step_tensor"] = torch.tensor(
            [step], device=p.device, dtype=torch.int32,
        )

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        gen = self._pt_gen
        if gen is None or gen.device != device:
            gen = torch.Generator(device=device)
            self._pt_gen = gen
        gen.manual_seed(seed)
        return gen

    # ------------------------------------------------------------------
    #  DTensor / FSDP2 support
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
        """Unwrap a DTensor to its local shard, or return a plain tensor as-is.
        """
        if t is None:
            return None
        if hasattr(t, "_local_tensor"):
            return t._local_tensor
        return t

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

                # Unwrap DTensor (FSDP2) → local shard.
                param = self._unwrap_tensor(p)
                grad = self._unwrap_tensor(p.grad)

                if param.numel() == 0:
                    continue  # empty FSDP shard on this rank

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
                    # First step: pure FP32 + seed delayed scales.
                    # Always runs via PyTorch; use a seeded Generator
                    # so that stochastic rounding is deterministic
                    # across DDP ranks (given identical _base_seed).
                    gen = self._get_generator(
                        param.device,
                        F._make_step_seed(self._base_seed, t, state["param_id"]),
                    )
                    Q_m, Q_w, S_m, S_v = F.luma_init_step(
                        param, grad, beta1, beta2, eps, lr, wd,
                        generator=gen,
                    )
                    state["Q_m"] = Q_m
                    state["Q_w"] = Q_w

                    if self._use_triton and param.is_cuda:
                        self._init_triton_buffers(
                            state, param, S_m, S_v, beta1, beta2,
                        )
                    else:
                        state["S_m"] = S_m
                        state["S_v"] = S_v
                else:
                    # Quantised step — Q_m / Q_w modified in-place
                    if self._use_triton and param.is_cuda:
                        luma_triton_step(
                            param, grad,
                            state["Q_m"], state["Q_w"],
                            state["_old_scalars"], state["_new_grid"],
                            state["_s_m_block"], state["_s_v_block"],
                            beta1, beta2, eps, lr, wd,
                            state["param_id"],
                            state["_step_tensor"],
                            self._base_seed,
                        )
                        # S_m/S_v live in _new_grid[0:2] — no CPU sync
                    else:
                        gen = self._get_generator(
                            param.device,
                            F._make_step_seed(self._base_seed, t, state["param_id"]),
                        )
                        S_m, S_v = F.luma_update_step(
                            param, grad,
                            state["Q_m"], state["Q_w"],
                            state["S_m"], state["S_v"],
                            t, beta1, beta2, eps, lr, wd,
                            generator=gen,
                        )
                        state["S_m"] = S_m
                        state["S_v"] = S_v

        return loss
