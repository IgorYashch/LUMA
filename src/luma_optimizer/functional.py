"""LUMA optimizer — pure-PyTorch functional implementation.

Implements the preconditioner-domain quantization with log-to-linear
corrected stochastic rounding (Algorithm 1 from the paper).

All heavy arithmetic (log1p, expm1, exp, PRNG) runs element-wise on the
compute device; the small set of per-shard scalars (Δ_m, Δ_w, Z_m, Z_w,
w_min) is precomputed on the host CPU in Python ``math``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import K_M, K_W, SCALE_FLOOR_M  # single source of truth

# ---------------------------------------------------------------------------
# uint16 helpers  (store unsigned 0..65535 in torch.int16 via bit wrapping)
# ---------------------------------------------------------------------------

def _encode_uint16(values: Tensor) -> Tensor:
    """Encode unsigned int values [0, 65535] → ``torch.int16`` storage."""
    return values.to(torch.int16)  # two's-complement wrap-around


def _decode_uint16(q: Tensor) -> Tensor:
    """Decode ``torch.int16`` storage → float values [0, 65535]."""
    return (q.to(torch.int32) & 0xFFFF).float()


# ---------------------------------------------------------------------------
# Log-to-linear corrected stochastic rounding  (LogSR)
# ---------------------------------------------------------------------------

def _log_sr(
    y: Tensor,
    delta: float,
    z_denom: float,
    rand: Tensor,
    k_max: int,
) -> Tensor:
    """Log-to-linear corrected stochastic rounding.

    Given a continuous log-domain bin index *y*, stochastically round to an
    integer grid point so that the expected *linear-domain* reconstruction is
    exactly unbiased (neutralises Jensen's-inequality bias).

    Parameters
    ----------
    y : Tensor   – continuous bin indices  (≥ 0)
    delta : float – grid step size  (Δ)
    z_denom : float – ``expm1(Δ)`` precomputed on host
    rand : Tensor – uniform samples in [0, 1)
    k_max : int   – upper clamp for the integer output
    """
    if delta <= 0.0:
        return torch.zeros_like(y)

    floor_y = y.floor()
    frac = y - floor_y
    # Corrected probability  p* = expm1(frac · Δ) / expm1(Δ)
    p_star = torch.expm1(frac * delta) / z_denom
    rounded = floor_y + (rand < p_star).to(y.dtype)
    return rounded.clamp(0, k_max)


# ---------------------------------------------------------------------------
# Host-side precompute  (one call per FSDP shard per step)
# ---------------------------------------------------------------------------

def _precompute(
    S_m: float,
    S_v: float,
    eps: float,
) -> tuple[float, float, float, float, float]:
    """Return ``(w_min, delta_m, z_m, delta_w, z_w)``."""
    w_min = 1.0 / (math.sqrt(S_v) + eps)

    delta_m = math.log1p(S_m) / K_M
    z_m = math.expm1(delta_m)

    ratio = 1.0 / (2.0 * eps * w_min)
    if ratio > 1.0:
        delta_w = math.log(ratio) / K_W
    else:
        delta_w = 0.0
    z_w = math.expm1(delta_w) if delta_w > 0.0 else 1.0

    return w_min, delta_m, z_m, delta_w, z_w


# ---------------------------------------------------------------------------
# Deterministic per-step seed  (mirrors Triton kernel B convention)
# ---------------------------------------------------------------------------

def _make_step_seed(base_seed: int, step: int, param_id: int) -> int:
    """Return a deterministic 31-bit seed for stochastic rounding.

    Uses the same LCG-XOR formula as Triton kernel B so that,
    given identical ``(base_seed, step, param_id)``, both backends
    produce a reproducible (though not identical) PRNG stream.
    """
    return ((step * 1103515245 + base_seed) ^ (param_id * 22695477)) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Decode quantised states  →  FP32 continuous values
# ---------------------------------------------------------------------------

def _decode_momentum(Q_m: Tensor, delta_m: float) -> Tensor:
    """``m = sgn(Q_m) · expm1(|Q_m| · Δ_m)``"""
    q = Q_m.float()
    return q.sign() * torch.expm1(q.abs() * delta_m)


def _decode_preconditioner(
    Q_w: Tensor,
    w_min: float,
    delta_w: float,
) -> Tensor:
    """``w = w_min · exp(Q_w · Δ_w)``"""
    q = _decode_uint16(Q_w)
    return w_min * torch.exp(q * delta_w)


# ---------------------------------------------------------------------------
# Quantise (encode) continuous values  →  int16 grid
# ---------------------------------------------------------------------------

def _quantize_momentum(
    m: Tensor,
    S_m: float,
    delta_m: float,
    z_m: float,
    *,
    out: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    m_clip = m.clamp(-S_m, S_m)
    y = torch.log1p(m_clip.abs()) / delta_m
    rand = torch.rand(y.shape, dtype=y.dtype, device=y.device, generator=generator)
    q_mag = _log_sr(y, delta_m, z_m, rand, K_M)
    q = (m_clip.sign() * q_mag).to(torch.int16)
    if out is not None:
        out.copy_(q)
        return out
    return q


def _quantize_preconditioner(
    v: Tensor,
    eps: float,
    w_min: float,
    delta_w: float,
    z_w: float,
    *,
    out: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    w = 1.0 / (v.sqrt() + eps)
    w_max = 1.0 / (2.0 * eps)
    w_clip = w.clamp(w_min, w_max)
    if delta_w > 0.0:
        y = torch.log(w_clip / w_min) / delta_w
    else:
        y = torch.zeros_like(w_clip)
    rand = torch.rand(y.shape, dtype=y.dtype, device=y.device, generator=generator)
    q = _log_sr(y, delta_w, z_w, rand, K_W)
    q_int = _encode_uint16(q.to(torch.int32))
    if out is not None:
        out.copy_(q_int)
        return out
    return q_int


# ===================================================================== #
#  Full step functions
# ===================================================================== #

def luma_init_step(
    param: Tensor,
    grad: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, float, float]:
    """Step 0 — initialise in full FP32, then quantise & seed delayed scales.

    Parameters must be float32.  Gradients may be any floating-point dtype
    (e.g. fp16 / bf16 from ``torch.autocast``) and are promoted internally.

    Returns ``(Q_m, Q_w, S_m, S_v)``.
    """
    grad_fp32 = grad.float()

    # ── raw EMA from zero-init ──────────────────────────────────────────
    m = (1.0 - beta1) * grad_fp32
    v = ((1.0 - beta2) * grad_fp32.square()).clamp(min=eps * eps)

    # ── bias-corrected parameter update  (step = 1) ────────────────────
    bc1 = 1.0 - beta1
    bc2 = 1.0 - beta2
    eta_t = lr / bc1

    param.mul_(1.0 - lr * weight_decay)
    param.addcdiv_(m, (v / bc2).sqrt().add_(eps), value=-eta_t)

    # ── seed delayed scales ─────────────────────────────────────────────
    S_m: float = max(m.abs().max().item(), SCALE_FLOOR_M)
    S_v: float = max(v.max().item(), eps * eps)

    # ── quantise ────────────────────────────────────────────────────────
    w_min, delta_m, z_m, delta_w, z_w = _precompute(S_m, S_v, eps)
    Q_m = _quantize_momentum(m, S_m, delta_m, z_m, generator=generator)
    Q_w = _quantize_preconditioner(v, eps, w_min, delta_w, z_w, generator=generator)

    return Q_m, Q_w, S_m, S_v


def luma_update_step(
    param: Tensor,
    grad: Tensor,
    Q_m: Tensor,
    Q_w: Tensor,
    S_m: float,
    S_v: float,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Steps >= 2 — decode → update → re-quantise (matched scales).

    Two-pass approach: compute new scales first, then quantise on the
    matching grid.  This eliminates the encode-decode drift of single-pass
    delayed scaling.

    Parameters must be float32.  Gradients may be any floating-point dtype
    (e.g. fp16 / bf16 from ``torch.autocast``) and are promoted internally.

    Modifies *param*, *Q_m*, *Q_w* **in-place**.
    Returns ``(S_m_next, S_v_next)``.
    """
    grad_fp32 = grad.float()

    # ── host precompute ─────────────────────────────────────────────────
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    eta_t = lr / bc1
    w_min, delta_m, z_m, delta_w, z_w = _precompute(S_m, S_v, eps)

    # ── 1. decode states (already float32) ──────────────────────────────
    m = _decode_momentum(Q_m, delta_m)
    w = _decode_preconditioner(Q_w, w_min, delta_w)
    v = (1.0 / w - eps).square()

    # ── 2. EMA + param update (exact PyTorch AdamW) ─────────────────────
    m_new = beta1 * m + (1.0 - beta1) * grad_fp32
    v_new = (beta2 * v + (1.0 - beta2) * grad_fp32.square()).clamp(min=eps * eps)

    param.mul_(1.0 - weight_decay * lr)
    param.addcdiv_(m_new, (v_new / bc2).sqrt().add_(eps), value=-eta_t)

    # ── 3. track new scales ─────────────────────────────────────────────
    S_m_next: float = max(m_new.abs().max().item(), SCALE_FLOOR_M)
    S_v_next: float = max(v_new.max().item(), eps * eps)

    # ── 4. re-quantise with NEW matched scales (in-place) ───────────────
    #   Two-pass: we already know S_m_next/S_v_next, so we build the NEW
    #   grid and quantise on it.  At the next step decode uses the same
    #   grid → exact encode-decode match, zero systematic drift.
    w_min_new, delta_m_new, z_m_new, delta_w_new, z_w_new = _precompute(
        S_m_next, S_v_next, eps,
    )
    _quantize_momentum(m_new, S_m_next, delta_m_new, z_m_new, out=Q_m, generator=generator)
    _quantize_preconditioner(v_new, eps, w_min_new, delta_w_new, z_w_new, out=Q_w, generator=generator)

    return S_m_next, S_v_next
