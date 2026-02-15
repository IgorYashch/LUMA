"""LUMA optimizer — Triton two-kernel implementation (CUDA only).

Uses two lean kernels with **zero temporary allocation**:

    Kernel A  (decode + EMA + param update + block-max tracking)
          ↓
    Python:  reduce block maxima → new grid params
          ↓
    Kernel B  (re-decode + re-EMA + quantize with NEW grid)

By recomputing the decode+EMA in Kernel B (cheap ALU hidden behind HBM
latency on every GPU generation), we avoid any float32 temp buffers while
guaranteeing exact encode-decode scale matching at every step.

The only costs vs. a hypothetical single-pass kernel are ~40 % additional
HBM traffic and one extra kernel launch (~5 μs).

Numerical notes
---------------
* ``expm1(x)`` and ``log1p(x)`` are approximated as ``exp(x) - 1`` and
  ``log(1 + x)`` respectively.  For the tiny grid steps Δ ≈ 1e-4 used in
  practice the relative error is < 1e-3, well within the stochastic
  rounding noise floor.
* The PRNG uses Triton's built-in Philox generator.  Momentum and
  preconditioner streams are made independent by offsetting the counter
  by ``n_elements``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .config import K_M, K_W, SCALE_FLOOR_M, get_kernel_config


# =====================================================================
#  Kernel A — decode, EMA, param update, block-level scale maxima
# =====================================================================
@triton.jit
def _luma_decode_update_kernel(
    # ── tensor pointers ──────────────────────────────────────────────
    param_ptr,
    grad_ptr,
    q_m_ptr,          # read-only (not overwritten)
    q_w_ptr,          # read-only (not overwritten)
    # ── per-block reduction output ───────────────────────────────────
    s_m_block_ptr,
    s_v_block_ptr,
    # ── scalar hyper-parameters ──────────────────────────────────────
    beta1,
    beta2,
    eps,
    lr,
    weight_decay,
    eta_t,            # lr / (1 - beta1^t)
    bc2,              # 1 - beta2^t
    # ── OLD grid constants (for decode) ──────────────────────────────
    w_min,
    delta_m,
    delta_w,
    # ── tensor length ────────────────────────────────────────────────
    n_elements,
    # ── compile-time constant ────────────────────────────────────────
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ── Load from HBM ────────────────────────────────────────────────
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad  = tl.load(grad_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
    q_m_i32 = tl.load(q_m_ptr + offsets, mask=mask, other=0).to(tl.int32)
    q_w_i32 = tl.load(q_w_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # ── 1. Decode quantised states → FP32 ────────────────────────────
    q_m_abs = tl.abs(q_m_i32).to(tl.float32)
    m_sign  = tl.where(q_m_i32 > 0, 1.0,
                       tl.where(q_m_i32 < 0, -1.0, 0.0))
    m = m_sign * (tl.exp(q_m_abs * delta_m) - 1.0)

    q_w_unsigned = (q_w_i32 & 0xFFFF).to(tl.float32)
    w     = w_min * tl.exp(q_w_unsigned * delta_w)
    inv_w = 1.0 / w
    v     = (inv_w - eps) * (inv_w - eps)

    # ── 2. EMA update ────────────────────────────────────────────────
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = tl.maximum(beta2 * v + (1.0 - beta2) * grad * grad,
                       eps * eps)

    # ── 3. Parameter update (AdamW) ──────────────────────────────────
    denom = tl.sqrt(v_new / bc2) + eps
    param = param * (1.0 - weight_decay * lr) - eta_t * m_new / denom
    # float32 → param's native dtype auto-converted by Triton
    tl.store(param_ptr + offsets, param, mask=mask)

    # ── 4. Block-level scale tracking (mask phantom tail elements) ───
    abs_m_safe = tl.where(mask, tl.abs(m_new), 0.0)
    v_new_safe = tl.where(mask, v_new, 0.0)
    tl.store(s_m_block_ptr + pid, tl.max(abs_m_safe, axis=0))
    tl.store(s_v_block_ptr + pid, tl.max(v_new_safe, axis=0))

    # Q_m / Q_w are intentionally NOT written — Kernel B will do that.


# =====================================================================
#  Kernel B — re-decode, re-EMA, quantize with NEW matched grid
# =====================================================================
@triton.jit
def _luma_requantize_kernel(
    # ── tensor pointers ──────────────────────────────────────────────
    grad_ptr,         # still in HBM from backward()
    q_m_ptr,          # OLD values read, then overwritten in-place
    q_w_ptr,          # OLD values read, then overwritten in-place
    # ── EMA coefficients ─────────────────────────────────────────────
    beta1,
    beta2,
    eps,
    # ── OLD grid (for re-decode) ─────────────────────────────────────
    w_min_old,
    delta_m_old,
    delta_w_old,
    # ── NEW grid (for quantise) ──────────────────────────────────────
    delta_m_new,
    z_m_new,
    w_min_new,
    delta_w_new,
    z_w_new,
    w_max,
    # ── PRNG ─────────────────────────────────────────────────────────
    seed,
    # ── tensor length ────────────────────────────────────────────────
    n_elements,
    # ── compile-time constant ────────────────────────────────────────
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ── Load grad + OLD quantised states ─────────────────────────────
    grad    = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q_m_i32 = tl.load(q_m_ptr + offsets, mask=mask, other=0).to(tl.int32)
    q_w_i32 = tl.load(q_w_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # ── Re-decode with OLD grid (same math as Kernel A) ──────────────
    q_m_abs = tl.abs(q_m_i32).to(tl.float32)
    m_sign  = tl.where(q_m_i32 > 0, 1.0,
                       tl.where(q_m_i32 < 0, -1.0, 0.0))
    m = m_sign * (tl.exp(q_m_abs * delta_m_old) - 1.0)

    q_w_unsigned = (q_w_i32 & 0xFFFF).to(tl.float32)
    w     = w_min_old * tl.exp(q_w_unsigned * delta_w_old)
    inv_w = 1.0 / w
    v     = (inv_w - eps) * (inv_w - eps)

    # ── Re-compute EMA (same math as Kernel A — free ALU) ────────────
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = tl.maximum(beta2 * v + (1.0 - beta2) * grad * grad,
                       eps * eps)

    # ================================================================
    # Quantise momentum with NEW grid
    # ================================================================
    m_abs = tl.abs(m_new)
    y_m = tl.log(1.0 + m_abs) / delta_m_new
    y_m = tl.minimum(y_m, 32767.0)                          # safety cap

    floor_y_m = tl.floor(y_m)
    frac_m    = y_m - floor_y_m

    p_star_m = (tl.exp(frac_m * delta_m_new) - 1.0) / z_m_new
    p_star_m = tl.minimum(tl.maximum(p_star_m, 0.0), 1.0)

    rand_m  = tl.rand(seed, offsets)
    q_m_mag = floor_y_m + tl.where(rand_m < p_star_m, 1.0, 0.0)
    q_m_mag = tl.minimum(tl.maximum(q_m_mag, 0.0), 32767.0)

    m_sign_new = tl.where(m_new > 0.0, 1.0,
                          tl.where(m_new < 0.0, -1.0, 0.0))
    tl.store(q_m_ptr + offsets,
             (m_sign_new * q_m_mag).to(tl.int16), mask=mask)

    # ================================================================
    # Quantise preconditioner with NEW grid
    # ================================================================
    w_new  = 1.0 / (tl.sqrt(v_new) + eps)
    w_clip = tl.minimum(tl.maximum(w_new, w_min_new), w_max)

    y_w = tl.where(delta_w_new > 0.0,
                   tl.log(w_clip / w_min_new) / delta_w_new,
                   0.0)
    y_w = tl.minimum(y_w, 65535.0)                           # safety cap

    floor_y_w = tl.floor(y_w)
    frac_w    = y_w - floor_y_w

    p_star_w = tl.where(
        delta_w_new > 0.0,
        (tl.exp(frac_w * delta_w_new) - 1.0) / z_w_new,
        0.0,
    )
    p_star_w = tl.minimum(tl.maximum(p_star_w, 0.0), 1.0)

    # Independent PRNG stream: shift counter by n_elements
    rand_w  = tl.rand(seed, offsets + n_elements)
    q_w_new = floor_y_w + tl.where(rand_w < p_star_w, 1.0, 0.0)
    q_w_new = tl.minimum(tl.maximum(q_w_new, 0.0), 65535.0)

    tl.store(q_w_ptr + offsets, q_w_new.to(tl.int16), mask=mask)


# =====================================================================
#  Python wrapper  (host pre-compute + two launches + reduction)
# =====================================================================

def luma_triton_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    Q_m: torch.Tensor,
    Q_w: torch.Tensor,
    S_m: float,
    S_v: float,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
) -> tuple[float, float]:
    """Two-kernel fused Triton step with matched encode-decode scales.

    Modifies *param*, *Q_m*, *Q_w* **in-place**.
    Returns ``(S_m_next, S_v_next)``.

    Zero temporary allocation: Kernel B re-reads ``grad`` and the OLD
    ``Q_m`` / ``Q_w`` from HBM (still resident), recomputes the decode +
    EMA, then quantises with the NEW grid.  The redundant ALU is hidden
    behind HBM latency on every NVIDIA architecture.
    """
    # ── contiguity guard ─────────────────────────────────────────────
    if not param.is_contiguous():
        raise ValueError("LUMA Triton kernel requires contiguous parameters")
    grad = grad.contiguous()
    if not Q_m.is_contiguous() or not Q_w.is_contiguous():
        raise ValueError("LUMA state tensors must be contiguous")

    n = param.numel()

    # ── host pre-compute: OLD grid (for decode in both kernels) ──────
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    eta_t = lr / bc1

    w_min   = 1.0 / (math.sqrt(S_v) + eps)
    delta_m = math.log1p(S_m) / K_M

    ratio   = 1.0 / (2.0 * eps * w_min)
    delta_w = math.log(ratio) / K_W if ratio > 1.0 else 0.0

    # ── kernel launch config (architecture-aware) ───────────────────
    kcfg = get_kernel_config(param.device)
    BLOCK_SIZE = kcfg["BLOCK_SIZE"]
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ── per-block reduction buffers (tiny: num_blocks × 4 B each) ───
    s_m_block = torch.empty(num_blocks, device=param.device, dtype=torch.float32)
    s_v_block = torch.empty(num_blocks, device=param.device, dtype=torch.float32)

    # ── Kernel A: decode + update + block maxima ─────────────────────
    _luma_decode_update_kernel[(num_blocks,)](
        param, grad, Q_m, Q_w,
        s_m_block, s_v_block,
        beta1, beta2, eps, lr, weight_decay,
        eta_t, bc2,
        w_min, delta_m, delta_w,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=kcfg["num_warps"],
        num_stages=kcfg["num_stages"],
    )

    # ── reduce block maxima → global scales ──────────────────────────
    S_m_next = max(s_m_block.max().item(), SCALE_FLOOR_M)
    S_v_next = max(s_v_block.max().item(), eps * eps)

    # ── host pre-compute: NEW grid (for quantise in Kernel B) ────────
    w_min_new   = 1.0 / (math.sqrt(S_v_next) + eps)
    delta_m_new = math.log1p(S_m_next) / K_M
    z_m_new     = math.expm1(delta_m_new)

    ratio_new   = 1.0 / (2.0 * eps * w_min_new)
    delta_w_new = math.log(ratio_new) / K_W if ratio_new > 1.0 else 0.0
    z_w_new     = math.expm1(delta_w_new) if delta_w_new > 0.0 else 1.0
    w_max       = 1.0 / (2.0 * eps)

    # deterministic seed unique to (step, parameter)
    seed = ((step * 0x9E3779B9) ^ (param.data_ptr() >> 4)) & 0x7FFFFFFF

    # ── Kernel B: recompute + quantize with NEW grid ─────────────────
    _luma_requantize_kernel[(num_blocks,)](
        grad, Q_m, Q_w,
        beta1, beta2, eps,
        # OLD grid (for re-decode)
        w_min, delta_m, delta_w,
        # NEW grid (for quantise)
        delta_m_new, z_m_new,
        w_min_new, delta_w_new, z_w_new, w_max,
        seed, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=kcfg["num_warps"],
        num_stages=kcfg["num_stages"],
    )

    return S_m_next, S_v_next
