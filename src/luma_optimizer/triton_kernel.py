"""LUMA optimizer — Triton kernel implementation (CUDA only).

Fully asynchronous, CUDA-Graph-compatible pipelines with **zero CPU-GPU
synchronization**.

Steady-state step (t >= 2)::

    Kernel P  (precompute OLD grid + bias-correction scalars on GPU)
          ↓
    Kernel A  (decode + EMA + param update + block-max tracking)
          ↓
    Kernel R  (reduce block maxima → compute NEW grid params)
          ↓
    Kernel B  (re-decode + re-EMA + quantize with NEW grid)

Init step (t = 1)::

    Kernel I_A  (EMA from grad + param update + block-max tracking)
          ↓
    Kernel R    (reduce block maxima → compute NEW grid params)
          ↓
    Kernel I_B  (recompute m/v from grad + quantize with NEW grid)

GPU buffer layouts
------------------
``old_scalars`` [8] float32  (written by Kernel P, read by Kernel A/B):
    [0] w_min_old   [1] delta_m_old  [2] delta_w_old
    [3] eta_t       [4] bc2          [5] bc1
    [6] (reserved)  [7] (reserved)

``_step_tensor`` [1] int64  (incremented by Kernel P, read by Kernel B):
    [0] step counter (limit 2⁶³)

``new_grid`` [8] float32  (written by Kernel R, read by Kernel B):
    [0] S_m_next   [1] S_v_next   [2] delta_m_new  [3] z_m_new
    [4] w_min_new  [5] delta_w_new [6] z_w_new      [7] w_max

Numerical notes
---------------
* ``log1p(x)`` and ``expm1(x)`` use inline 2-term Taylor expansions for
  ``|x| < 1e-5`` to avoid catastrophic cancellation in float32.
* The PRNG uses Triton's built-in Philox generator seeded from
  ``(step, param_id, base_seed)``.
* The ``uint16`` preconditioner bins are stored via an ``int32``
  intermediate cast to guarantee two's-complement bit-wrapping
  (PTX ``cvt.rzi.s16.f32`` saturates floats > 32767).
* The step counter lives in a dedicated ``int64`` GPU tensor
  (``_step_tensor``), incremented by Kernel P on each step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .config import K_M, K_W, SCALE_FLOOR_M, get_kernel_config


# =====================================================================
#  Numerically safe helpers
# =====================================================================

@triton.jit
def _safe_log1p(x):
    """``log(1 + x)``, numerically stable for small *x*."""
    return tl.where(x * x < 1e-10, x - 0.5 * x * x, tl.log(1.0 + x))


@triton.jit
def _safe_expm1(x):
    """``exp(x) - 1``, numerically stable for small *x*."""
    return tl.where(x * x < 1e-10, x + 0.5 * x * x, tl.exp(x) - 1.0)


# =====================================================================
#  Shared quantization helper (used by Kernel B and Kernel I_B)
# =====================================================================

@triton.jit
def _quantize_mv(
    m_new, v_new,
    q_m_ptr, q_w_ptr,
    delta_m_new, z_m_new,
    w_min_new, delta_w_new, z_w_new, w_max,
    seed_m, seed_w,
    eps, offsets, mask,
):
    """Quantize m_new/v_new to int16 Q_m/Q_w with log-space stochastic rounding.

    Writes results in-place to q_m_ptr and q_w_ptr.
    """
    # ── Momentum ─────────────────────────────────────────────────────
    m_abs = tl.abs(m_new)
    y_m = _safe_log1p(m_abs) / delta_m_new
    y_m = tl.minimum(y_m, 32767.0)

    floor_y_m = tl.floor(y_m)
    frac_m = y_m - floor_y_m
    p_star_m = _safe_expm1(frac_m * delta_m_new) / z_m_new
    p_star_m = tl.minimum(tl.maximum(p_star_m, 0.0), 1.0)

    rand_m = tl.rand(seed_m, offsets)
    q_m_mag = floor_y_m + tl.where(rand_m < p_star_m, 1.0, 0.0)
    q_m_mag = tl.minimum(tl.maximum(q_m_mag, 0.0), 32767.0)

    m_sign = tl.where(m_new > 0.0, 1.0, tl.where(m_new < 0.0, -1.0, 0.0))
    tl.store(q_m_ptr + offsets, (m_sign * q_m_mag).to(tl.int16), mask=mask)

    # ── Preconditioner ───────────────────────────────────────────────
    w_new = 1.0 / (tl.sqrt(tl.maximum(v_new, 0.0)) + eps)
    w_clip = tl.minimum(tl.maximum(w_new, w_min_new), w_max)

    y_w = tl.where(delta_w_new > 0.0,
                   tl.log(w_clip / w_min_new) / delta_w_new, 0.0)
    y_w = tl.minimum(y_w, 65535.0)

    floor_y_w = tl.floor(y_w)
    frac_w = y_w - floor_y_w
    p_star_w = tl.where(delta_w_new > 0.0,
                        _safe_expm1(frac_w * delta_w_new) / z_w_new, 0.0)
    p_star_w = tl.minimum(tl.maximum(p_star_w, 0.0), 1.0)

    rand_w = tl.rand(seed_w, offsets)
    q_w_new = floor_y_w + tl.where(rand_w < p_star_w, 1.0, 0.0)
    q_w_new = tl.minimum(tl.maximum(q_w_new, 0.0), 65535.0)

    tl.store(q_w_ptr + offsets, q_w_new.to(tl.int32).to(tl.int16), mask=mask)


@triton.jit
def _decode_states(q_m_ptr, q_w_ptr, delta_m, delta_w, w_min, eps, offsets, mask):
    """Decode int16 quantised states → FP32 (m, v)."""
    q_m_i32 = tl.load(q_m_ptr + offsets, mask=mask, other=0).to(tl.int32)
    q_w_i32 = tl.load(q_w_ptr + offsets, mask=mask, other=0).to(tl.int32)

    q_m_abs = tl.abs(q_m_i32).to(tl.float32)
    m_sign = tl.where(q_m_i32 > 0, 1.0, tl.where(q_m_i32 < 0, -1.0, 0.0))
    m = m_sign * _safe_expm1(q_m_abs * delta_m)

    q_w_unsigned = (q_w_i32 & 0xFFFF).to(tl.float32)
    w = w_min * tl.exp(q_w_unsigned * delta_w)
    v_delta = tl.maximum(1.0 / w - eps, 0.0)
    v = v_delta * v_delta

    return m, v


# =====================================================================
#  Kernel P — precompute per-step scalars on GPU
# =====================================================================
@triton.jit
def _luma_precompute_kernel(
    old_scalars_ptr, new_grid_ptr, step_ptr,
    beta1, beta2, eps, lr,
    K_M_VAL: tl.constexpr,
    K_W_VAL: tl.constexpr,
):
    """Single-element kernel: increment step, update bias corrections, compute OLD grid."""
    step_old = tl.load(step_ptr)
    tl.store(step_ptr, step_old + 1)

    bc1_old = tl.load(old_scalars_ptr + 5)
    bc2_old = tl.load(old_scalars_ptr + 4)
    S_m = tl.load(new_grid_ptr + 0)
    S_v = tl.load(new_grid_ptr + 1)

    # Bias correction recurrence: equivalent to 1 - β^t in exact arithmetic.
    # Float32 accumulates ~1 ULP/step — negligible in practice.
    bc1_new = 1.0 - beta1 + beta1 * bc1_old
    bc2_new = 1.0 - beta2 + beta2 * bc2_old
    eta_t = lr / bc1_new

    w_min = 1.0 / (tl.sqrt(S_v) + eps)
    delta_m = _safe_log1p(S_m) / K_M_VAL
    ratio = 1.0 / (eps * w_min)
    delta_w = tl.log(tl.maximum(ratio, 1.0)) / K_W_VAL

    tl.store(old_scalars_ptr + 0, w_min)
    tl.store(old_scalars_ptr + 1, delta_m)
    tl.store(old_scalars_ptr + 2, delta_w)
    tl.store(old_scalars_ptr + 3, eta_t)
    tl.store(old_scalars_ptr + 4, bc2_new)
    tl.store(old_scalars_ptr + 5, bc1_new)


# =====================================================================
#  Kernel A — decode, EMA, param update, block-level scale maxima
# =====================================================================
@triton.jit
def _luma_decode_update_kernel(
    param_ptr, grad_ptr, q_m_ptr, q_w_ptr,
    s_m_block_ptr, s_v_block_ptr, old_scalars_ptr,
    beta1, beta2, eps, lr, weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    w_min   = tl.load(old_scalars_ptr + 0)
    delta_m = tl.load(old_scalars_ptr + 1)
    delta_w = tl.load(old_scalars_ptr + 2)
    eta_t   = tl.load(old_scalars_ptr + 3)
    bc2     = tl.load(old_scalars_ptr + 4)

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad  = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m, v = _decode_states(q_m_ptr, q_w_ptr, delta_m, delta_w, w_min, eps, offsets, mask)

    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad * grad

    denom = tl.sqrt(v_new / bc2) + eps
    param = param * (1.0 - weight_decay * lr) - eta_t * m_new / denom
    tl.store(param_ptr + offsets, param, mask=mask)

    abs_m_safe = tl.where(mask, tl.abs(m_new), 0.0)
    v_new_safe = tl.where(mask, v_new, 0.0)
    tl.store(s_m_block_ptr + pid, tl.max(abs_m_safe, axis=0))
    tl.store(s_v_block_ptr + pid, tl.max(v_new_safe, axis=0))


# =====================================================================
#  Kernel R — reduce block maxima → NEW grid parameters
# =====================================================================
@triton.jit
def _luma_reduce_grid_kernel(
    s_m_block_ptr, s_v_block_ptr, new_grid_ptr,
    num_blocks, scale_floor_m, eps,
    K_M_VAL: tl.constexpr,
    K_W_VAL: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    """Single-block kernel: reduces per-block maxima → new_grid[0:8]."""
    max_m_vec = tl.full([REDUCE_BLOCK], value=0.0, dtype=tl.float32)
    max_v_vec = tl.full([REDUCE_BLOCK], value=0.0, dtype=tl.float32)

    for start in range(0, num_blocks, REDUCE_BLOCK):
        offsets = start + tl.arange(0, REDUCE_BLOCK)
        rmask = offsets < num_blocks
        max_m_vec = tl.maximum(max_m_vec,
                               tl.load(s_m_block_ptr + offsets, mask=rmask, other=0.0))
        max_v_vec = tl.maximum(max_v_vec,
                               tl.load(s_v_block_ptr + offsets, mask=rmask, other=0.0))

    S_m = tl.maximum(tl.max(max_m_vec, axis=0), scale_floor_m)
    S_v = tl.maximum(tl.max(max_v_vec, axis=0), 0.0)
    w_min   = 1.0 / (tl.sqrt(S_v) + eps)
    delta_m = _safe_log1p(S_m) / K_M_VAL
    z_m     = _safe_expm1(delta_m)
    ratio   = 1.0 / (eps * w_min)
    delta_w = tl.log(tl.maximum(ratio, 1.0)) / K_W_VAL
    z_w     = tl.where(delta_w > 0.0, _safe_expm1(delta_w), 1.0)

    tl.store(new_grid_ptr + 0, S_m)
    tl.store(new_grid_ptr + 1, S_v)
    tl.store(new_grid_ptr + 2, delta_m)
    tl.store(new_grid_ptr + 3, z_m)
    tl.store(new_grid_ptr + 4, w_min)
    tl.store(new_grid_ptr + 5, delta_w)
    tl.store(new_grid_ptr + 6, z_w)
    tl.store(new_grid_ptr + 7, 1.0 / eps)  # w_max


# =====================================================================
#  Kernel B — re-decode, re-EMA, quantize with NEW matched grid
# =====================================================================
@triton.jit
def _luma_requantize_kernel(
    grad_ptr, q_m_ptr, q_w_ptr,
    old_scalars_ptr, new_grid_ptr, step_ptr,
    beta1, beta2, eps,
    param_id, base_seed,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # OLD grid
    w_min_old   = tl.load(old_scalars_ptr + 0)
    delta_m_old = tl.load(old_scalars_ptr + 1)
    delta_w_old = tl.load(old_scalars_ptr + 2)

    # NEW grid
    delta_m_new = tl.load(new_grid_ptr + 2)
    z_m_new     = tl.load(new_grid_ptr + 3)
    w_min_new   = tl.load(new_grid_ptr + 4)
    delta_w_new = tl.load(new_grid_ptr + 5)
    z_w_new     = tl.load(new_grid_ptr + 6)
    w_max       = tl.load(new_grid_ptr + 7)

    # PRNG seeds
    step_64 = tl.load(step_ptr).to(tl.int64)
    pid_64 = param_id.to(tl.int64)
    bs_64 = base_seed.to(tl.int64)
    seed_m = ((step_64 << 32) | (pid_64 * 2)) ^ bs_64
    seed_w = ((step_64 << 32) | (pid_64 * 2 + 1)) ^ bs_64

    # Re-decode with OLD grid, re-compute EMA
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m, v = _decode_states(q_m_ptr, q_w_ptr, delta_m_old, delta_w_old, w_min_old, eps, offsets, mask)
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad * grad

    _quantize_mv(m_new, v_new, q_m_ptr, q_w_ptr,
                 delta_m_new, z_m_new, w_min_new, delta_w_new, z_w_new, w_max,
                 seed_m, seed_w, eps, offsets, mask)


# =====================================================================
#  Init Kernel I_A — compute m/v from grad, param update, block maxima
# =====================================================================
@triton.jit
def _luma_init_update_kernel(
    param_ptr, grad_ptr, s_m_block_ptr, s_v_block_ptr,
    beta1, beta2, eps, lr, weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Step-1: compute m/v from gradient, update param, track block maxima."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad  = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    m_new = (1.0 - beta1) * grad
    v_new = (1.0 - beta2) * grad * grad

    bc1 = 1.0 - beta1
    bc2 = 1.0 - beta2
    denom = tl.sqrt(v_new / bc2) + eps
    param = param * (1.0 - weight_decay * lr) - (lr / bc1) * m_new / denom
    tl.store(param_ptr + offsets, param, mask=mask)

    abs_m_safe = tl.where(mask, tl.abs(m_new), 0.0)
    v_new_safe = tl.where(mask, v_new, 0.0)
    tl.store(s_m_block_ptr + pid, tl.max(abs_m_safe, axis=0))
    tl.store(s_v_block_ptr + pid, tl.max(v_new_safe, axis=0))


# =====================================================================
#  Init Kernel I_B — recompute m/v from grad, quantize with NEW grid
# =====================================================================
@triton.jit
def _luma_init_quantize_kernel(
    grad_ptr, q_m_ptr, q_w_ptr, new_grid_ptr,
    beta1, beta2, eps,
    param_id, base_seed,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Step-1: recompute m/v from grad, quantize with NEW grid."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # NEW grid
    delta_m_new = tl.load(new_grid_ptr + 2)
    z_m_new     = tl.load(new_grid_ptr + 3)
    w_min_new   = tl.load(new_grid_ptr + 4)
    delta_w_new = tl.load(new_grid_ptr + 5)
    z_w_new     = tl.load(new_grid_ptr + 6)
    w_max       = tl.load(new_grid_ptr + 7)

    # PRNG seeds (step = 1 for init)
    pid_64 = param_id.to(tl.int64)
    bs_64 = base_seed.to(tl.int64)
    step_1 = tl.full([], 1, dtype=tl.int64)
    seed_m = ((step_1 << 32) | (pid_64 * 2)) ^ bs_64
    seed_w = ((step_1 << 32) | (pid_64 * 2 + 1)) ^ bs_64

    # Recompute m/v from grad
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m_new = (1.0 - beta1) * grad
    v_new = (1.0 - beta2) * grad * grad

    _quantize_mv(m_new, v_new, q_m_ptr, q_w_ptr,
                 delta_m_new, z_m_new, w_min_new, delta_w_new, z_w_new, w_max,
                 seed_m, seed_w, eps, offsets, mask)


# =====================================================================
#  Python wrappers
# =====================================================================

def _prepare_launch(param, grad):
    """Validate inputs and return (n, kernel_config, num_blocks)."""
    if param.dtype != torch.float32:
        raise ValueError(
            f"LUMA Triton kernel requires float32 parameters (got {param.dtype})"
        )
    if not param.is_contiguous():
        raise ValueError("LUMA Triton kernel requires contiguous parameters")
    if not grad.is_contiguous():
        raise ValueError("LUMA Triton kernel requires contiguous gradients")
    n = param.numel()
    kcfg = get_kernel_config(param.device)
    bs = kcfg["BLOCK_SIZE"]
    num_blocks = (n + bs - 1) // bs
    return grad, n, kcfg, num_blocks


def luma_triton_init_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    Q_m: torch.Tensor,
    Q_w: torch.Tensor,
    new_grid: torch.Tensor,
    s_m_block: torch.Tensor,
    s_v_block: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    param_id: int,
    base_seed: int,
) -> None:
    """Three-kernel fused Triton init step.  Zero CPU-GPU sync.

    Pipeline: I_A (update + block-max) → R (reduce) → I_B (quantize).
    """
    grad, n, kcfg, num_blocks = _prepare_launch(param, grad)
    BS = kcfg["BLOCK_SIZE"]
    kw = dict(num_warps=kcfg["num_warps"], num_stages=kcfg["num_stages"])

    _luma_init_update_kernel[(num_blocks,)](
        param, grad, s_m_block, s_v_block,
        beta1, beta2, eps, lr, weight_decay, n,
        BLOCK_SIZE=BS, **kw,
    )
    _luma_reduce_grid_kernel[(1,)](
        s_m_block, s_v_block, new_grid, num_blocks,
        SCALE_FLOOR_M, eps, K_M_VAL=K_M, K_W_VAL=K_W, REDUCE_BLOCK=1024,
    )
    _luma_init_quantize_kernel[(num_blocks,)](
        grad, Q_m, Q_w, new_grid,
        beta1, beta2, eps, param_id, base_seed, n,
        BLOCK_SIZE=BS, **kw,
    )


def luma_triton_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    Q_m: torch.Tensor,
    Q_w: torch.Tensor,
    old_scalars: torch.Tensor,
    new_grid: torch.Tensor,
    s_m_block: torch.Tensor,
    s_v_block: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    param_id: int,
    step_tensor: torch.Tensor,
    base_seed: int,
) -> None:
    """Four-kernel fused Triton step.  Zero CPU-GPU sync.

    Pipeline: P (precompute) → A (update + block-max) → R (reduce) → B (quantize).
    """
    grad, n, kcfg, num_blocks = _prepare_launch(param, grad)
    BS = kcfg["BLOCK_SIZE"]
    kw = dict(num_warps=kcfg["num_warps"], num_stages=kcfg["num_stages"])

    _luma_precompute_kernel[(1,)](
        old_scalars, new_grid, step_tensor,
        beta1, beta2, eps, lr, K_M_VAL=K_M, K_W_VAL=K_W,
    )
    _luma_decode_update_kernel[(num_blocks,)](
        param, grad, Q_m, Q_w, s_m_block, s_v_block, old_scalars,
        beta1, beta2, eps, lr, weight_decay, n,
        BLOCK_SIZE=BS, **kw,
    )
    _luma_reduce_grid_kernel[(1,)](
        s_m_block, s_v_block, new_grid, num_blocks,
        SCALE_FLOOR_M, eps, K_M_VAL=K_M, K_W_VAL=K_W, REDUCE_BLOCK=1024,
    )
    _luma_requantize_kernel[(num_blocks,)](
        grad, Q_m, Q_w, old_scalars, new_grid, step_tensor,
        beta1, beta2, eps, param_id, base_seed, n,
        BLOCK_SIZE=BS, **kw,
    )
