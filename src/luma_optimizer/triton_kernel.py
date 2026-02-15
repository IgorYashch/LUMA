"""LUMA optimizer — Triton four-kernel implementation (CUDA only).

Fully asynchronous, CUDA-Graph-compatible pipeline with **zero CPU-GPU
synchronization**:

    Kernel P  (precompute OLD grid + step scalars on GPU)
          ↓
    Kernel A  (decode + EMA + param update + block-max tracking)
          ↓
    Kernel R  (reduce block maxima → compute NEW grid params)
          ↓
    Kernel B  (re-decode + re-EMA + quantize with NEW grid)

All four kernels are queued back-to-back.  No ``.item()`` calls.
Per-step scalars (``eta_t``, ``bc2``, OLD grid constants, step counter)
live entirely in GPU buffers, enabling ``torch.cuda.CUDAGraph`` capture.

GPU buffer layouts
------------------
``old_scalars`` [8] float32  (written by Kernel P, read by Kernel A/B):
    [0] w_min_old   [1] delta_m_old  [2] delta_w_old
    [3] eta_t       [4] bc2          [5] bc1
    [6] step        [7] (reserved)

``new_grid`` [8] float32  (written by Kernel R, read by Kernel B):
    [0] S_m_next   [1] S_v_next   [2] delta_m_new  [3] z_m_new
    [4] w_min_new  [5] delta_w_new [6] z_w_new      [7] w_max

Numerical notes
---------------
* ``expm1(x)`` and ``log1p(x)`` are approximated as ``exp(x) - 1`` and
  ``log(1 + x)`` respectively.  For the tiny grid steps Δ ≈ 1e-4 the
  relative error is < 1e-3, well within the stochastic rounding noise.
* The PRNG uses Triton's built-in Philox generator seeded from
  ``(step, param_id)`` — both deterministic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .config import K_M, K_W, SCALE_FLOOR_M, get_kernel_config


# =====================================================================
#  Kernel P — precompute per-step scalars on GPU
# =====================================================================
@triton.jit
def _luma_precompute_kernel(
    old_scalars_ptr,     # [8] float32 — read bc1/bc2/step, write all
    new_grid_ptr,        # [8] float32 — read S_m ([0]), S_v ([1])
    beta1, beta2, eps, lr,
    K_M_VAL: tl.constexpr,
    K_W_VAL: tl.constexpr,
):
    """Single-element kernel: update step counter, bias corrections, OLD grid.

    Reads S_m/S_v from ``new_grid`` (the NEW grid of the *previous* step)
    and computes the OLD grid constants needed by Kernel A and B.
    """
    # ── read previous-step state ──────────────────────────────────────
    bc1_old = tl.load(old_scalars_ptr + 5)
    bc2_old = tl.load(old_scalars_ptr + 4)
    step_old = tl.load(old_scalars_ptr + 6)

    S_m = tl.load(new_grid_ptr + 0)
    S_v = tl.load(new_grid_ptr + 1)

    # ── increment step ────────────────────────────────────────────────
    step_new = step_old + 1.0

    # ── bias correction recurrence: bc(t+1) = 1 - β·(1 - bc(t)) ─────
    bc1_new = 1.0 - beta1 + beta1 * bc1_old
    bc2_new = 1.0 - beta2 + beta2 * bc2_old
    eta_t = lr / bc1_new

    # ── OLD quantisation grid from S_m, S_v ───────────────────────────
    w_min = 1.0 / (tl.sqrt(S_v) + eps)
    delta_m = tl.log(1.0 + S_m) / K_M_VAL
    two_eps = 2.0 * eps
    ratio = 1.0 / (two_eps * w_min)
    delta_w = tl.log(tl.maximum(ratio, 1.0)) / K_W_VAL

    # ── store ─────────────────────────────────────────────────────────
    tl.store(old_scalars_ptr + 0, w_min)
    tl.store(old_scalars_ptr + 1, delta_m)
    tl.store(old_scalars_ptr + 2, delta_w)
    tl.store(old_scalars_ptr + 3, eta_t)
    tl.store(old_scalars_ptr + 4, bc2_new)
    tl.store(old_scalars_ptr + 5, bc1_new)
    tl.store(old_scalars_ptr + 6, step_new)


# =====================================================================
#  Kernel A — decode, EMA, param update, block-level scale maxima
# =====================================================================
@triton.jit
def _luma_decode_update_kernel(
    param_ptr, grad_ptr,
    q_m_ptr, q_w_ptr,
    s_m_block_ptr, s_v_block_ptr,
    old_scalars_ptr,      # per-step scalars from Kernel P
    beta1, beta2, eps, lr, weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ── per-step scalars from GPU buffer ──────────────────────────────
    w_min   = tl.load(old_scalars_ptr + 0)
    delta_m = tl.load(old_scalars_ptr + 1)
    delta_w = tl.load(old_scalars_ptr + 2)
    eta_t   = tl.load(old_scalars_ptr + 3)
    bc2     = tl.load(old_scalars_ptr + 4)

    # ── Load from HBM ────────────────────────────────────────────────
    param   = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad    = tl.load(grad_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
    q_m_i32 = tl.load(q_m_ptr  + offsets, mask=mask, other=0).to(tl.int32)
    q_w_i32 = tl.load(q_w_ptr  + offsets, mask=mask, other=0).to(tl.int32)

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
    tl.store(param_ptr + offsets, param, mask=mask)

    # ── 4. Block-level scale tracking ────────────────────────────────
    abs_m_safe = tl.where(mask, tl.abs(m_new), 0.0)
    v_new_safe = tl.where(mask, v_new, 0.0)
    tl.store(s_m_block_ptr + pid, tl.max(abs_m_safe, axis=0))
    tl.store(s_v_block_ptr + pid, tl.max(v_new_safe, axis=0))


# =====================================================================
#  Kernel R — reduce block maxima → NEW grid parameters
# =====================================================================
@triton.jit
def _luma_reduce_grid_kernel(
    s_m_block_ptr, s_v_block_ptr,
    new_grid_ptr,
    num_blocks,
    scale_floor_m, eps,
    K_M_VAL: tl.constexpr,
    K_W_VAL: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    """Single-block kernel: reduces per-block maxima → new_grid[0:8]."""
    max_m_vec = tl.full([REDUCE_BLOCK], value=0.0, dtype=tl.float32)
    max_v_vec = tl.full([REDUCE_BLOCK], value=0.0, dtype=tl.float32)

    for start in range(0, num_blocks, REDUCE_BLOCK):
        offsets = start + tl.arange(0, REDUCE_BLOCK)
        mask = offsets < num_blocks
        max_m_vec = tl.maximum(max_m_vec,
                               tl.load(s_m_block_ptr + offsets, mask=mask, other=0.0))
        max_v_vec = tl.maximum(max_v_vec,
                               tl.load(s_v_block_ptr + offsets, mask=mask, other=0.0))

    S_m = tl.maximum(tl.max(max_m_vec, axis=0), scale_floor_m)
    S_v = tl.maximum(tl.max(max_v_vec, axis=0), eps * eps)

    w_min   = 1.0 / (tl.sqrt(S_v) + eps)
    delta_m = tl.log(1.0 + S_m) / K_M_VAL
    z_m     = tl.exp(delta_m) - 1.0
    two_eps = 2.0 * eps
    ratio   = 1.0 / (two_eps * w_min)
    delta_w = tl.log(tl.maximum(ratio, 1.0)) / K_W_VAL
    z_w     = tl.where(delta_w > 0.0, tl.exp(delta_w) - 1.0, 1.0)
    w_max   = 1.0 / two_eps

    tl.store(new_grid_ptr + 0, S_m)
    tl.store(new_grid_ptr + 1, S_v)
    tl.store(new_grid_ptr + 2, delta_m)
    tl.store(new_grid_ptr + 3, z_m)
    tl.store(new_grid_ptr + 4, w_min)
    tl.store(new_grid_ptr + 5, delta_w)
    tl.store(new_grid_ptr + 6, z_w)
    tl.store(new_grid_ptr + 7, w_max)


# =====================================================================
#  Kernel B — re-decode, re-EMA, quantize with NEW matched grid
# =====================================================================
@triton.jit
def _luma_requantize_kernel(
    grad_ptr, q_m_ptr, q_w_ptr,
    old_scalars_ptr,      # OLD grid [0:3] + step [6]
    new_grid_ptr,         # NEW grid [2:8]
    beta1, beta2, eps,
    param_id,             # constant scalar — for PRNG seed
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ── OLD grid from Kernel P ────────────────────────────────────────
    w_min_old   = tl.load(old_scalars_ptr + 0)
    delta_m_old = tl.load(old_scalars_ptr + 1)
    delta_w_old = tl.load(old_scalars_ptr + 2)

    # ── NEW grid from Kernel R ────────────────────────────────────────
    delta_m_new = tl.load(new_grid_ptr + 2)
    z_m_new     = tl.load(new_grid_ptr + 3)
    w_min_new   = tl.load(new_grid_ptr + 4)
    delta_w_new = tl.load(new_grid_ptr + 5)
    z_w_new     = tl.load(new_grid_ptr + 6)
    w_max       = tl.load(new_grid_ptr + 7)

    # ── deterministic PRNG seed from (step, param_id) ─────────────────
    step_i = tl.load(old_scalars_ptr + 6).to(tl.int32)
    seed = ((step_i * 1103515245 + 12345) ^ (param_id * 22695477)) & 0x7FFFFFFF

    # ── Load grad + OLD quantised states ─────────────────────────────
    grad    = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q_m_i32 = tl.load(q_m_ptr + offsets, mask=mask, other=0).to(tl.int32)
    q_w_i32 = tl.load(q_w_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # ── Re-decode with OLD grid ──────────────────────────────────────
    q_m_abs = tl.abs(q_m_i32).to(tl.float32)
    m_sign  = tl.where(q_m_i32 > 0, 1.0,
                       tl.where(q_m_i32 < 0, -1.0, 0.0))
    m = m_sign * (tl.exp(q_m_abs * delta_m_old) - 1.0)

    q_w_unsigned = (q_w_i32 & 0xFFFF).to(tl.float32)
    w     = w_min_old * tl.exp(q_w_unsigned * delta_w_old)
    inv_w = 1.0 / w
    v     = (inv_w - eps) * (inv_w - eps)

    # ── Re-compute EMA (free ALU hidden behind HBM latency) ──────────
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = tl.maximum(beta2 * v + (1.0 - beta2) * grad * grad,
                       eps * eps)

    # ── Quantise momentum with NEW grid ──────────────────────────────
    m_abs = tl.abs(m_new)
    y_m = tl.log(1.0 + m_abs) / delta_m_new
    y_m = tl.minimum(y_m, 32767.0)

    floor_y_m = tl.floor(y_m)
    frac_m    = y_m - floor_y_m
    p_star_m  = (tl.exp(frac_m * delta_m_new) - 1.0) / z_m_new
    p_star_m  = tl.minimum(tl.maximum(p_star_m, 0.0), 1.0)

    rand_m  = tl.rand(seed, offsets)
    q_m_mag = floor_y_m + tl.where(rand_m < p_star_m, 1.0, 0.0)
    q_m_mag = tl.minimum(tl.maximum(q_m_mag, 0.0), 32767.0)

    m_sign_new = tl.where(m_new > 0.0, 1.0,
                          tl.where(m_new < 0.0, -1.0, 0.0))
    tl.store(q_m_ptr + offsets,
             (m_sign_new * q_m_mag).to(tl.int16), mask=mask)

    # ── Quantise preconditioner with NEW grid ────────────────────────
    w_new  = 1.0 / (tl.sqrt(v_new) + eps)
    w_clip = tl.minimum(tl.maximum(w_new, w_min_new), w_max)

    y_w = tl.where(delta_w_new > 0.0,
                   tl.log(w_clip / w_min_new) / delta_w_new, 0.0)
    y_w = tl.minimum(y_w, 65535.0)

    floor_y_w = tl.floor(y_w)
    frac_w    = y_w - floor_y_w
    p_star_w  = tl.where(delta_w_new > 0.0,
                         (tl.exp(frac_w * delta_w_new) - 1.0) / z_w_new, 0.0)
    p_star_w  = tl.minimum(tl.maximum(p_star_w, 0.0), 1.0)

    rand_w  = tl.rand(seed, offsets + n_elements)
    q_w_new = floor_y_w + tl.where(rand_w < p_star_w, 1.0, 0.0)
    q_w_new = tl.minimum(tl.maximum(q_w_new, 0.0), 65535.0)

    tl.store(q_w_ptr + offsets, q_w_new.to(tl.int16), mask=mask)


# =====================================================================
#  Python wrapper — four fully-async launches, zero CPU-GPU sync
# =====================================================================

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
) -> None:
    """Four-kernel fused Triton step.  CUDA-Graph-compatible.

    Modifies *param*, *Q_m*, *Q_w*, *old_scalars*, *new_grid* **in-place**.
    ``S_m_next`` and ``S_v_next`` are available in ``new_grid[0:2]`` after
    the kernels complete (no CPU readback needed).

    All four kernels (P → A → R → B) are queued without any CPU-GPU
    synchronisation, making the call compatible with
    ``torch.cuda.CUDAGraph`` capture.
    """
    if not param.is_contiguous():
        raise ValueError("LUMA Triton kernel requires contiguous parameters")
    grad = grad.contiguous()

    n = param.numel()
    kcfg = get_kernel_config(param.device)
    BLOCK_SIZE = kcfg["BLOCK_SIZE"]
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ── Kernel P: precompute per-step scalars (async) ─────────────────
    _luma_precompute_kernel[(1,)](
        old_scalars, new_grid,
        beta1, beta2, eps, lr,
        K_M_VAL=K_M,
        K_W_VAL=K_W,
    )

    # ── Kernel A: decode + update + block maxima (async) ──────────────
    _luma_decode_update_kernel[(num_blocks,)](
        param, grad, Q_m, Q_w,
        s_m_block, s_v_block,
        old_scalars,
        beta1, beta2, eps, lr, weight_decay,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=kcfg["num_warps"],
        num_stages=kcfg["num_stages"],
    )

    # ── Kernel R: reduce block maxima → new grid params (async) ───────
    _luma_reduce_grid_kernel[(1,)](
        s_m_block, s_v_block,
        new_grid,
        num_blocks,
        SCALE_FLOOR_M, eps,
        K_M_VAL=K_M,
        K_W_VAL=K_W,
        REDUCE_BLOCK=1024,
    )

    # ── Kernel B: recompute + quantize with NEW grid (async) ──────────
    _luma_requantize_kernel[(num_blocks,)](
        grad, Q_m, Q_w,
        old_scalars, new_grid,
        beta1, beta2, eps,
        param_id, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=kcfg["num_warps"],
        num_stages=kcfg["num_stages"],
    )
