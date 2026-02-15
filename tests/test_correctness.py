"""Numerical correctness tests for the LUMA building blocks."""

import math

import pytest
import torch

from luma_optimizer import LUMA
from luma_optimizer.functional import (
    K_M,
    K_W,
    _decode_momentum,
    _decode_preconditioner,
    _decode_uint16,
    _encode_uint16,
    _log_sr,
    _precompute,
)

# ── Triton + CUDA availability flag ─────────────────────────────────────────
_CUDA_AND_TRITON = torch.cuda.is_available()
try:
    import triton  # noqa: F401
except ImportError:
    _CUDA_AND_TRITON = False


# ── uint16 round-trip  (CPU-only, pure integer logic) ──────────────────────


class TestUint16:
    @pytest.mark.parametrize("val", [0, 1, 32767, 32768, 65535])
    def test_roundtrip(self, val):
        t = torch.tensor([val], dtype=torch.int32)
        assert _decode_uint16(_encode_uint16(t)).item() == float(val)

    def test_batch_roundtrip(self):
        vals = torch.arange(0, K_W + 1, step=1000, dtype=torch.int32)
        assert torch.equal(_decode_uint16(_encode_uint16(vals)), vals.float())


# ── LogSR unbiasedness  (parametrised over devices) ────────────────────────


class TestLogSR:
    """Statistical tests: mean decoded value ≈ true linear-domain value."""

    @staticmethod
    def _run_unbiased_test(
        true_val: float,
        delta: float,
        z: float,
        k_max: int,
        decode_fn,
        n_samples: int = 200_000,
        tol: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ):
        y_scalar = decode_fn["to_y"](true_val)
        y = torch.full((n_samples,), y_scalar, device=device)
        rand = torch.rand(n_samples, device=device)
        q = _log_sr(y, delta, z, rand, k_max)
        decoded = decode_fn["from_q"](q)
        mean = decoded.mean().item()
        rel_err = abs(mean - true_val) / (abs(true_val) + 1e-30)
        assert rel_err < tol, (
            f"Biased on {device}: true={true_val:.6g}, mean={mean:.6g}, "
            f"rel_err={rel_err:.4f}"
        )

    @pytest.mark.parametrize("val", [0.001, 0.01, 0.1, 0.5, 0.99])
    def test_momentum_unbiased(self, val, device):
        S_m = 1.0
        delta_m = math.log1p(S_m) / K_M
        z_m = math.expm1(delta_m)
        self._run_unbiased_test(
            true_val=val,
            delta=delta_m,
            z=z_m,
            k_max=K_M,
            decode_fn={
                "to_y": lambda v: math.log1p(v) / delta_m,
                "from_q": lambda q: torch.expm1(q * delta_m),
            },
            device=device,
        )

    @pytest.mark.parametrize("frac", [0.01, 0.25, 0.5, 0.75, 0.99])
    def test_preconditioner_unbiased(self, frac, device):
        S_v, eps = 1.0, 1e-8
        w_min = 1.0 / (math.sqrt(S_v) + eps)
        w_max = 1.0 / (2.0 * eps)
        delta_w = math.log(w_max / w_min) / K_W
        z_w = math.expm1(delta_w)
        w_test = w_min + frac * (w_max - w_min)

        self._run_unbiased_test(
            true_val=w_test,
            delta=delta_w,
            z=z_w,
            k_max=K_W,
            decode_fn={
                "to_y": lambda v: math.log(v / w_min) / delta_w,
                "from_q": lambda q: w_min * torch.exp(q * delta_w),
            },
            tol=0.02,
            device=device,
        )


# ── encode / decode consistency ────────────────────────────────────────────


class TestDecodeMomentum:
    def test_zero(self, device):
        q = torch.tensor([0], dtype=torch.int16, device=device)
        assert _decode_momentum(q, 1e-4).item() == 0.0

    def test_positive(self, device):
        delta_m = 1e-4
        q = torch.tensor([1, 100, K_M], dtype=torch.int16, device=device)
        m = _decode_momentum(q, delta_m)
        assert (m > 0).all()

    def test_negative(self, device):
        delta_m = 1e-4
        q = torch.tensor([-1, -100], dtype=torch.int16, device=device)
        m = _decode_momentum(q, delta_m)
        assert (m < 0).all()

    def test_symmetry(self, device):
        delta_m = 1e-4
        q_pos = torch.tensor([42], dtype=torch.int16, device=device)
        q_neg = torch.tensor([-42], dtype=torch.int16, device=device)
        assert _decode_momentum(q_pos, delta_m).item() == pytest.approx(
            -_decode_momentum(q_neg, delta_m).item()
        )

    def test_max_bin_equals_scale(self, device):
        """Bin K_M should decode to S_m."""
        S_m = 2.5
        delta_m = math.log1p(S_m) / K_M
        q = torch.tensor([K_M], dtype=torch.int16, device=device)
        decoded = _decode_momentum(q, delta_m).item()
        assert decoded == pytest.approx(S_m, rel=1e-5)


class TestDecodePreconditioner:
    def test_range(self, device):
        eps = 1e-8
        S_v = 1.0
        w_min, _, _, delta_w, _ = _precompute(1.0, S_v, eps)
        w_max = 1.0 / (2.0 * eps)

        q_lo = _encode_uint16(torch.tensor([0], dtype=torch.int32)).to(device)
        q_hi = _encode_uint16(torch.tensor([K_W], dtype=torch.int32)).to(device)
        assert _decode_preconditioner(q_lo, w_min, delta_w).item() == pytest.approx(
            w_min, rel=1e-5
        )
        assert _decode_preconditioner(q_hi, w_min, delta_w).item() == pytest.approx(
            w_max, rel=1e-3
        )


# ── tracking fidelity vs FP32 AdamW ───────────────────────────────────────


class TestTracking:
    def test_quadratic(self, device):
        """LUMA should closely track FP32 AdamW on a well-conditioned quadratic.

        300 steps on a diagonal quadratic with eigenvalues in [0.5, 1.5].
        Checks: strong convergence, trajectory closeness, parameter proximity.
        """
        d = 32
        a = torch.rand(d, device=device) + 0.5   # eigenvalues in [0.5, 1.5]
        b = torch.randn(d, device=device)

        x_adamw = torch.randn(d, device=device, requires_grad=True)
        x_luma = x_adamw.data.clone().requires_grad_(True)

        opt_adamw = torch.optim.AdamW(
            [x_adamw], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        opt_luma = LUMA(
            [x_luma], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        loss_fn = lambda x: 0.5 * (a * (x - b).square()).sum()
        init_loss = loss_fn(x_adamw).item()

        losses_adamw: list[float] = []
        losses_luma: list[float] = []
        for _ in range(300):
            for opt, x, losses in [
                (opt_adamw, x_adamw, losses_adamw),
                (opt_luma, x_luma, losses_luma),
            ]:
                opt.zero_grad()
                loss = loss_fn(x)
                losses.append(loss.item())
                loss.backward()
                opt.step()

        # ── Strong convergence (>95% loss reduction) ──────────────────
        assert losses_luma[-1] < init_loss * 0.05, (
            f"LUMA didn't converge on {device}: "
            f"init={init_loss:.4f}, final={losses_luma[-1]:.4f}"
        )

        # ── Final loss close to AdamW (within 2×) ────────────────────
        assert losses_luma[-1] < losses_adamw[-1] * 2.0 + 1e-7, (
            f"LUMA too far from AdamW on {device}: "
            f"LUMA={losses_luma[-1]:.6g} vs AdamW={losses_adamw[-1]:.6g}"
        )

        # ── Trajectories stay close at checkpoints ───────────────────
        for step in [49, 149, 299]:
            l_a, l_h = losses_adamw[step], losses_luma[step]
            rel = abs(l_h - l_a) / max(l_a, 1e-8)
            assert rel < 0.5, (
                f"Trajectories diverged at step {step + 1} on {device}: "
                f"LUMA={l_h:.6g}, AdamW={l_a:.6g}, rel_diff={rel:.2f}"
            )

        # ── Parameters close in L2 ───────────────────────────────────
        param_dist = (x_luma.data - x_adamw.data).norm().item()
        param_scale = x_adamw.data.norm().item()
        assert param_dist < 0.05 * param_scale + 1e-6, (
            f"Parameters diverged on {device}: "
            f"||delta||={param_dist:.6f}, ||x_adamw||={param_scale:.4f}"
        )


# ── Triton ↔ PyTorch fallback parity (CUDA only) ────────────────────────────


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="Requires CUDA + Triton")
class TestTritonFallbackParity:
    """Triton kernel and PyTorch fallback must produce near-identical results."""

    @staticmethod
    def _make_pair(d: int = 64):
        """Two LUMA optimisers on CUDA: one Triton, one PyTorch fallback."""
        x_tri = torch.randn(d, device="cuda", requires_grad=True)
        x_pt = x_tri.data.clone().requires_grad_(True)

        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        opt_tri = LUMA([x_tri], backend="triton", **kw)
        opt_pt = LUMA([x_pt], backend="pytorch", **kw)

        return x_tri, x_pt, opt_tri, opt_pt

    # ── 1. Init step uses the same code path → bit-exact ────────────

    def test_init_step_identical(self):
        """After step 1 (shared ``luma_init_step``), params are bit-exact."""
        d = 64
        x_tri, x_pt, opt_tri, opt_pt = self._make_pair(d)

        a = torch.rand(d, device="cuda") + 0.5
        b = torch.randn(d, device="cuda")
        loss_fn = lambda x: 0.5 * (a * (x - b).square()).sum()

        for opt, x in [(opt_tri, x_tri), (opt_pt, x_pt)]:
            opt.zero_grad()
            loss_fn(x).backward()
            opt.step()

        assert torch.equal(x_tri.data, x_pt.data), (
            "Params differ after init step (should be bit-identical)"
        )

    # ── 2. Step 2: same decode → same param update ──────────────────

    def test_second_step_param_close(self):
        """After 2 steps, params nearly identical (only PRNG in quant differs)."""
        d = 64
        x_tri, x_pt, opt_tri, opt_pt = self._make_pair(d)

        a = torch.rand(d, device="cuda") + 0.5
        b = torch.randn(d, device="cuda")
        loss_fn = lambda x: 0.5 * (a * (x - b).square()).sum()

        for _ in range(2):
            for opt, x in [(opt_tri, x_tri), (opt_pt, x_pt)]:
                opt.zero_grad()
                loss_fn(x).backward()
                opt.step()

        # After step 2 the decoded states are identical (same Q from step 1),
        # so the param update is deterministic up to expm1 vs exp-1 (~ulp).
        diff = (x_tri.data - x_pt.data).abs().max().item()
        scale = x_pt.data.abs().max().item()
        assert diff < 1e-5 * scale + 1e-8, (
            f"Params diverged after 2 steps: max_diff={diff:.2e}, scale={scale:.4f}"
        )

    # ── 3. 100 steps: trajectories & params stay within 0.5 % ──────

    def test_hundred_steps_trajectory(self):
        """100 steps on a quadratic: losses ≤ 0.5 %, param L2 ≤ 0.5 %."""
        d = 64
        x_tri, x_pt, opt_tri, opt_pt = self._make_pair(d)

        a = torch.rand(d, device="cuda") + 0.5
        b = torch.randn(d, device="cuda")
        loss_fn = lambda x: 0.5 * (a * (x - b).square()).sum()

        losses_tri: list[float] = []
        losses_pt: list[float] = []
        for _ in range(100):
            for opt, x, losses in [
                (opt_tri, x_tri, losses_tri),
                (opt_pt, x_pt, losses_pt),
            ]:
                opt.zero_grad()
                loss = loss_fn(x)
                losses.append(loss.item())
                loss.backward()
                opt.step()

        # ── loss trajectories ────────────────────────────────────────
        for step in [9, 49, 99]:
            l_t, l_p = losses_tri[step], losses_pt[step]
            rel = abs(l_t - l_p) / max(l_p, 1e-8)
            assert rel < 0.005, (
                f"Losses diverged at step {step + 1}: "
                f"triton={l_t:.6g}, pytorch={l_p:.6g}, rel={rel:.4%}"
            )

        # ── final param L2 ──────────────────────────────────────────
        param_dist = (x_tri.data - x_pt.data).norm().item()
        param_scale = x_pt.data.norm().item()
        assert param_dist < 0.005 * param_scale + 1e-6, (
            f"Params diverged: dist={param_dist:.6f}, scale={param_scale:.4f}"
        )
