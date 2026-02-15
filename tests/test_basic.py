"""Basic API and smoke tests for LUMA optimizer."""

import pytest
import torch
import torch.nn as nn

from luma_optimizer import LUMA


# ── creation  (device-independent) ──────────────────────────────────────────


class TestCreation:
    def test_default_params(self):
        model = nn.Linear(8, 4)
        opt = LUMA(model.parameters())
        assert opt is not None

    def test_custom_params(self):
        model = nn.Linear(8, 4)
        opt = LUMA(
            model.parameters(),
            lr=5e-4,
            betas=(0.95, 0.98),
            eps=1e-6,
            weight_decay=0.1,
        )
        assert opt.defaults["lr"] == 5e-4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": -1},
            {"eps": -1},
            {"eps": 0},
            {"betas": (1.0, 0.999)},
            {"betas": (0.9, -0.1)},
            {"weight_decay": -0.01},
        ],
    )
    def test_invalid_hyperparams(self, kwargs):
        model = nn.Linear(4, 2)
        with pytest.raises(ValueError):
            LUMA(model.parameters(), **kwargs)


# ── stepping  (parametrised over devices) ───────────────────────────────────


class TestStepping:
    def test_single_step(self, device):
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters())
        loss = model(torch.randn(2, 8, device=device)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_multiple_steps(self, device):
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), lr=1e-3)
        for _ in range(10):
            loss = model(torch.randn(2, 8, device=device)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_no_grad_param_skipped(self, device):
        """Parameters without gradients should be silently skipped."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters())
        opt.step()  # no backward yet — should not raise

    def test_sparse_grad_rejected(self, device):
        emb = nn.Embedding(10, 4, sparse=True).to(device)
        opt = LUMA(emb.parameters())
        emb(torch.tensor([1, 2], device=device)).sum().backward()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_closure(self, device):
        model = nn.Linear(4, 2).to(device)
        opt = LUMA(model.parameters())
        x = torch.randn(2, 4, device=device)
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()


# ── state shapes & dtypes ──────────────────────────────────────────────────


class TestStateDtypes:
    def test_quantised_dtypes(self, device):
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(1, 16, device=device)).sum().backward()
        opt.step()
        for p in model.parameters():
            s = opt.state[p]
            assert s["Q_m"].dtype == torch.int16
            assert s["Q_w"].dtype == torch.int16

    def test_state_shapes_match_param(self, device):
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(1, 16, device=device)).sum().backward()
        opt.step()
        for p in model.parameters():
            s = opt.state[p]
            assert s["Q_m"].shape == p.shape
            assert s["Q_w"].shape == p.shape

    def test_state_on_same_device(self, device):
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(1, 16, device=device)).sum().backward()
        opt.step()
        for p in model.parameters():
            s = opt.state[p]
            assert s["Q_m"].device.type == device.type
            assert s["Q_w"].device.type == device.type


# ── memory footprint ───────────────────────────────────────────────────────


class TestMemory:
    def test_4_bytes_per_param(self, device):
        """Quantised states must use exactly 4 bytes per parameter."""
        n = 256
        model = nn.Linear(n, n, bias=False).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(1, n, device=device)).sum().backward()
        opt.step()

        for p in model.parameters():
            s = opt.state[p]
            q_m_bytes = s["Q_m"].element_size() * s["Q_m"].numel()
            q_w_bytes = s["Q_w"].element_size() * s["Q_w"].numel()
            assert q_m_bytes + q_w_bytes == 4 * p.numel()


# ── mixed precision (bf16 / fp16) ────────────────────────────────────────────


class TestMixedPrecision:
    """Verify LUMA works correctly with reduced-precision params and grads."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_step_no_crash(self, dtype, device):
        """Single step succeeds; param dtype preserved, states are int16."""
        model = nn.Linear(16, 8).to(dtype).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(2, 16, device=device, dtype=dtype)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert p.dtype == dtype, f"dtype changed from {dtype}"
            assert p.isfinite().all(), f"NaN/Inf in params with {dtype}"
            s = opt.state[p]
            assert s["Q_m"].dtype == torch.int16
            assert s["Q_w"].dtype == torch.int16

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_twenty_steps_stable(self, dtype, device):
        """Twenty consecutive steps produce no NaN/Inf."""
        model = nn.Linear(16, 8).to(dtype).to(device)
        opt = LUMA(model.parameters(), lr=1e-3)
        for _ in range(20):
            model(torch.randn(2, 16, device=device, dtype=dtype)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert p.isfinite().all(), f"Non-finite params with {dtype} on {device}"

    def test_convergence_bfloat16(self, device):
        """LUMA converges on a regression task with bfloat16 params."""
        dtype = torch.bfloat16
        d = 16
        model = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(dtype).to(device)
        opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)

        X = torch.randn(100, d, device=device, dtype=dtype)
        y = torch.randn(100, 1, device=device, dtype=dtype)

        init_loss = None
        for _ in range(200):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X), y)
            if init_loss is None:
                init_loss = loss.item()
            loss.backward()
            opt.step()

        assert loss.item() < init_loss * 0.5, (
            f"bf16 convergence failed on {device}: "
            f"{init_loss:.4f} -> {loss.item():.4f}"
        )
