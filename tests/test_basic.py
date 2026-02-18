"""Basic API and smoke tests for LUMA optimizer."""

import pytest
import torch
import torch.nn as nn

from luma_optimizer import LUMA


# =====================================================================
#  Creation (device-independent)
# =====================================================================


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


# =====================================================================
#  Stepping (parametrised over devices)
# =====================================================================


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
        """step(closure) calls the closure and returns its loss."""
        model = nn.Linear(4, 2).to(device)
        opt = LUMA(model.parameters())
        x = torch.randn(2, 4, device=device)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        returned = opt.step(closure)
        assert returned is not None
        assert torch.is_tensor(returned)


# =====================================================================
#  Edge cases
# =====================================================================


class TestEdgeCases:
    def test_lr_zero_no_update(self, device):
        """With lr=0, parameters must remain unchanged after stepping."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), lr=0.0, weight_decay=0.0)
        before = [p.data.clone() for p in model.parameters()]

        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()

        for p, b in zip(model.parameters(), before):
            assert torch.equal(p.data, b)

    def test_scalar_parameter(self, device):
        """A single-element (scalar-shaped) parameter must work."""
        x = torch.tensor(5.0, device=device, requires_grad=True)
        opt = LUMA([x], lr=1e-2)
        for _ in range(5):
            opt.zero_grad()
            (x ** 2).backward()
            opt.step()
        assert x.item() < 5.0

    def test_gradient_accumulation(self, device):
        """Multiple backward passes before one step must work correctly."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), lr=1e-3)
        before = [p.data.clone() for p in model.parameters()]

        for _ in range(3):
            model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()

        for p, b in zip(model.parameters(), before):
            assert not torch.equal(p.data, b), "Params should change"

    def test_zero_grad_set_to_none_false(self, device):
        """zero_grad(set_to_none=False) zeros gradients instead of None."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), lr=1e-3)

        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=False)

        for p in model.parameters():
            assert p.grad is not None
            assert (p.grad == 0).all()

        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()

        for p in model.parameters():
            assert p.isfinite().all()

    def test_zero_grad_set_to_none_true(self, device):
        """zero_grad(set_to_none=True) sets gradients to None."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), lr=1e-3)

        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        for p in model.parameters():
            assert p.grad is None

        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()

        for p in model.parameters():
            assert p.isfinite().all()


# =====================================================================
#  State shapes & dtypes
# =====================================================================


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


# =====================================================================
#  Memory footprint
# =====================================================================


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
