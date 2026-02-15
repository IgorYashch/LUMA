"""Tests for state dict save/load and AdamW export."""

import pytest
import torch
import torch.nn as nn

from luma_optimizer import LUMA
from luma_optimizer.functional import _precompute, _decode_momentum, _decode_preconditioner

# ── Triton + CUDA availability flag ─────────────────────────────────────────
_CUDA_AND_TRITON = torch.cuda.is_available()
try:
    import triton  # noqa: F401
except ImportError:
    _CUDA_AND_TRITON = False


# =====================================================================
#  Helpers
# =====================================================================

def _make_quadratic(d: int, device: str = "cpu"):
    """Quadratic loss: f(x) = 0.5 * sum(a * (x - b)^2)."""
    a = torch.rand(d, device=device) + 0.5
    b = torch.randn(d, device=device)
    return a, b, lambda x: 0.5 * (a * (x - b).square()).sum()


def _train_steps(opt, x, loss_fn, n_steps: int):
    """Run *n_steps* optimiser steps and return the final loss."""
    for _ in range(n_steps):
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        opt.step()
    return loss.item()


# =====================================================================
#  State dict round-trip — PyTorch backend
# =====================================================================


class TestStateDictPyTorch:
    """Save / load / resume produces identical results."""

    def test_round_trip_single_param(self, device):
        """Save after 5 steps, load, continue — must match straight run."""
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # ── straight run: 10 steps ───────────────────────────────────
        torch.manual_seed(0)
        x_ref = torch.randn(d, device=device, requires_grad=True)
        opt_ref = LUMA([x_ref], backend="pytorch", **kw)
        _train_steps(opt_ref, x_ref, loss_fn, 10)

        # ── split run: 5 + save + load + 5 ──────────────────────────
        torch.manual_seed(0)
        x_a = torch.randn(d, device=device, requires_grad=True)
        opt_a = LUMA([x_a], backend="pytorch", **kw)
        _train_steps(opt_a, x_a, loss_fn, 5)

        sd = opt_a.state_dict()
        x_b = x_a.data.clone().detach().requires_grad_(True)
        opt_b = LUMA([x_b], backend="pytorch", **kw)
        opt_b.load_state_dict(sd)
        _train_steps(opt_b, x_b, loss_fn, 5)

        assert torch.allclose(x_b.data, x_ref.data, atol=1e-6), (
            f"Round-trip diverged: max diff = "
            f"{(x_b.data - x_ref.data).abs().max().item():.2e}"
        )

    def test_state_dict_keys(self, device):
        """Saved state must contain step, param_id, Q_m, Q_w, S_m, S_v."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend="pytorch")
        model(torch.randn(2, 8, device=device)).sum().backward()
        opt.step()

        sd = opt.state_dict()
        for idx, s in sd["state"].items():
            assert "step" in s
            assert "param_id" in s
            assert "Q_m" in s
            assert "Q_w" in s
            assert "S_m" in s
            assert "S_v" in s
            # No internal buffers
            for k in s:
                assert not k.startswith("_"), f"Private key {k!r} leaked"

    def test_state_dict_no_triton_buffers(self, device):
        """Even after Triton steps, state_dict must not contain _ keys."""
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())  # auto backend
        for _ in range(3):
            model(torch.randn(2, 16, device=device)).sum().backward()
            opt.step()
            opt.zero_grad()

        sd = opt.state_dict()
        for idx, s in sd["state"].items():
            for k in s:
                assert not k.startswith("_"), f"Private key {k!r} in state_dict"
            assert "S_m" in s
            assert "S_v" in s

    def test_state_dict_before_step(self, device):
        """state_dict / load_state_dict work before any steps."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend="pytorch")

        sd = opt.state_dict()
        assert sd["state"] == {}
        assert len(sd["param_groups"]) == 1

        # Load empty state into fresh optimizer — must not crash
        model2 = nn.Linear(8, 4).to(device)
        opt2 = LUMA(model2.parameters(), backend="pytorch")
        opt2.load_state_dict(sd)

        # Should still be able to train after loading empty state
        model2(torch.randn(1, 8, device=device)).sum().backward()
        opt2.step()

    def test_param_id_counter_after_load(self, device):
        """_next_param_id continues from the highest loaded ID."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend="pytorch")
        model(torch.randn(1, 8, device=device)).sum().backward()
        opt.step()

        sd = opt.state_dict()
        max_id = max(s["param_id"] for s in sd["state"].values())

        model2 = nn.Linear(8, 4).to(device)
        opt2 = LUMA(model2.parameters(), backend="pytorch")
        opt2.load_state_dict(sd)

        assert opt2._next_param_id == max_id + 1

    def test_double_save_load(self, device):
        """Two consecutive save/load cycles produce same result as straight run."""
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # ── straight run: 15 steps ───────────────────────────────────
        torch.manual_seed(0)
        x_ref = torch.randn(d, device=device, requires_grad=True)
        opt_ref = LUMA([x_ref], backend="pytorch", **kw)
        _train_steps(opt_ref, x_ref, loss_fn, 15)

        # ── triple split: 5 + save/load + 5 + save/load + 5 ─────────
        torch.manual_seed(0)
        x = torch.randn(d, device=device, requires_grad=True)
        opt = LUMA([x], backend="pytorch", **kw)
        _train_steps(opt, x, loss_fn, 5)

        for _ in range(2):  # two save/load cycles
            sd = opt.state_dict()
            x = x.data.clone().detach().requires_grad_(True)
            opt = LUMA([x], backend="pytorch", **kw)
            opt.load_state_dict(sd)
            _train_steps(opt, x, loss_fn, 5)

        assert torch.allclose(x.data, x_ref.data, atol=1e-6), (
            f"Double save/load diverged: max diff = "
            f"{(x.data - x_ref.data).abs().max().item():.2e}"
        )


# =====================================================================
#  State dict round-trip — multi param group
# =====================================================================


class TestStateDictMultiGroup:
    def test_multi_param_groups(self, device):
        """State dict preserves per-group hyper-parameters and state."""
        model = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        opt = LUMA(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
        )

        X = torch.randn(20, 10, device=device)
        for _ in range(5):
            opt.zero_grad()
            model(X).sum().backward()
            opt.step()

        sd = opt.state_dict()

        # Restore into a fresh optimiser
        model2 = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        # Copy weights so the models match
        model2.load_state_dict(model.state_dict())

        opt2 = LUMA(
            [
                {"params": model2[0].parameters(), "lr": 2e-3},
                {"params": model2[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
        )
        opt2.load_state_dict(sd)

        # Continue training — both should stay in sync
        for _ in range(5):
            opt.zero_grad()
            opt2.zero_grad()
            loss1 = model(X).sum()
            loss2 = model2(X).sum()
            loss1.backward()
            loss2.backward()
            opt.step()
            opt2.step()

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-4), (
                f"Multi-group round-trip diverged: "
                f"max diff = {(p1 - p2).abs().max().item():.2e}"
            )


# =====================================================================
#  Export to AdamW
# =====================================================================


class TestExportAdamW:
    def test_export_structure(self, device):
        """Exported state dict has correct AdamW keys."""
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(2, 16, device=device)).sum().backward()
        opt.step()

        adamw_sd = opt.export_adamw_state()

        assert "state" in adamw_sd
        assert "param_groups" in adamw_sd
        for idx, s in adamw_sd["state"].items():
            assert "step" in s
            assert "exp_avg" in s
            assert "exp_avg_sq" in s
            assert isinstance(s["step"], torch.Tensor)
            assert s["exp_avg"].dtype == torch.float32
            assert s["exp_avg_sq"].dtype == torch.float32

    def test_export_shapes_match(self, device):
        """Exported exp_avg and exp_avg_sq shapes match parameters."""
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(2, 16, device=device)).sum().backward()
        opt.step()

        adamw_sd = opt.export_adamw_state()
        params = list(model.parameters())
        for idx, s in adamw_sd["state"].items():
            assert s["exp_avg"].shape == params[idx].shape
            assert s["exp_avg_sq"].shape == params[idx].shape

    def test_export_loads_into_adamw(self, device):
        """Exported state dict loads into a real AdamW without errors."""
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)

        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(2, 16, device=device)).sum().backward()
            opt.step()

        adamw_sd = opt.export_adamw_state()
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        adamw.load_state_dict(adamw_sd)

        # AdamW should be able to continue stepping
        for _ in range(5):
            adamw.zero_grad()
            model(torch.randn(2, 16, device=device)).sum().backward()
            adamw.step()

        for p in model.parameters():
            assert p.isfinite().all(), "Non-finite params after AdamW steps"

    def test_export_values_reasonable(self, device):
        """Decoded m and v are non-zero and finite after training."""
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        x = torch.randn(d, device=device, requires_grad=True)
        opt = LUMA([x], lr=1e-2, weight_decay=0.0)
        _train_steps(opt, x, loss_fn, 20)

        adamw_sd = opt.export_adamw_state()
        s = adamw_sd["state"][0]

        assert s["exp_avg"].isfinite().all(), "exp_avg has NaN/Inf"
        assert s["exp_avg_sq"].isfinite().all(), "exp_avg_sq has NaN/Inf"
        assert s["exp_avg"].abs().max() > 0, "exp_avg is all zeros"
        assert s["exp_avg_sq"].min() >= 0, "exp_avg_sq has negative values"
        assert s["step"].item() == 20.0

    def test_export_then_resume_close(self, device):
        """LUMA→AdamW handoff: trajectories stay close on a quadratic."""
        d = 64
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

        # ── Phase 1: LUMA trains 50 steps ────────────────────────────
        x = torch.randn(d, device=device, requires_grad=True)
        opt_luma = LUMA([x], **kw)
        _train_steps(opt_luma, x, loss_fn, 50)

        loss_at_handoff = loss_fn(x).item()

        # ── Phase 2: export → AdamW continues 50 steps ──────────────
        adamw_sd = opt_luma.export_adamw_state()
        opt_adamw = torch.optim.AdamW([x], **kw)
        opt_adamw.load_state_dict(adamw_sd)
        _train_steps(opt_adamw, x, loss_fn, 50)

        # Must converge further (not diverge)
        final_loss = loss_fn(x).item()
        assert final_loss < loss_at_handoff, (
            f"AdamW didn't improve after LUMA handoff: "
            f"{loss_at_handoff:.6g} → {final_loss:.6g}"
        )

    def test_export_after_one_step(self, device):
        """Export works at step 1 (edge case)."""
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters())
        model(torch.randn(1, 8, device=device)).sum().backward()
        opt.step()

        adamw_sd = opt.export_adamw_state()
        assert len(adamw_sd["state"]) == 2  # weight + bias

    def test_export_multi_group(self, device):
        """export_adamw_state preserves per-group hyper-params and loads into AdamW."""
        model = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        opt = LUMA(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
        )
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(5, 10, device=device)).sum().backward()
            opt.step()

        adamw_sd = opt.export_adamw_state()

        assert len(adamw_sd["param_groups"]) == 2
        assert adamw_sd["param_groups"][0]["lr"] == 2e-3
        assert adamw_sd["param_groups"][1]["lr"] == 5e-4
        assert len(adamw_sd["state"]) == 4  # weight + bias per Linear

        # Load into real AdamW
        model2 = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        model2.load_state_dict(model.state_dict())
        adamw = torch.optim.AdamW(
            [
                {"params": model2[0].parameters(), "lr": 2e-3},
                {"params": model2[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
        )
        adamw.load_state_dict(adamw_sd)

        # AdamW must continue training without errors
        adamw.zero_grad()
        model2(torch.randn(5, 10, device=device)).sum().backward()
        adamw.step()
        for p in model2.parameters():
            assert p.isfinite().all()


# =====================================================================
#  Triton-specific state dict tests (CUDA only)
# =====================================================================


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="Requires CUDA + Triton")
class TestStateDictTriton:
    """Triton backend: state_dict round-trip & cross-backend loading."""

    def test_round_trip_triton(self):
        """Save/load with Triton backend: resumed training matches."""
        d = 64
        a, b, loss_fn = _make_quadratic(d, "cuda")
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # Straight run: 10 steps
        torch.manual_seed(42)
        x_ref = torch.randn(d, device="cuda", requires_grad=True)
        opt_ref = LUMA([x_ref], backend="triton", **kw)
        _train_steps(opt_ref, x_ref, loss_fn, 10)

        # Split: 5 + save/load + 5
        torch.manual_seed(42)
        x_a = torch.randn(d, device="cuda", requires_grad=True)
        opt_a = LUMA([x_a], backend="triton", **kw)
        _train_steps(opt_a, x_a, loss_fn, 5)

        sd = opt_a.state_dict()

        x_b = x_a.data.clone().detach().requires_grad_(True)
        opt_b = LUMA([x_b], backend="triton", **kw)
        opt_b.load_state_dict(sd)
        _train_steps(opt_b, x_b, loss_fn, 5)

        diff = (x_b.data - x_ref.data).abs().max().item()
        scale = x_ref.data.abs().max().item()
        assert diff < 1e-4 * scale + 1e-7, (
            f"Triton round-trip diverged: diff={diff:.2e}, scale={scale:.4f}"
        )

    def test_cross_backend_pytorch_to_triton(self):
        """Checkpoint saved with PyTorch loads into Triton — loss decreases."""
        d = 64
        a, b, loss_fn = _make_quadratic(d, "cuda")
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        torch.manual_seed(42)
        x_pt = torch.randn(d, device="cuda", requires_grad=True)
        opt_pt = LUMA([x_pt], backend="pytorch", **kw)
        _train_steps(opt_pt, x_pt, loss_fn, 5)
        sd = opt_pt.state_dict()

        x_tri = x_pt.data.clone().detach().requires_grad_(True)
        opt_tri = LUMA([x_tri], backend="triton", **kw)
        opt_tri.load_state_dict(sd)
        loss_before = loss_fn(x_tri).item()
        _train_steps(opt_tri, x_tri, loss_fn, 5)
        loss_after = loss_fn(x_tri).item()

        assert loss_after < loss_before, (
            f"PyTorch→Triton cross-load didn't converge: "
            f"{loss_before:.6g} → {loss_after:.6g}"
        )

    def test_cross_backend_triton_to_pytorch(self):
        """Checkpoint saved with Triton loads into PyTorch — loss decreases."""
        d = 64
        a, b, loss_fn = _make_quadratic(d, "cuda")
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        torch.manual_seed(42)
        x_tri = torch.randn(d, device="cuda", requires_grad=True)
        opt_tri = LUMA([x_tri], backend="triton", **kw)
        _train_steps(opt_tri, x_tri, loss_fn, 5)
        sd = opt_tri.state_dict()

        x_pt = x_tri.data.clone().detach().requires_grad_(True)
        opt_pt = LUMA([x_pt], backend="pytorch", **kw)
        opt_pt.load_state_dict(sd)
        loss_before = loss_fn(x_pt).item()
        _train_steps(opt_pt, x_pt, loss_fn, 5)
        loss_after = loss_fn(x_pt).item()

        assert loss_after < loss_before, (
            f"Triton→PyTorch cross-load didn't converge: "
            f"{loss_before:.6g} → {loss_after:.6g}"
        )

    def test_export_triton_backend(self):
        """export_adamw_state works with Triton backend (reads _new_grid)."""
        model = nn.Linear(16, 8).to("cuda")
        opt = LUMA(model.parameters(), backend="triton")
        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(2, 16, device="cuda")).sum().backward()
            opt.step()

        adamw_sd = opt.export_adamw_state()
        for idx, s in adamw_sd["state"].items():
            assert s["exp_avg"].isfinite().all()
            assert s["exp_avg_sq"].isfinite().all()
            assert s["exp_avg_sq"].min() >= 0

        # Must load into AdamW and continue training
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
        adamw.load_state_dict(adamw_sd)
        adamw.zero_grad()
        model(torch.randn(2, 16, device="cuda")).sum().backward()
        adamw.step()
        for p in model.parameters():
            assert p.isfinite().all()
