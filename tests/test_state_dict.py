"""Tests for state dict save/load, AdamW export and import."""

import pytest
import torch
import torch.nn as nn

from luma_optimizer import LUMA
from utils import requires_cuda_triton, seed_all


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
#  State dict round-trip
# =====================================================================


class TestStateDict:
    """Save / load / resume — parametrised over device × backend."""

    def test_round_trip_single_param(self, device_backend):
        """Save after 5 steps, load, continue — must match straight run."""
        device, backend = device_backend
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # ── straight run: 10 steps ───────────────────────────────────
        seed_all(0)
        x_ref = torch.randn(d, device=device, requires_grad=True)
        opt_ref = LUMA([x_ref], backend=backend, **kw)
        _train_steps(opt_ref, x_ref, loss_fn, 10)

        # ── split run: 5 + save + load + 5 ──────────────────────────
        seed_all(0)
        x_a = torch.randn(d, device=device, requires_grad=True)
        opt_a = LUMA([x_a], backend=backend, **kw)
        _train_steps(opt_a, x_a, loss_fn, 5)

        sd = opt_a.state_dict()
        x_b = x_a.data.clone().detach().requires_grad_(True)
        opt_b = LUMA([x_b], backend=backend, **kw)
        opt_b.load_state_dict(sd)
        _train_steps(opt_b, x_b, loss_fn, 5)

        diff = (x_b.data - x_ref.data).abs().max().item()
        assert diff < 1e-6, (
            f"Round-trip diverged: max diff = {diff:.2e}"
        )

    def test_state_dict_keys(self, device_backend):
        """Saved state must contain step, param_id, Q_m, Q_w, S_m, S_v."""
        device, backend = device_backend
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend=backend)
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
        """Auto backend after multiple steps: state_dict must be clean."""
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

    def test_state_dict_before_step(self, device_backend):
        """state_dict / load_state_dict work before any steps."""
        device, backend = device_backend
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend=backend)

        sd = opt.state_dict()
        assert sd["state"] == {}
        assert len(sd["param_groups"]) == 1

        # Load empty state into fresh optimizer — must not crash
        model2 = nn.Linear(8, 4).to(device)
        opt2 = LUMA(model2.parameters(), backend=backend)
        opt2.load_state_dict(sd)

        # Should still be able to train after loading empty state
        model2(torch.randn(1, 8, device=device)).sum().backward()
        opt2.step()

    def test_param_id_preserved_after_load(self, device_backend):
        """param_id values survive a save/load round-trip."""
        device, backend = device_backend
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend=backend)
        model(torch.randn(1, 8, device=device)).sum().backward()
        opt.step()

        orig_ids = {
            idx: s["param_id"]
            for idx, s in opt.state_dict()["state"].items()
        }

        model2 = nn.Linear(8, 4).to(device)
        opt2 = LUMA(model2.parameters(), backend=backend)
        opt2.load_state_dict(opt.state_dict())

        loaded_ids = {
            idx: s["param_id"]
            for idx, s in opt2.state_dict()["state"].items()
        }
        assert orig_ids == loaded_ids

    def test_double_save_load(self, device_backend):
        """Two consecutive save/load cycles produce same result as straight run."""
        device, backend = device_backend
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # ── straight run: 15 steps ───────────────────────────────────
        seed_all(0)
        x_ref = torch.randn(d, device=device, requires_grad=True)
        opt_ref = LUMA([x_ref], backend=backend, **kw)
        _train_steps(opt_ref, x_ref, loss_fn, 15)

        # ── triple split: 5 + save/load + 5 + save/load + 5 ─────────
        seed_all(0)
        x = torch.randn(d, device=device, requires_grad=True)
        opt = LUMA([x], backend=backend, **kw)
        _train_steps(opt, x, loss_fn, 5)

        for _ in range(2):  # two save/load cycles
            sd = opt.state_dict()
            x = x.data.clone().detach().requires_grad_(True)
            opt = LUMA([x], backend=backend, **kw)
            opt.load_state_dict(sd)
            _train_steps(opt, x, loss_fn, 5)

        diff = (x.data - x_ref.data).abs().max().item()
        assert diff < 1e-6, (
            f"Double save/load diverged: max diff = {diff:.2e}"
        )


# =====================================================================
#  State dict round-trip — multi param group
# =====================================================================


class TestStateDictMultiGroup:
    def test_multi_param_groups(self, device_backend):
        """State dict preserves per-group hyper-parameters and state."""
        device, backend = device_backend
        model = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        opt = LUMA(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
            backend=backend,
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
            backend=backend,
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

        for i, (p1, p2) in enumerate(zip(model.parameters(), model2.parameters())):
            diff = (p1 - p2).abs().max().item()
            assert diff < 5e-4, (
                f"Multi-group round-trip diverged (param {i}): "
                f"max diff = {diff:.2e}"
            )


# =====================================================================
#  Export to AdamW
# =====================================================================


class TestExportAdamW:
    def test_export_structure(self, device_backend):
        """Exported state dict has correct AdamW keys."""
        device, backend = device_backend
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters(), backend=backend)
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

    def test_export_shapes_match(self, device_backend):
        """Exported exp_avg and exp_avg_sq shapes match parameters."""
        device, backend = device_backend
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters(), backend=backend)
        model(torch.randn(2, 16, device=device)).sum().backward()
        opt.step()

        adamw_sd = opt.export_adamw_state()
        params = list(model.parameters())
        for idx, s in adamw_sd["state"].items():
            assert s["exp_avg"].shape == params[idx].shape
            assert s["exp_avg_sq"].shape == params[idx].shape

    def test_export_loads_into_adamw(self, device_backend):
        """Exported state dict loads into a real AdamW without errors."""
        device, backend = device_backend
        model = nn.Linear(16, 8).to(device)
        opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01, backend=backend)

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

    def test_export_values_reasonable(self, device_backend):
        """Decoded m and v are non-zero and finite after training."""
        device, backend = device_backend
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        x = torch.randn(d, device=device, requires_grad=True)
        opt = LUMA([x], lr=1e-2, weight_decay=0.0, backend=backend)
        _train_steps(opt, x, loss_fn, 20)

        adamw_sd = opt.export_adamw_state()
        s = adamw_sd["state"][0]

        assert s["exp_avg"].isfinite().all(), "exp_avg has NaN/Inf"
        assert s["exp_avg_sq"].isfinite().all(), "exp_avg_sq has NaN/Inf"
        assert s["exp_avg"].abs().max() > 0, "exp_avg is all zeros"
        assert s["exp_avg_sq"].min() >= 0, "exp_avg_sq has negative values"
        assert s["step"].item() == 20.0

    def test_export_then_resume_close(self, device_backend):
        """LUMA→AdamW handoff: trajectories stay close on a quadratic."""
        device, backend = device_backend
        d = 64
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

        # ── Phase 1: LUMA trains 50 steps ────────────────────────────
        x = torch.randn(d, device=device, requires_grad=True)
        opt_luma = LUMA([x], backend=backend, **kw)
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

    def test_export_after_one_step(self, device_backend):
        """Export works at step 1 (edge case)."""
        device, backend = device_backend
        model = nn.Linear(8, 4).to(device)
        opt = LUMA(model.parameters(), backend=backend)
        model(torch.randn(1, 8, device=device)).sum().backward()
        opt.step()

        adamw_sd = opt.export_adamw_state()
        assert len(adamw_sd["state"]) == 2  # weight + bias

    def test_export_multi_group(self, device_backend):
        """export_adamw_state preserves per-group hyper-params and loads into AdamW."""
        device, backend = device_backend
        model = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)
        opt = LUMA(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
            backend=backend,
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
#  Import from AdamW
# =====================================================================


class TestImportAdamW:
    """AdamW → LUMA migration: state import and convergence."""

    def test_import_structure(self, device_backend):
        """Imported state has correct LUMA keys."""
        device, backend = device_backend
        model = nn.Linear(16, 8).to(device)
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model(torch.randn(2, 16, device=device)).sum().backward()
        adamw.step()

        luma = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01, backend=backend)
        luma.import_adamw_state(adamw.state_dict())

        sd = luma.state_dict()
        for idx, s in sd["state"].items():
            assert "step" in s
            assert "param_id" in s
            assert "Q_m" in s
            assert "Q_w" in s
            assert "S_m" in s
            assert "S_v" in s
            for k in s:
                assert not k.startswith("_"), f"Private key {k!r} leaked"

    def test_import_preserves_step(self, device_backend):
        """Step counter is correctly imported from AdamW."""
        device, backend = device_backend
        model = nn.Linear(16, 8).to(device)
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(10):
            adamw.zero_grad()
            model(torch.randn(2, 16, device=device)).sum().backward()
            adamw.step()

        luma = LUMA(model.parameters(), lr=1e-3, backend=backend)
        luma.import_adamw_state(adamw.state_dict())

        sd = luma.state_dict()
        for idx, s in sd["state"].items():
            assert s["step"] == 10

    def test_import_then_continue_converges(self, device_backend):
        """AdamW→LUMA handoff: LUMA continues to converge."""
        device, backend = device_backend
        d = 64
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

        # Train with AdamW for 50 steps
        x = torch.randn(d, device=device, requires_grad=True)
        adamw = torch.optim.AdamW([x], **kw)
        _train_steps(adamw, x, loss_fn, 50)

        loss_at_handoff = loss_fn(x).item()

        # Import into LUMA and continue
        luma = LUMA([x], backend=backend, **kw)
        luma.import_adamw_state(adamw.state_dict())
        _train_steps(luma, x, loss_fn, 50)

        final_loss = loss_fn(x).item()
        assert final_loss < loss_at_handoff, (
            f"LUMA didn't improve after AdamW import: "
            f"{loss_at_handoff:.6g} → {final_loss:.6g}"
        )

    def test_import_trajectory_close_to_adamw(self, device_backend):
        """AdamW→LUMA should track AdamW→AdamW closely for several steps."""
        device, backend = device_backend
        d = 64
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # Phase 1: AdamW warms up for 20 steps
        seed_all(0)
        x = torch.randn(d, device=device, requires_grad=True)
        adamw = torch.optim.AdamW([x], **kw)
        _train_steps(adamw, x, loss_fn, 20)

        adamw_sd = adamw.state_dict()
        x_snapshot = x.data.clone()

        # Branch A: AdamW continues for 10 more steps
        x_adamw = x_snapshot.clone().detach().requires_grad_(True)
        adamw_ref = torch.optim.AdamW([x_adamw], **kw)
        adamw_ref.load_state_dict(adamw_sd)
        _train_steps(adamw_ref, x_adamw, loss_fn, 10)

        # Branch B: LUMA continues from AdamW state for 10 more steps
        x_luma = x_snapshot.clone().detach().requires_grad_(True)
        luma = LUMA([x_luma], backend=backend, **kw)
        luma.import_adamw_state(adamw_sd)
        _train_steps(luma, x_luma, loss_fn, 10)

        diff = (x_luma.data - x_adamw.data).abs().max().item()
        scale = max(x_adamw.data.abs().max().item(), 1e-6)
        rel = diff / scale
        # Quantization introduces bounded error — trajectories stay close
        assert diff < 0.03 * scale + 1e-5, (
            f"LUMA diverged from AdamW continuation: "
            f"diff={diff:.4e}, scale={scale:.4f}, rel={rel:.6f}"
        )

    def test_import_roundtrip_quantization_error(self, device_backend):
        """AdamW → LUMA → export: quantization error is bounded."""
        device, backend = device_backend
        model = nn.Linear(32, 16).to(device)
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        for _ in range(10):
            adamw.zero_grad()
            model(torch.randn(4, 32, device=device)).sum().backward()
            adamw.step()

        original_sd = adamw.state_dict()

        # Import into LUMA, then export back
        luma = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01, backend=backend)
        luma.import_adamw_state(original_sd)
        exported_sd = luma.export_adamw_state()

        for idx in original_sd["state"]:
            orig_m = original_sd["state"][idx]["exp_avg"].float()
            exp_m = exported_sd["state"][idx]["exp_avg"].float()
            orig_v = original_sd["state"][idx]["exp_avg_sq"].float()
            exp_v = exported_sd["state"][idx]["exp_avg_sq"].float()

            # Momentum: int16 quantisation → bounded relative error
            m_scale = orig_m.abs().max().item()
            if m_scale > 1e-8:
                m_err = (exp_m - orig_m).abs().max().item() / m_scale
                assert m_err < 0.002, (
                    f"exp_avg roundtrip relative error {m_err:.6f} > 0.2% "
                    f"for param {idx}"
                )

            # Second moment: positive, bounded relative error
            v_scale = orig_v.max().item()
            if v_scale > 1e-8:
                v_err = (exp_v - orig_v).abs().max().item() / v_scale
                assert v_err < 0.002, (
                    f"exp_avg_sq roundtrip relative error {v_err:.6f} > 0.2% "
                    f"for param {idx}"
                )

    def test_import_multi_group(self, device_backend):
        """import_adamw_state works with multiple param groups."""
        device, backend = device_backend
        model = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1),
        ).to(device)

        adamw = torch.optim.AdamW(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
        )
        X = torch.randn(10, 10, device=device)
        for _ in range(5):
            adamw.zero_grad()
            model(X).sum().backward()
            adamw.step()

        adamw_sd = adamw.state_dict()

        # Import into LUMA with matching group structure
        luma = LUMA(
            [
                {"params": model[0].parameters(), "lr": 2e-3},
                {"params": model[2].parameters(), "lr": 5e-4},
            ],
            weight_decay=0.01,
            backend=backend,
        )
        luma.import_adamw_state(adamw_sd)

        sd = luma.state_dict()
        assert len(sd["state"]) == 4  # weight + bias per Linear
        for idx, s in sd["state"].items():
            assert s["step"] == 5

        # Continue training — must not crash and must converge
        for _ in range(5):
            luma.zero_grad()
            model(X).sum().backward()
            luma.step()

        for p in model.parameters():
            assert p.isfinite().all(), "Non-finite params after LUMA steps"

    def test_import_then_luma_save_load(self, device_backend):
        """AdamW → LUMA import → save → load → continue works correctly."""
        device, backend = device_backend
        d = 32
        a, b, loss_fn = _make_quadratic(d, device)
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # AdamW phase
        x = torch.randn(d, device=device, requires_grad=True)
        adamw = torch.optim.AdamW([x], **kw)
        _train_steps(adamw, x, loss_fn, 10)

        # Import into LUMA and train a bit
        luma = LUMA([x], backend=backend, **kw)
        luma.import_adamw_state(adamw.state_dict())
        _train_steps(luma, x, loss_fn, 5)

        # Save LUMA checkpoint
        sd = luma.state_dict()
        x_resumed = x.data.clone().detach().requires_grad_(True)
        luma2 = LUMA([x_resumed], backend=backend, **kw)
        luma2.load_state_dict(sd)

        # Continue training — must converge
        loss_before = loss_fn(x_resumed).item()
        _train_steps(luma2, x_resumed, loss_fn, 10)
        loss_after = loss_fn(x_resumed).item()

        assert loss_after < loss_before, (
            f"LUMA didn't converge after import→save→load: "
            f"{loss_before:.6g} → {loss_after:.6g}"
        )


# =====================================================================
#  Cross-backend state dict tests (CUDA + Triton only)
# =====================================================================


@requires_cuda_triton
class TestCrossBackend:
    """Cross-backend loading: PyTorch ↔ Triton checkpoints."""

    def test_cross_backend_pytorch_to_triton(self):
        """Checkpoint saved with PyTorch loads into Triton — loss decreases."""
        d = 64
        a, b, loss_fn = _make_quadratic(d, "cuda")
        kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        seed_all(42)
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

        seed_all(42)
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
