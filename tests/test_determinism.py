"""Tests for deterministic stochastic rounding (PRNG isolation).

Verifies that LUMA's quantisation noise is:
  1. Fully reproducible given the same seed.
  2. Isolated from the global PyTorch / Python PRNG state.
  3. Identical across simulated DDP ranks (same _base_seed).
  4. Actually influenced by the seed (sanity check).
"""

import random

import torch
import torch.nn as nn

from luma_optimizer import LUMA
from utils import seed_all


# ── helpers ──────────────────────────────────────────────────────────────────

def _step_with_fixed_grad(opt, x, grad, n_steps):
    """Run *n_steps* using a fixed (constant) gradient tensor."""
    for _ in range(n_steps):
        opt.zero_grad()
        x.grad = grad.clone()
        opt.step()


def _snapshot(opt, x):
    """Return cloned (param, Q_m, Q_w) for a single-param optimiser."""
    s = opt.state[x]
    return x.data.clone(), s["Q_m"].clone(), s["Q_w"].clone()


# =====================================================================
#  Core determinism suite
# =====================================================================


class TestDeterminism:
    """Stochastic rounding must be deterministic and isolated."""

    # ── 1. Same seed → bitwise identical result ─────────────────────

    def test_same_seed_bitwise_identical(self, device_backend):
        """Two independent runs with the same seed produce bitwise-
        identical parameters AND quantised states after 50 steps."""
        device, backend = device_backend
        d = 64

        def run():
            seed_all(42)
            x = torch.randn(d, device=device, requires_grad=True)
            g = torch.randn(d, device=device)
            opt = LUMA(
                [x], lr=1e-2, backend=backend, weight_decay=0.01,
            )
            _step_with_fixed_grad(opt, x, g, 50)
            return _snapshot(opt, x)

        p1, qm1, qw1 = run()
        p2, qm2, qw2 = run()

        assert torch.equal(p1, p2), (
            f"Params differ: max_diff={( p1 - p2).abs().max():.2e}"
        )
        assert torch.equal(qm1, qm2), "Q_m differs between identical runs"
        assert torch.equal(qw1, qw2), "Q_w differs between identical runs"

    # ── 2. Global RNG pollution does NOT affect the optimiser ───────

    def test_global_rng_isolation(self, device_backend):
        """Polluting the global PyTorch + Python RNG between steps
        must NOT change the optimiser's output."""
        device, backend = device_backend
        d = 64
        n_steps = 20

        # Shared deterministic inputs
        seed_all(0)
        x_init = torch.randn(d, device=device)
        g = torch.randn(d, device=device)

        # ── clean run ─────────────────────────────────────────────
        seed_all(42)
        x_clean = x_init.clone().requires_grad_(True)
        opt_clean = LUMA([x_clean], lr=1e-2, backend=backend)
        _step_with_fixed_grad(opt_clean, x_clean, g, n_steps)

        # ── polluted run ──────────────────────────────────────────
        seed_all(42)
        x_dirty = x_init.clone().requires_grad_(True)
        opt_dirty = LUMA([x_dirty], lr=1e-2, backend=backend)
        for _ in range(n_steps):
            opt_dirty.zero_grad()
            x_dirty.grad = g.clone()
            # Heavy pollution of BOTH global RNGs
            torch.rand(5000, device=device)
            random.random()
            opt_dirty.step()

        assert torch.equal(x_clean.data, x_dirty.data), (
            "Global RNG pollution affected parameters"
        )
        sc = opt_clean.state[x_clean]
        sd = opt_dirty.state[x_dirty]
        assert torch.equal(sc["Q_m"], sd["Q_m"]), (
            "Global RNG pollution affected Q_m"
        )
        assert torch.equal(sc["Q_w"], sd["Q_w"]), (
            "Global RNG pollution affected Q_w"
        )

    # ── 3. DDP simulation — same _base_seed, diverged global RNG ───

    def test_ddp_ranks_identical(self, device_backend):
        """Simulate two DDP ranks: same _base_seed, different global
        RNG trajectories (different forward pass / data loader noise).
        Both must produce identical optimiser state."""
        device, backend = device_backend
        d = 128
        n_steps = 30

        seed_all(0)
        x_init = torch.randn(d, device=device)
        g = torch.randn(d, device=device)

        results = []
        for rank in range(2):
            seed_all(42)
            x = x_init.clone().requires_grad_(True)
            opt = LUMA(
                [x], lr=1e-2, backend=backend, weight_decay=0.01,
            )
            for step_i in range(n_steps):
                opt.zero_grad()
                x.grad = g.clone()
                # Simulate rank-dependent RNG consumption
                # (data loading, dropout, augmentation, etc.)
                torch.rand(rank * 500 + step_i * 37 + 1, device=device)
                for _ in range(rank * 7 + step_i):
                    random.random()
                opt.step()

            results.append(_snapshot(opt, x))

        p0, qm0, qw0 = results[0]
        p1, qm1, qw1 = results[1]

        assert torch.equal(p0, p1), (
            f"DDP ranks diverged (params): "
            f"max_diff={(p0 - p1).abs().max():.2e}"
        )
        assert torch.equal(qm0, qm1), "DDP ranks diverged (Q_m)"
        assert torch.equal(qw0, qw1), "DDP ranks diverged (Q_w)"

    # ── 4. Different seed → different quantised states ─────────────

    def test_different_seed_different_quant(self, device_backend):
        """Different _base_seed must produce different Q_m
        (sanity: the Generator is actually being used)."""
        device, backend = device_backend
        d = 64

        # Fixed inputs (created BEFORE we manipulate the seed)
        seed_all(0)
        g = torch.randn(d, device=device)
        x_init = torch.randn(d, device=device)

        quants = []
        for prng_seed in [100, 200]:
            torch.manual_seed(prng_seed)
            x = x_init.clone().requires_grad_(True)
            opt = LUMA([x], lr=1e-2, backend=backend)
            # Step 1: params updated identically (before quant), but
            # Q_m/Q_w differ.  Step 2: different decoded states →
            # different EMA → different Q_m/Q_w AND params.
            _step_with_fixed_grad(opt, x, g, 2)
            quants.append(opt.state[x]["Q_m"].clone())

        assert not torch.equal(quants[0], quants[1]), (
            "Different base seeds produced identical Q_m — "
            "Generator may not be in effect"
        )

    # ── 5. Multi-parameter model ──────────────────────────────────

    def test_multi_param_determinism(self, device_backend):
        """Determinism holds for a model with multiple parameter
        tensors (weights + biases across layers)."""
        device, backend = device_backend

        def run():
            seed_all(42)
            model = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
            ).to(device)
            opt = LUMA(
                model.parameters(), lr=1e-3,
                backend=backend, weight_decay=0.01,
            )
            x = torch.randn(4, 32, device=device)
            for _ in range(30):
                opt.zero_grad()
                model(x).sum().backward()
                opt.step()
            return [p.data.clone() for p in model.parameters()]

        params_a = run()
        params_b = run()

        for i, (pa, pb) in enumerate(zip(params_a, params_b)):
            assert torch.equal(pa, pb), (
                f"Parameter {i} differs: "
                f"max_diff={(pa - pb).abs().max():.2e}"
            )

    # ── 6. Determinism after state_dict round-trip ─────────────────

    def test_determinism_survives_checkpoint(self, device_backend):
        """Save after 10 steps, load into a fresh optimiser, continue
        for 20 more steps → result identical to a straight 30-step run."""
        device, backend = device_backend
        d = 64

        seed_all(0)
        x_init = torch.randn(d, device=device)
        g = torch.randn(d, device=device)

        # ── straight run: 30 steps ────────────────────────────────
        seed_all(42)
        x_ref = x_init.clone().requires_grad_(True)
        opt_ref = LUMA([x_ref], lr=1e-2, backend=backend)
        _step_with_fixed_grad(opt_ref, x_ref, g, 30)

        # ── split run: 10 + checkpoint + 20 ───────────────────────
        seed_all(42)
        x_a = x_init.clone().requires_grad_(True)
        opt_a = LUMA([x_a], lr=1e-2, backend=backend)
        _step_with_fixed_grad(opt_a, x_a, g, 10)

        sd = opt_a.state_dict()

        # fresh optimiser, same param tensor
        x_b = x_a.clone().detach().requires_grad_(True)
        opt_b = LUMA([x_b], lr=1e-2, backend=backend)
        opt_b.load_state_dict(sd)
        _step_with_fixed_grad(opt_b, x_b, g, 20)

        if backend == "triton":
            # Triton state_dict round-trip converts S_m/S_v through
            # float32 → Python float → _init_triton_buffers, losing
            # ~1 ULP per grid recomputation.  Over 20 steps this
            # accumulates to low single-digit µ-level drift.
            assert torch.allclose(x_ref.data, x_b.data, atol=1e-5), (
                f"Checkpoint broke determinism (triton): "
                f"max_diff={(x_ref.data - x_b.data).abs().max():.2e}"
            )
        else:
            assert torch.equal(x_ref.data, x_b.data), (
                f"Checkpoint broke determinism: "
                f"max_diff={(x_ref.data - x_b.data).abs().max():.2e}"
            )

    # ── 7. import_adamw_state produces identical Q_m/Q_w ──────────

    def test_import_adamw_determinism(self, device_backend):
        """Importing the same AdamW state twice with the same seed
        produces bitwise-identical quantised states."""
        device, backend = device_backend
        d = 64

        # Train an AdamW optimizer to get a non-trivial state
        seed_all(0)
        x_adamw = torch.randn(d, device=device, requires_grad=True)
        opt_adamw = torch.optim.AdamW([x_adamw], lr=1e-2)
        g = torch.randn(d, device=device)
        for _ in range(5):
            opt_adamw.zero_grad()
            x_adamw.grad = g.clone()
            opt_adamw.step()
        adamw_sd = opt_adamw.state_dict()

        # Import into two LUMA instances with the same seed
        results = []
        for _ in range(2):
            seed_all(42)
            x = torch.randn(d, device=device, requires_grad=True)
            opt = LUMA([x], lr=1e-2, backend=backend)
            opt.import_adamw_state(adamw_sd)
            s = opt.state[x]
            results.append((s["Q_m"].clone(), s["Q_w"].clone()))

        assert torch.equal(results[0][0], results[1][0]), (
            "Q_m differs after identical import_adamw_state calls"
        )
        assert torch.equal(results[0][1], results[1][1]), (
            "Q_w differs after identical import_adamw_state calls"
        )
