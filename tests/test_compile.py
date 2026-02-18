"""Tests that the Triton kernel pipeline is torch.compile-compatible
and does not introduce graph breaks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from utils import requires_cuda_triton


# Dynamo must be reset between compile tests to avoid stale cache.
@pytest.fixture(autouse=True)
def _dynamo_reset():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _make_triton_buffers(n: int, device: str = "cuda"):
    """Allocate all tensors needed by ``luma_triton_step``."""
    from luma_optimizer.config import get_kernel_config

    kcfg = get_kernel_config(torch.device(device))
    block_size = kcfg["BLOCK_SIZE"]
    num_blocks = (n + block_size - 1) // block_size

    param = torch.randn(n, device=device)
    grad = torch.randn(n, device=device)
    Q_m = torch.zeros(n, device=device, dtype=torch.int16)
    Q_w = torch.zeros(n, device=device, dtype=torch.int16)

    old_scalars = torch.zeros(8, device=device)
    old_scalars[4] = 0.001  # bc2
    old_scalars[5] = 0.1    # bc1

    new_grid = torch.zeros(8, device=device)
    new_grid[0] = 1.0  # S_m
    new_grid[1] = 1.0  # S_v

    s_m_block = torch.empty(num_blocks, device=device)
    s_v_block = torch.empty(num_blocks, device=device)
    step_tensor = torch.tensor([1], device=device, dtype=torch.int32)

    return dict(
        param=param, grad=grad, Q_m=Q_m, Q_w=Q_w,
        old_scalars=old_scalars, new_grid=new_grid,
        s_m_block=s_m_block, s_v_block=s_v_block,
        step_tensor=step_tensor,
    )


# =====================================================================
#  fullgraph=True — verifies zero graph breaks in the Triton pipeline
# =====================================================================


@requires_cuda_triton
class TestTritonCompileFullgraph:
    """luma_triton_step must be torch.compile-able with fullgraph=True."""

    def test_compiles_without_graph_breaks(self):
        from luma_optimizer.triton_kernel import luma_triton_step

        bufs = _make_triton_buffers(4096)
        param_before = bufs["param"].clone()

        compiled_step = torch.compile(luma_triton_step, fullgraph=True)
        compiled_step(
            bufs["param"], bufs["grad"], bufs["Q_m"], bufs["Q_w"],
            bufs["old_scalars"], bufs["new_grid"],
            bufs["s_m_block"], bufs["s_v_block"],
            0.9, 0.999, 1e-8, 1e-3, 0.01,
            0, bufs["step_tensor"], 42,
        )
        torch.cuda.synchronize()

        assert not torch.equal(bufs["param"], param_before), (
            "Parameters should have been updated"
        )
        assert bufs["param"].isfinite().all(), "Parameters must be finite"

    def test_multiple_compiled_steps(self):
        """Repeated calls through the same compiled function must work."""
        from luma_optimizer.triton_kernel import luma_triton_step

        bufs = _make_triton_buffers(2048)
        compiled_step = torch.compile(luma_triton_step, fullgraph=True)

        for i in range(5):
            bufs["grad"] = torch.randn_like(bufs["param"])
            compiled_step(
                bufs["param"], bufs["grad"], bufs["Q_m"], bufs["Q_w"],
                bufs["old_scalars"], bufs["new_grid"],
                bufs["s_m_block"], bufs["s_v_block"],
                0.9, 0.999, 1e-8, 1e-3, 0.01,
                0, bufs["step_tensor"], 42,
            )

        torch.cuda.synchronize()
        assert bufs["param"].isfinite().all()
        assert int(bufs["step_tensor"].item()) == 1 + 5

    @pytest.mark.parametrize("n", [1, 127, 1024, 8192])
    def test_various_sizes(self, n: int):
        """fullgraph compilation must not depend on tensor size."""
        from luma_optimizer.triton_kernel import luma_triton_step

        bufs = _make_triton_buffers(n)
        compiled_step = torch.compile(luma_triton_step, fullgraph=True)
        compiled_step(
            bufs["param"], bufs["grad"], bufs["Q_m"], bufs["Q_w"],
            bufs["old_scalars"], bufs["new_grid"],
            bufs["s_m_block"], bufs["s_v_block"],
            0.9, 0.999, 1e-8, 1e-3, 0.01,
            0, bufs["step_tensor"], 42,
        )
        torch.cuda.synchronize()
        assert bufs["param"].isfinite().all()


# =====================================================================
#  torch._dynamo.explain — explicit graph-break diagnosis
# =====================================================================


@requires_cuda_triton
class TestTritonDynamoExplain:
    """Use dynamo's explain() to verify zero graph breaks with diagnostics."""

    def test_zero_graph_breaks(self):
        from luma_optimizer.triton_kernel import luma_triton_step

        bufs = _make_triton_buffers(4096)
        explanation = torch._dynamo.explain(luma_triton_step)(
            bufs["param"], bufs["grad"], bufs["Q_m"], bufs["Q_w"],
            bufs["old_scalars"], bufs["new_grid"],
            bufs["s_m_block"], bufs["s_v_block"],
            0.9, 0.999, 1e-8, 1e-3, 0.01,
            0, bufs["step_tensor"], 42,
        )
        torch.cuda.synchronize()
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}.\n"
            f"Break reasons:\n{explanation.break_reasons}"
        )


# =====================================================================
#  Compiled training loop smoke test
# =====================================================================


@requires_cuda_triton
class TestCompiledTrainingLoop:
    """End-to-end: full train step (fwd + bwd + opt.step) under torch.compile."""

    def test_compiled_train_step(self):
        from luma_optimizer import LUMA

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).cuda()
        opt = LUMA(model.parameters(), lr=1e-3, backend="triton")

        x = torch.randn(8, 32, device="cuda")
        y = torch.randn(8, 1, device="cuda")

        @torch.compile()
        def train_step(x, y):
            opt.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            opt.step()
            return loss

        losses = []
        for _ in range(10):
            loss = train_step(x, y)
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(losses)))
        for p in model.parameters():
            assert p.isfinite().all()

    def test_compiled_train_step_with_autocast(self):
        """torch.compile over full train step with autocast + Triton LUMA."""
        from luma_optimizer import LUMA

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).cuda()
        opt = LUMA(model.parameters(), lr=1e-3, backend="triton")

        x = torch.randn(8, 32, device="cuda")
        y = torch.randn(8, 1, device="cuda")

        @torch.compile()
        def train_step(x, y):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = (pred.float() - y).square().mean()
            loss.backward()
            opt.step()
            return loss

        for _ in range(5):
            train_step(x, y)

        for p in model.parameters():
            assert p.isfinite().all()
