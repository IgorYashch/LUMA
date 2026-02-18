"""FSDP / FSDP2 integration tests (require >= 2 CUDA GPUs).

Each test spawns ``world_size`` GPU worker processes via
``torch.multiprocessing.spawn``.  If any worker raises, pytest reports the
failure.  Tests are skipped entirely on single-GPU or CPU-only machines.
"""

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from luma_optimizer import LUMA
from luma_optimizer.config import get_kernel_config
from utils import CUDA_AND_TRITON, MULTI_GPU, seed_all

# ── FSDP2 availability ──────────────────────────────────────────────────────
_FSDP2_AVAILABLE = False
try:
    from torch.distributed._composable.fsdp import fully_shard  # noqa: F401

    _FSDP2_AVAILABLE = True
except ImportError:
    pass

requires_fsdp2 = pytest.mark.skipif(
    not (MULTI_GPU and _FSDP2_AVAILABLE),
    reason="Requires >= 2 CUDA GPUs + PyTorch FSDP2 (torch >= 2.4)",
)

requires_fsdp2_triton = pytest.mark.skipif(
    not (MULTI_GPU and _FSDP2_AVAILABLE and CUDA_AND_TRITON),
    reason="Requires >= 2 CUDA GPUs + FSDP2 + Triton",
)

requires_multi_gpu = pytest.mark.skipif(
    not MULTI_GPU,
    reason="Requires >= 2 CUDA GPUs",
)


# =====================================================================
#  Distributed test harness
# =====================================================================

def _find_free_port() -> int:
    """Grab an ephemeral port that is (momentarily) free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_workers(fn, world_size: int = 2):
    """Launch *fn(rank, world_size)* on *world_size* GPU processes."""
    port = _find_free_port()
    mp.spawn(
        _worker_entry,
        args=(world_size, port, fn),
        nprocs=world_size,
        join=True,
    )


def _worker_entry(rank: int, world_size: int, port: int, fn):
    """Per-process bootstrap: init NCCL, run test, tear down."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # Isolate Triton JIT cache per worker to prevent compilation races
    # when multiple spawned processes compile the same kernel concurrently.
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), f".triton_test_rank{rank}_{os.getpid()}",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    seed_all(42)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


# =====================================================================
#  Shared model
# =====================================================================

class SimpleMLP(nn.Module):
    """Tiny MLP used across FSDP tests."""

    def __init__(self, d_in=64, d_hidden=128, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# =====================================================================
#  FSDP2 tests (fully_shard / DTensor)
# =====================================================================


@requires_fsdp2
class TestFSDP2:

    def test_basic_convergence(self):
        """FSDP2 + LUMA converges on a fixed regression task."""
        _run_workers(_fsdp2_convergence_impl)

    def test_state_dict_roundtrip(self):
        """state_dict → load_state_dict under FSDP2: training continues."""
        _run_workers(_fsdp2_state_dict_impl)

    def test_per_shard_scaling(self):
        """S_m / S_v are computed per local shard, not globally."""
        _run_workers(_fsdp2_per_shard_scaling_impl)

    def test_export_adamw(self):
        """export_adamw_state produces valid per-shard states."""
        _run_workers(_fsdp2_export_adamw_impl)


@requires_fsdp2_triton
class TestFSDP2Triton:

    def test_triton_buffer_sizes(self):
        """Triton buffers are sized for the local shard, not global param."""
        _run_workers(_fsdp2_triton_buffers_impl)


# ── FSDP2 worker implementations ─────────────────────────────────────────────

def _fsdp2_convergence_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    model = SimpleMLP().cuda()
    fully_shard(model.fc1)
    fully_shard(model.fc2)
    fully_shard(model)

    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Fixed data — same on all ranks (CUDA generators seeded identically)
    torch.cuda.manual_seed(0)
    x = torch.randn(32, 64, device="cuda")
    target = torch.randn(32, 1, device="cuda")

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    init_avg = sum(losses[:5]) / 5
    final_avg = sum(losses[-5:]) / 5
    assert final_avg < init_avg * 0.5, (
        f"[Rank {rank}] FSDP2 + LUMA didn't converge: "
        f"first_5_avg={init_avg:.4f}, last_5_avg={final_avg:.4f}"
    )


def _fsdp2_state_dict_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    model = SimpleMLP().cuda()
    fully_shard(model.fc1)
    fully_shard(model.fc2)
    fully_shard(model)

    opt = LUMA(model.parameters(), lr=1e-2)

    torch.cuda.manual_seed(0)
    x = torch.randn(16, 64, device="cuda")
    target = torch.randn(16, 1, device="cuda")

    # Train 5 steps
    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()
        opt.step()

    loss_before_save = nn.functional.mse_loss(model(x), target).item()

    # ── Save ─────────────────────────────────────────────────────────
    sd = opt.state_dict()

    # Verify state dict is clean — no private Triton keys leaked
    for idx, s in sd["state"].items():
        for k in s:
            assert not k.startswith("_"), (
                f"Private key {k!r} leaked in FSDP2 state_dict"
            )
        assert "S_m" in s
        assert "S_v" in s
        assert "Q_m" in s
        assert "Q_w" in s

    # ── Load into a fresh optimizer ──────────────────────────────────
    opt2 = LUMA(model.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)

    # Continue training — loss must keep decreasing
    for _ in range(10):
        opt2.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()
        opt2.step()

    loss_after_load = nn.functional.mse_loss(model(x), target).item()
    assert loss_after_load < loss_before_save, (
        f"[Rank {rank}] FSDP2 state_dict round-trip broke convergence: "
        f"before_save={loss_before_save:.6g}, after_load={loss_after_load:.6g}"
    )


def _fsdp2_per_shard_scaling_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    # Large linear — FSDP shards weight[256, 128] along dim 0,
    # so rank 0 gets rows 0-127, rank 1 gets rows 128-255.
    model = nn.Linear(128, 256, bias=False).cuda()
    fully_shard(model)

    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Structured target: large for first-half outputs, small for second-half.
    # This makes grad rows 0-127 large and rows 128-255 small,
    # which aligns with the shard split → different S_m per rank.
    torch.cuda.manual_seed(0)
    x = torch.randn(32, 128, device="cuda")
    target = torch.zeros(32, 256, device="cuda")
    target[:, :128] = 10.0
    target[:, 128:] = 0.01

    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()
        opt.step()

    # ── Gather S_m from all ranks and compare ────────────────────────
    for p in model.parameters():
        state = opt.state[p]
        if "_new_grid" in state:
            S_m = state["_new_grid"][0].item()
        else:
            S_m = state["S_m"]

        local_t = torch.tensor([S_m], device="cuda")
        gathered = [torch.zeros(1, device="cuda") for _ in range(world_size)]
        dist.all_gather(gathered, local_t)

        if rank == 0:
            values = [g.item() for g in gathered]
            rel_diff = abs(values[0] - values[1]) / max(
                abs(values[0]), abs(values[1]), 1e-10,
            )
            assert rel_diff > 0.01, (
                f"S_m nearly identical across ranks "
                f"({values[0]:.6g} vs {values[1]:.6g}, rel_diff={rel_diff:.4f}) "
                f"— scaling may not be per-shard"
            )

        # Sanity: Q_m shape must match local shard, not global param
        local_numel = opt._unwrap_tensor(p).numel()
        assert state["Q_m"].numel() == local_numel, (
            f"Q_m numel {state['Q_m'].numel()} != local shard {local_numel}"
        )


def _fsdp2_export_adamw_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    model = SimpleMLP().cuda()
    fully_shard(model.fc1)
    fully_shard(model.fc2)
    fully_shard(model)

    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.01)

    torch.cuda.manual_seed(0)
    x = torch.randn(16, 64, device="cuda")
    target = torch.randn(16, 1, device="cuda")

    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()
        opt.step()

    adamw_sd = opt.export_adamw_state()

    assert "state" in adamw_sd
    assert "param_groups" in adamw_sd

    # Exported shapes must match LOCAL shard shapes (not global DTensor)
    params = list(model.parameters())
    for idx, s in adamw_sd["state"].items():
        local_shape = opt._unwrap_tensor(params[idx]).shape
        assert s["exp_avg"].shape == local_shape, (
            f"exp_avg shape {s['exp_avg'].shape} != local shard {local_shape}"
        )
        assert s["exp_avg_sq"].shape == local_shape, (
            f"exp_avg_sq shape {s['exp_avg_sq'].shape} != "
            f"local shard {local_shape}"
        )
        assert s["exp_avg"].isfinite().all(), "exp_avg has NaN/Inf"
        assert s["exp_avg_sq"].isfinite().all(), "exp_avg_sq has NaN/Inf"
        assert s["exp_avg_sq"].min() >= 0, "exp_avg_sq has negative values"


def _fsdp2_triton_buffers_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    model = nn.Linear(128, 256, bias=False).cuda()
    fully_shard(model)

    opt = LUMA(model.parameters(), lr=1e-2, backend="triton")

    torch.cuda.manual_seed(0)
    x = torch.randn(16, 128, device="cuda")

    # Step 1 = init (FP32), step 2 = first Triton quantised step
    for _ in range(2):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()

    for p in model.parameters():
        state = opt.state[p]
        local_shard = opt._unwrap_tensor(p)
        local_numel = local_shard.numel()

        # Quantised states must match LOCAL shard
        assert state["Q_m"].numel() == local_numel, (
            f"Q_m numel {state['Q_m'].numel()} != local shard {local_numel}"
        )
        assert state["Q_w"].numel() == local_numel, (
            f"Q_w numel {state['Q_w'].numel()} != local shard {local_numel}"
        )

        # Block reduction buffers sized for local shard, not global
        kcfg = get_kernel_config(local_shard.device)
        block_size = kcfg["BLOCK_SIZE"]
        expected_blocks = (local_numel + block_size - 1) // block_size
        assert state["_s_m_block"].numel() == expected_blocks, (
            f"_s_m_block has {state['_s_m_block'].numel()} blocks, "
            f"expected {expected_blocks} for {local_numel} local elements"
        )
        assert state["_s_v_block"].numel() == expected_blocks, (
            f"_s_v_block has {state['_s_v_block'].numel()} blocks, "
            f"expected {expected_blocks} for {local_numel} local elements"
        )

        # Step tensor must exist and be on GPU
        assert "_step_tensor" in state
        assert state["_step_tensor"].device.type == "cuda"
        assert state["_step_tensor"].item() == 2


# =====================================================================
#  FSDP1 tests (FullyShardedDataParallel — legacy)
# =====================================================================


@requires_multi_gpu
class TestDDPDeterminism:

    def test_ranks_identical_after_training(self):
        """Real DDP: all ranks have identical params after training
        with different data and polluted global CUDA RNG."""
        _run_workers(_ddp_determinism_impl)


@requires_fsdp2
class TestFSDP2Determinism:

    def test_run_to_run_determinism(self):
        """Two FSDP2 runs with same seed produce identical params,
        even when global CUDA RNG is polluted on the second run."""
        _run_workers(_fsdp2_determinism_impl)


@requires_multi_gpu
class TestFSDP1:

    def test_basic_convergence(self):
        """FSDP1 + LUMA converges on a fixed regression task."""
        _run_workers(_fsdp1_convergence_impl)


# ── FSDP1 worker implementation ──────────────────────────────────────────────

def _fsdp1_convergence_impl(rank, world_size):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    model = SimpleMLP().cuda()
    model = FSDP(model)

    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.0)

    torch.cuda.manual_seed(0)
    x = torch.randn(32, 64, device="cuda")
    target = torch.randn(32, 1, device="cuda")

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    init_avg = sum(losses[:5]) / 5
    final_avg = sum(losses[-5:]) / 5
    assert final_avg < init_avg * 0.5, (
        f"[Rank {rank}] FSDP1 + LUMA didn't converge: "
        f"first_5_avg={init_avg:.4f}, last_5_avg={final_avg:.4f}"
    )


# ── DDP determinism worker ────────────────────────────────────────────────────

def _ddp_determinism_impl(rank, world_size):
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = SimpleMLP().cuda()
    model = DDP(model, device_ids=[rank])
    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.01)

    n_steps = 20
    for step in range(n_steps):
        # Per-rank different data (realistic distributed scenario)
        torch.cuda.manual_seed(step * world_size + rank)
        x = torch.randn(16, 64, device="cuda")
        target = torch.randn(16, 1, device="cuda")

        opt.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()

        # Pollute global CUDA RNG (simulates dropout / augmentation)
        torch.rand(rank * 200 + step * 13 + 1, device="cuda")
        opt.step()

    # All ranks must have bitwise-identical params
    for name, p in model.named_parameters():
        gathered = [torch.zeros_like(p.data) for _ in range(world_size)]
        dist.all_gather(gathered, p.data.contiguous())

        if rank == 0:
            for r in range(1, world_size):
                assert torch.equal(gathered[0], gathered[r]), (
                    f"DDP param '{name}' diverged: rank 0 vs rank {r}, "
                    f"max_diff={(gathered[0] - gathered[r]).abs().max():.2e}"
                )


# ── FSDP2 determinism worker ─────────────────────────────────────────────────

def _fsdp2_determinism_impl(rank, world_size):
    from torch.distributed._composable.fsdp import fully_shard

    results = []
    for run_idx in range(2):
        seed_all(42)
        torch.cuda.manual_seed(42)

        model = SimpleMLP().cuda()
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)

        opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.01)

        torch.cuda.manual_seed(0)
        x = torch.randn(16, 64, device="cuda")
        target = torch.randn(16, 1, device="cuda")

        for step in range(20):
            opt.zero_grad()
            nn.functional.mse_loss(model(x), target).backward()
            # Pollute global CUDA RNG on the second run only
            if run_idx == 1:
                torch.rand(rank * 100 + step + 1, device="cuda")
            opt.step()

        results.append([p.data.clone() for p in model.parameters()])

    for i, (p1, p2) in enumerate(zip(results[0], results[1])):
        assert torch.equal(p1, p2), (
            f"[Rank {rank}] FSDP2 param {i} not deterministic across runs: "
            f"max_diff={(p1 - p2).abs().max():.2e}"
        )
