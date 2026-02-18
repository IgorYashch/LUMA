"""Mixed-precision integration tests for LUMA.

Tests cover the standard PyTorch mixed-precision workflows:

- Rejection of non-fp32 parameters (any device)
- ``torch.amp.autocast`` single-device training (CPU bf16, CUDA fp16/bf16)
- ``torch.amp.GradScaler`` fp16 training (CUDA)
- FSDP2 ``MixedPrecisionPolicy`` distributed training (multi-GPU)
"""

from functools import partial

import pytest
import torch
import torch.nn as nn

from luma_optimizer import LUMA
from utils import SimpleMLP, requires_cuda, requires_fsdp2_mp, run_workers


# =====================================================================
#  Rejection of non-fp32 parameters
# =====================================================================


class TestRejectsNonFP32:
    """LUMA must reject parameters that are not float32."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rejects_reduced_precision_params(self, dtype, device):
        """model.to(fp16/bf16) + step() → ValueError."""
        model = nn.Linear(16, 8).to(dtype).to(device)
        opt = LUMA(model.parameters())
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        with pytest.raises(ValueError, match="float32"):
            opt.step()


# =====================================================================
#  Autocast (single device)
# =====================================================================


class TestAutocast:
    """Verify LUMA works correctly under ``torch.amp.autocast``."""

    def test_single_step(self, autocast_cfg):
        """Autocast forward+backward → step: params stay fp32 and finite."""
        device_type, amp_dtype = autocast_cfg
        model = nn.Linear(16, 8).to(device_type)
        opt = LUMA(model.parameters())
        x = torch.randn(2, 16, device=device_type)

        with torch.autocast(device_type, dtype=amp_dtype):
            loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            assert p.dtype == torch.float32
            assert p.isfinite().all()
            s = opt.state[p]
            assert s["Q_m"].dtype == torch.int16
            assert s["Q_w"].dtype == torch.int16

    def test_twenty_steps_stable(self, autocast_cfg):
        """Twenty autocast steps produce no NaN/Inf."""
        device_type, amp_dtype = autocast_cfg
        model = nn.Linear(16, 8).to(device_type)
        opt = LUMA(model.parameters(), lr=1e-3)

        for _ in range(20):
            x = torch.randn(2, 16, device=device_type)
            with torch.autocast(device_type, dtype=amp_dtype):
                loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert p.isfinite().all()

    def test_convergence(self, autocast_cfg):
        """MLP regression converges under autocast."""
        device_type, amp_dtype = autocast_cfg
        d = 16
        model = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1),
        ).to(device_type)
        opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)

        X = torch.randn(100, d, device=device_type)
        y = torch.randn(100, 1, device=device_type)

        init_loss = None
        for _ in range(200):
            opt.zero_grad()
            with torch.autocast(device_type, dtype=amp_dtype):
                pred = model(X)
            loss = nn.functional.mse_loss(pred.float(), y)
            if init_loss is None:
                init_loss = loss.item()
            loss.backward()
            opt.step()

        assert loss.item() < init_loss * 0.5, (
            f"autocast convergence failed on {device_type}/{amp_dtype}: "
            f"{init_loss:.4f} -> {loss.item():.4f}"
        )


# =====================================================================
#  GradScaler (CUDA fp16)
# =====================================================================


@requires_cuda
class TestGradScaler:
    """Verify LUMA works with ``autocast`` + ``GradScaler`` (fp16 training)."""

    def test_single_step(self):
        """GradScaler pipeline: scale → backward → step → update."""
        model = nn.Linear(16, 8).cuda()
        opt = LUMA(model.parameters())
        scaler = torch.amp.GradScaler()
        x = torch.randn(2, 16, device="cuda")

        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x).sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        for p in model.parameters():
            assert p.dtype == torch.float32
            assert p.isfinite().all()

    def test_twenty_steps_stable(self):
        """Twenty GradScaler steps produce no NaN/Inf."""
        model = nn.Linear(16, 8).cuda()
        opt = LUMA(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler()

        for _ in range(20):
            opt.zero_grad()
            x = torch.randn(2, 16, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x).sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for p in model.parameters():
            assert p.isfinite().all()

    def test_convergence(self):
        """MLP regression converges with GradScaler + autocast fp16."""
        d = 16
        model = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1),
        ).cuda()
        opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)
        scaler = torch.amp.GradScaler()

        X = torch.randn(100, d, device="cuda")
        y = torch.randn(100, 1, device="cuda")

        init_loss = None
        for _ in range(200):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(X)
            loss = nn.functional.mse_loss(pred.float(), y)
            if init_loss is None:
                init_loss = loss.item()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        assert loss.item() < init_loss * 0.5, (
            f"GradScaler convergence failed: "
            f"{init_loss:.4f} -> {loss.item():.4f}"
        )


# =====================================================================
#  FSDP2 + MixedPrecisionPolicy (multi-GPU)
# =====================================================================

_MP_POLICIES = [
    pytest.param(
        {"param_dtype": torch.bfloat16},
        id="compute-bf16",
    ),
    pytest.param(
        {"param_dtype": torch.float16},
        id="compute-fp16",
    ),
    pytest.param(
        {"param_dtype": torch.bfloat16, "reduce_dtype": torch.bfloat16},
        id="full-bf16",
    ),
    pytest.param(
        {"param_dtype": torch.bfloat16, "output_dtype": torch.float32},
        id="compute-bf16-output-fp32",
    ),
]


@requires_fsdp2_mp
class TestFSDP2MixedPrecision:
    """FSDP2 + ``MixedPrecisionPolicy`` with various dtype configurations."""

    @pytest.mark.parametrize("mp_kwargs", _MP_POLICIES)
    def test_convergence(self, mp_kwargs):
        """FSDP2 mixed precision + LUMA converges on a regression task."""
        run_workers(partial(_fsdp2_mp_convergence_impl, mp_kwargs=mp_kwargs))

    def test_bf16_state_dict_roundtrip(self):
        """state_dict round-trip under FSDP2 mixed precision (bf16)."""
        run_workers(
            partial(_fsdp2_mp_state_dict_impl,
                    mp_kwargs={"param_dtype": torch.bfloat16}),
        )

    def test_bf16_export_adamw(self):
        """export_adamw_state returns valid fp32 tensors under mixed precision."""
        run_workers(
            partial(_fsdp2_mp_export_adamw_impl,
                    mp_kwargs={"param_dtype": torch.bfloat16}),
        )


# ── FSDP2 MixedPrecision worker implementations ─────────────────────────────

def _make_fsdp2_mp_model(mp_kwargs):
    """Build a SimpleMLP sharded with the given MixedPrecisionPolicy."""
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(**mp_kwargs)
    model = SimpleMLP().cuda()
    fully_shard(model.fc1, mp_policy=mp_policy)
    fully_shard(model.fc2, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model


def _fsdp2_mp_convergence_impl(rank, world_size, mp_kwargs):
    model = _make_fsdp2_mp_model(mp_kwargs)
    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.0)

    torch.cuda.manual_seed(0)
    x = torch.randn(32, 64, device="cuda")
    target = torch.randn(32, 1, device="cuda")

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x).float(), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    init_avg = sum(losses[:5]) / 5
    final_avg = sum(losses[-5:]) / 5
    assert final_avg < init_avg * 0.5, (
        f"[Rank {rank}] FSDP2 MP ({mp_kwargs}) didn't converge: "
        f"first_5_avg={init_avg:.4f}, last_5_avg={final_avg:.4f}"
    )


def _fsdp2_mp_state_dict_impl(rank, world_size, mp_kwargs):
    model = _make_fsdp2_mp_model(mp_kwargs)
    opt = LUMA(model.parameters(), lr=1e-2)

    torch.cuda.manual_seed(0)
    x = torch.randn(16, 64, device="cuda")
    target = torch.randn(16, 1, device="cuda")

    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(x).float(), target).backward()
        opt.step()

    loss_before_save = nn.functional.mse_loss(model(x).float(), target).item()

    sd = opt.state_dict()
    opt2 = LUMA(model.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)

    for _ in range(10):
        opt2.zero_grad()
        nn.functional.mse_loss(model(x).float(), target).backward()
        opt2.step()

    loss_after_load = nn.functional.mse_loss(model(x).float(), target).item()
    assert loss_after_load < loss_before_save, (
        f"[Rank {rank}] FSDP2 MP state_dict round-trip broke convergence: "
        f"before_save={loss_before_save:.6g}, after_load={loss_after_load:.6g}"
    )


def _fsdp2_mp_export_adamw_impl(rank, world_size, mp_kwargs):
    model = _make_fsdp2_mp_model(mp_kwargs)
    opt = LUMA(model.parameters(), lr=1e-2, weight_decay=0.01)

    torch.cuda.manual_seed(0)
    x = torch.randn(16, 64, device="cuda")
    target = torch.randn(16, 1, device="cuda")

    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(x).float(), target).backward()
        opt.step()

    adamw_sd = opt.export_adamw_state()

    assert "state" in adamw_sd
    assert "param_groups" in adamw_sd

    params = list(model.parameters())
    for idx, s in adamw_sd["state"].items():
        local_shape = opt._unwrap_tensor(params[idx]).shape
        assert s["exp_avg"].shape == local_shape
        assert s["exp_avg_sq"].shape == local_shape
        assert s["exp_avg"].dtype == torch.float32
        assert s["exp_avg_sq"].dtype == torch.float32
        assert s["exp_avg"].isfinite().all()
        assert s["exp_avg_sq"].isfinite().all()
        assert s["exp_avg_sq"].min() >= 0
