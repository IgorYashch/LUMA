# LUMA — Log-space Unbiased Momentum AdamW

Memory-efficient drop-in replacement for AdamW that compresses optimizer states to **4 bytes/param** (vs 8 bytes/param in AdamW) via preconditioner-domain quantization with unbiased log-to-linear stochastic rounding.

## Quick start

```python
from luma_optimizer import LUMA

opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)
```

## Install

```bash
pip install -e .

# with Triton support (CUDA only)
pip install -e ".[triton]"
```

## How it works

LUMA stores two `int16` tensors per parameter instead of two `float32` tensors:

| State | AdamW | LUMA |
|---|---|---|
| First moment (m) | float32 | int16 (signed, log-grid) |
| Second moment (v) | float32 | int16 (unsigned, log-grid) |
| **Total** | **8 B/param** | **4 B/param** |

Key ideas:
- **Log-space quantization grid** — maps continuous EMA values onto integer bins via `log1p`/`expm1`
- **Unbiased stochastic rounding (LogSR)** — corrects Jensen's inequality bias so the expected linear-domain reconstruction is exact
- **Two-pass matched scaling** — new scales are computed first, then states are quantized on the matching grid, eliminating encode-decode drift

## Backends

| Backend | Device | Selection |
|---|---|---|
| PyTorch (reference) | CPU / CUDA | `backend="pytorch"` |
| Triton (fused, zero temp alloc) | CUDA | `backend="triton"` |
| Auto | best available | `backend="auto"` (default) |

## Distributed training

LUMA is a standard `torch.optim.Optimizer` and works out of the box with common distributed setups:

| Setup | Compatible | Notes |
|---|---|---|
| **DDP** (`DistributedDataParallel`) | Yes | Seed `torch` and CUDA RNGs identically on all ranks before creating the optimizer (see *Determinism* below). |
| **FSDP** (`FullyShardedDataParallel`) | Yes | Scales `S_m` / `S_v` are computed per-shard by design. **For checkpointing**, call `opt.export_adamw_state()` to materialise full FP32 states before saving — `FSDP.optim_state_dict()` cannot correctly gather per-shard scalar scales across ranks. |
| **FSDP2** (DTensor-based) | Yes | DTensor parameters are transparently unwrapped to local shards via `_local_tensor`. No special configuration needed. |
| **GradScaler** (AMP) | Yes | LUMA internally promotes gradients to float32, so fp16/bf16 inputs are handled correctly. Use `GradScaler` with `unscale_()` before `step()` as normal. |
| **DeepSpeed ZeRO** | No | Will fix it in future |

## Determinism / Reproducibility

LUMA uses **stochastic rounding** at every optimiser step. To get fully
reproducible training runs, seed PyTorch before creating the optimiser:

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

opt = LUMA(model.parameters(), lr=1e-3)
```

`_base_seed` (the root of all quantisation-noise randomness) is drawn from
the PyTorch global RNG at construction time via `torch.randint`. Both backends
then derive a deterministic per-step per-parameter seed from `_base_seed`,
step counter, and parameter index, so stochastic rounding is reproducible
across DDP ranks as long as the PyTorch RNG state matches at optimizer
construction — which it will if `torch.manual_seed()` is called identically
on every rank before creating the model and optimizer.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
