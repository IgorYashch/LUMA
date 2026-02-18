#!/usr/bin/env python3
"""Benchmark: LUMA Triton vs AdamW — eager, compiled, fused.

Measures optimizer step time, full train step time, peak GPU memory,
and optimizer state footprint across a range of model sizes.

Usage:
    python benchmarks/bench_compile.py
    python benchmarks/bench_compile.py --sizes 1024 4096 16384 --steps 100
    python benchmarks/bench_compile.py --json results.json

Requires: CUDA GPU + triton
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from luma_optimizer import LUMA


# =====================================================================
#  Configuration
# =====================================================================

@dataclass
class BenchResult:
    label: str
    n_params: int
    step_ms: float          # median opt.step() time
    train_step_ms: float    # median full train step (fwd+bwd+opt)
    peak_mem_mb: float      # peak GPU memory during training
    state_mem_mb: float     # optimizer state footprint
    throughput_Mparam_s: float = 0.0  # M params/sec for full train step

    def __post_init__(self):
        if self.train_step_ms > 0:
            self.throughput_Mparam_s = (
                self.n_params / self.train_step_ms / 1e3  # M params/sec
            )


# =====================================================================
#  Helpers
# =====================================================================

def _state_bytes(opt) -> int:
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.element_size() * v.numel()
    return total


def _make_model(size: int, device: str = "cuda") -> nn.Module:
    return nn.Sequential(nn.Linear(size, size, bias=False)).to(device)


def _make_optimizer(name: str, model: nn.Module, lr: float = 1e-3):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if name == "adamw_fused":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, fused=True,
        )
    if name == "luma_triton":
        return LUMA(
            model.parameters(), lr=lr, weight_decay=0.01, backend="triton",
        )
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def _measure_step_time(
    model: nn.Module,
    opt,
    size: int,
    n_warmup: int,
    n_steps: int,
) -> float:
    """Measure median opt.step() time in ms using CUDA events."""
    device = next(model.parameters()).device
    # pre-generate grads
    x = torch.randn(4, size, device=device)
    loss = model(x).sum()
    loss.backward()

    # warmup
    for _ in range(n_warmup):
        opt.step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, size, device=device)
        model(x).sum().backward()

    # timed runs
    times = []
    for _ in range(n_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, size, device=device)
        model(x).sum().backward()

    times.sort()
    return times[len(times) // 2]  # median


def _measure_train_step(
    model: nn.Module,
    opt,
    size: int,
    n_warmup: int,
    n_steps: int,
    compiled_model: nn.Module | None = None,
) -> tuple[float, float]:
    """Measure median full train step (fwd+bwd+opt) and peak memory.

    Returns (median_ms, peak_mem_mb).
    """
    fwd_model = compiled_model if compiled_model is not None else model
    device = next(model.parameters()).device

    # warmup (includes compile overhead)
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, size, device=device)
        loss = fwd_model(x).sum()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)

    # timed runs
    times = []
    for _ in range(n_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, size, device=device)
        start.record()
        loss = fwd_model(x).sum()
        loss.backward()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    times.sort()
    return times[len(times) // 2], peak_mb


# =====================================================================
#  Benchmark runner
# =====================================================================

CONFIGS = [
    ("AdamW", "adamw", False),
    ("AdamW compiled", "adamw", True),
    ("AdamW fused", "adamw_fused", False),
    ("LUMA Triton", "luma_triton", False),
    ("LUMA Triton compiled", "luma_triton", True),
]


def run_one(
    label: str,
    opt_name: str,
    compile_model: bool,
    size: int,
    n_warmup: int,
    n_steps: int,
) -> BenchResult:
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = _make_model(size)
    opt = _make_optimizer(opt_name, model)
    n_params = sum(p.numel() for p in model.parameters())

    compiled = torch.compile(model) if compile_model else None

    # opt.step() time
    step_ms = _measure_step_time(model, opt, size, n_warmup, n_steps)

    # full train step + peak memory (fresh model to reset memory)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model2 = _make_model(size)
    opt2 = _make_optimizer(opt_name, model2)
    compiled2 = torch.compile(model2) if compile_model else None
    train_ms, peak_mb = _measure_train_step(
        model2, opt2, size, n_warmup, n_steps, compiled2,
    )

    state_mb = _state_bytes(opt2) / 1024**2

    return BenchResult(
        label=label,
        n_params=n_params,
        step_ms=step_ms,
        train_step_ms=train_ms,
        peak_mem_mb=peak_mb,
        state_mem_mb=state_mb,
    )


# =====================================================================
#  Output
# =====================================================================

SEP = "-" * 112
HEADER = (
    f"  {'Optimizer':<24} {'Params':>10} {'opt.step':>10} "
    f"{'train step':>11} {'throughput':>12} {'peak mem':>10} {'state':>10}"
)
UNITS = (
    f"  {'':<24} {'':>10} {'(ms)':>10} "
    f"{'(ms)':>11} {'(M p/s)':>12} {'(MB)':>10} {'(MB)':>10}"
)


def print_row(r: BenchResult):
    print(
        f"  {r.label:<24} {r.n_params:>10,} {r.step_ms:>10.3f} "
        f"{r.train_step_ms:>11.3f} {r.throughput_Mparam_s:>12.1f} "
        f"{r.peak_mem_mb:>10.1f} {r.state_mem_mb:>10.2f}"
    )


def print_summary(results: list[BenchResult], size: int):
    adamw = next((r for r in results if r.label == "AdamW"), None)
    luma_compiled = next(
        (r for r in results if r.label == "LUMA Triton compiled"), None,
    )
    if adamw and luma_compiled:
        speedup = adamw.train_step_ms / luma_compiled.train_step_ms
        mem_ratio = adamw.state_mem_mb / luma_compiled.state_mem_mb
        print(f"\n  LUMA compiled vs AdamW eager:")
        print(f"    train step speedup:   {speedup:.2f}x")
        print(f"    state memory saving:  {mem_ratio:.1f}x ({luma_compiled.state_mem_mb:.2f} MB vs {adamw.state_mem_mb:.2f} MB)")

    adamw_compiled = next(
        (r for r in results if r.label == "AdamW compiled"), None,
    )
    if adamw_compiled and luma_compiled:
        speedup = adamw_compiled.train_step_ms / luma_compiled.train_step_ms
        print(f"  LUMA compiled vs AdamW compiled:")
        print(f"    train step speedup:   {speedup:.2f}x")


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LUMA Triton vs AdamW (eager / compiled / fused)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 4096, 8192],
        help="Hidden dimensions to benchmark (params = size^2)",
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Timed iterations (default: 50)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {args.warmup}  |  Timed steps: {args.steps}")

    all_results: list[dict] = []

    for size in args.sizes:
        n_params = size * size
        print(f"\n{'=' * 112}")
        print(f"  Linear({size}, {size})  —  {n_params:,} params  ({n_params * 4 / 1024**2:.1f} MB fp32)")
        print(f"{'=' * 112}")
        print(HEADER)
        print(UNITS)
        print(f"  {SEP}")

        results = []
        for label, opt_name, do_compile in CONFIGS:
            try:
                r = run_one(
                    label, opt_name, do_compile, size,
                    args.warmup, args.steps,
                )
                results.append(r)
                print_row(r)
            except Exception as e:
                print(f"  {label:<24} SKIPPED: {e}")

        print_summary(results, size)
        all_results.extend(asdict(r) for r in results)

    if args.json:
        meta = {
            "gpu": gpu_name,
            "pytorch": torch.__version__,
            "warmup": args.warmup,
            "steps": args.steps,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {args.json}")

    print()


if __name__ == "__main__":
    main()
