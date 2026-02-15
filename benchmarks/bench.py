#!/usr/bin/env python3
"""Benchmarks: wall-clock step time and memory footprint for LUMA vs AdamW."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from luma_optimizer import LUMA


# ── helpers ──────────────────────────────────────────────────────────────────


def _warmup(model: nn.Module, opt, device: torch.device, n: int = 5) -> None:
    d_in = model[0].in_features  # type: ignore[union-attr]
    for _ in range(n):
        x = torch.randn(4, d_in, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed_steps(
    model: nn.Module, opt, device: torch.device, n_steps: int = 50
) -> float:
    """Return mean wall-clock time per step (seconds)."""
    d_in = model[0].in_features  # type: ignore[union-attr]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(4, d_in, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_steps


def _state_bytes(opt) -> int:
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.element_size() * v.numel()
    return total


# ── main ─────────────────────────────────────────────────────────────────────


def _run_benchmark(device: torch.device) -> None:
    tag = device.type.upper()
    sizes = [256, 1024, 4096]
    if device.type == "cuda":
        sizes.append(16384)
    n_steps = 50

    print(f"\n{'=' * 74}")
    print(f"  LUMA  vs  AdamW  —  Benchmark  ({tag})")
    print(f"{'=' * 74}")

    for size in sizes:
        n_params = size * size

        # ── AdamW ──
        m_a = nn.Sequential(nn.Linear(size, size, bias=False)).to(device)
        o_a = torch.optim.AdamW(m_a.parameters(), lr=1e-3, weight_decay=0.01)
        _warmup(m_a, o_a, device)
        t_a = _timed_steps(m_a, o_a, device, n_steps)
        mem_a = _state_bytes(o_a)

        # ── LUMA ──
        m_h = nn.Sequential(nn.Linear(size, size, bias=False)).to(device)
        o_h = LUMA(m_h.parameters(), lr=1e-3, weight_decay=0.01)
        _warmup(m_h, o_h, device)
        t_h = _timed_steps(m_h, o_h, device, n_steps)
        mem_h = _state_bytes(o_h)

        print(f"\n  Model: Linear({size}, {size})  —  {n_params:,} params")
        print(
            f"  AdamW   {t_a*1e3:8.2f} ms/step  |  "
            f"State {mem_a/1024:8.1f} KB  |  "
            f"{mem_a / n_params:.1f} B/param"
        )
        print(
            f"  LUMA    {t_h*1e3:8.2f} ms/step  |  "
            f"State {mem_h/1024:8.1f} KB  |  "
            f"{mem_h / n_params:.1f} B/param"
        )
        ratio_t = t_a / t_h if t_h > 0 else float("inf")
        ratio_m = mem_a / mem_h if mem_h > 0 else float("inf")
        print(f"  → time ratio (AdamW/LUMA) = {ratio_t:.2f}x")
        print(f"  → mem  ratio (AdamW/LUMA) = {ratio_m:.2f}x")


def main() -> None:
    _run_benchmark(torch.device("cpu"))
    if torch.cuda.is_available():
        _run_benchmark(torch.device("cuda"))
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
