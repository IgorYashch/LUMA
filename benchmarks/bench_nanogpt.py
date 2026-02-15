#!/usr/bin/env python3
"""nanoGPT on Shakespeare — LUMA vs AdamW loss trajectory comparison.

Self-contained: downloads tiny_shakespeare, builds a minimal char-level GPT,
trains both optimisers **simultaneously on the same batch every step**,
then prints a side-by-side loss table.

Features:
  - identical init (same seed + state_dict copy + assertion)
  - identical data (interleaved training, one batch → both models)
  - torch.compile for both models
"""

from __future__ import annotations

import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from luma_optimizer import LUMA

# ── hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE = 32
BLOCK_SIZE = 64
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0          # deterministic — fair comparison
LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
STEPS = 1000
EVAL_EVERY = 50
EVAL_ITERS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE = True          # torch.compile both models
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")


# ── data ─────────────────────────────────────────────────────────────────────

def get_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading tiny_shakespeare → {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size


def get_batch(split_data, device):
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([split_data[i : i + BLOCK_SIZE] for i in ix]).to(device)
    y = torch.stack([split_data[i + 1 : i + BLOCK_SIZE + 1] for i in ix]).to(device)
    return x, y


# ── model ────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok + pos))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss_pair(model_a, model_h, train_data, val_data, device):
    """Evaluate BOTH models on the SAME random batches."""
    model_a.eval()
    model_h.eval()
    out_a, out_h = {}, {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses_a, losses_h = [], []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(data, device)
            _, la = model_a(xb, yb)
            _, lh = model_h(xb, yb)
            losses_a.append(la.item())
            losses_h.append(lh.item())
        out_a[name] = sum(losses_a) / len(losses_a)
        out_h[name] = sum(losses_h) / len(losses_h)
    model_a.train()
    model_h.train()
    return out_a, out_h


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    train_data, val_data, vocab_size = get_data()
    print(f"Vocab: {vocab_size} chars  |  Train: {len(train_data):,}  |  Val: {len(val_data):,}")
    print(f"Device: {DEVICE}  |  Compile: {COMPILE}")

    # ── identical init ───────────────────────────────────────────────
    torch.manual_seed(1337)
    model_adam = NanoGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE).to(DEVICE)
    model_luma = NanoGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE).to(DEVICE)
    model_luma.load_state_dict(model_adam.state_dict())

    # sanity check: weights are bit-exact
    for (na, pa), (nh, ph) in zip(
        model_adam.named_parameters(), model_luma.named_parameters()
    ):
        assert torch.equal(pa.data, ph.data), f"Init mismatch in {na}"
    n_params = model_adam.count_params()
    print(f"Model: {N_LAYER}L / {N_HEAD}H / {N_EMBD}D  —  {n_params:,} params")
    print(f"Init weights verified bit-exact ✓")

    # ── optimisers (create BEFORE compile) ────────────────────────────
    opt_adam = torch.optim.AdamW(
        model_adam.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    opt_luma = LUMA(
        model_luma.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )

    # ── torch.compile ────────────────────────────────────────────────
    if COMPILE:
        print("Compiling models (first step will be slow)...")
        model_adam = torch.compile(model_adam)
        model_luma = torch.compile(model_luma)

    # ── interleaved training (SAME batch → BOTH models) ──────────────
    torch.manual_seed(42)
    tr_adam: dict[int, float] = {}
    tr_luma: dict[int, float] = {}
    val_adam: dict[int, float] = {}
    val_luma: dict[int, float] = {}

    print(f"\n{'─' * 80}")
    t0 = time.perf_counter()

    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_data, DEVICE)       # ONE batch for BOTH

        # ── AdamW step ───────────────────────────────────────────────
        _, loss_a = model_adam(xb, yb)
        opt_adam.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(model_adam.parameters(), 1.0)
        opt_adam.step()

        # ── LUMA step ────────────────────────────────────────────────
        _, loss_h = model_luma(xb, yb)
        opt_luma.zero_grad(set_to_none=True)
        loss_h.backward()
        torch.nn.utils.clip_grad_norm_(model_luma.parameters(), 1.0)
        opt_luma.step()

        # ── eval ─────────────────────────────────────────────────────
        if step % EVAL_EVERY == 0 or step == 1:
            ev_a, ev_h = estimate_loss_pair(
                model_adam, model_luma, train_data, val_data, DEVICE,
            )
            tr_adam[step] = ev_a["train"]
            tr_luma[step] = ev_h["train"]
            val_adam[step] = ev_a["val"]
            val_luma[step] = ev_h["val"]

            elapsed = time.perf_counter() - t0
            ms = elapsed / step * 1000
            rd_t = abs(ev_h["train"] - ev_a["train"]) / max(ev_a["train"], 1e-8)
            rd_v = abs(ev_h["val"] - ev_a["val"]) / max(ev_a["val"], 1e-8)
            print(
                f"  step {step:>5}/{STEPS}  "
                f"AdamW {ev_a['train']:.4f}/{ev_a['val']:.4f}  "
                f"LUMA {ev_h['train']:.4f}/{ev_h['val']:.4f}  "
                f"Δtrain {rd_t:.3%}  Δval {rd_v:.3%}  "
                f"({ms:.0f} ms/step)"
            )

    # ── comparison table ─────────────────────────────────────────────
    common = sorted(set(tr_adam) & set(tr_luma))
    print(f"\n{'=' * 80}")
    print("  nanoGPT Shakespeare — LUMA vs AdamW  (same init, same data, torch.compile)")
    print(f"{'=' * 80}")
    print(
        f"  {'step':>5}  "
        f"{'AdamW train':>12}  {'LUMA train':>12}  {'Δtrain':>8}  "
        f"{'AdamW val':>10}  {'LUMA val':>10}  {'Δval':>8}"
    )
    print("  " + "─" * 74)
    for s in common:
        ta, th = tr_adam[s], tr_luma[s]
        va, vh = val_adam[s], val_luma[s]
        rd_t = abs(th - ta) / max(ta, 1e-8)
        rd_v = abs(vh - va) / max(va, 1e-8)
        print(
            f"  {s:>5}  {ta:>12.4f}  {th:>12.4f}  {rd_t:>8.3%}  "
            f"{va:>10.4f}  {vh:>10.4f}  {rd_v:>8.3%}"
        )

    last = common[-1]
    print(f"\n  Final train: AdamW={tr_adam[last]:.4f}  LUMA={tr_luma[last]:.4f}  "
          f"Δ={abs(tr_luma[last]-tr_adam[last])/tr_adam[last]:.4%}")
    print(f"  Final val:   AdamW={val_adam[last]:.4f}  LUMA={val_luma[last]:.4f}  "
          f"Δ={abs(val_luma[last]-val_adam[last])/val_adam[last]:.4%}")

    total = time.perf_counter() - t0
    print(f"\n  Total time: {total:.1f}s  ({total/STEPS*1000:.0f} ms/step)")
    print("=" * 80)


if __name__ == "__main__":
    main()
