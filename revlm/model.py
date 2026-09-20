"""A 20.0M-parameter GPT whose residual trunk can be swapped between six integrators.

    standard  x_{n+1} = x_n + f(x_n), plain autograd (stores everything)
    ckpt      same forward, activation checkpointing per block (stores one tensor per layer)
    euler     same forward, backward inverts each block by fixed-point iteration (approx.)
    revnet    two-stream additive coupling, exactly invertible
    midpoint  x_{n+1} = x_{n-1} + 2h f(x_n), exactly invertible
    leapfrog  x_{n+1} = 2x_n - x_{n-1} + h^2 f(x_n), exactly invertible

All six have the same parameters (same blocks, same shapes), so parameter count, optimizer
memory and FLOPs per forward are identical and only activation memory and backward cost
differ.

The LM head + cross-entropy is computed in checkpointed row chunks in every mode. At large
batch the (B*T, vocab) logits would otherwise be the biggest tensor in the program and
would hide exactly the activation savings this repo is trying to measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .reversible import RevStack, make_variant

MODES = ("standard", "ckpt", "euler", "revnet", "midpoint", "leapfrog")
REVERSIBLE = ("euler", "revnet", "midpoint", "leapfrog")


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 14
    n_head: int = 5
    n_embd: int = 320
    mode: str = "standard"
    euler_iters: int = 6
    ce_chunk: int = 8192  # rows of B*T per checkpointed LM-head/CE chunk


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def F(self, x):
        return self.attn(self.ln1(x))

    def G(self, x):
        return self.mlp(self.ln2(x))

    def f(self, x):
        a = self.F(x)
        return a + self.G(x + a)


def _euler_step(blk: Block, x):
    return x + blk.f(x)


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.mode in MODES, cfg.mode
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.variant = make_variant(cfg.mode, cfg.euler_iters)
        # False = run the same integrator under plain autograd. Used only as the exact
        # reference when checking the reversible gradients.
        self.reversible_backward = True
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # residual writes, GPT-2 scaling
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk(self, x):
        mode, var = self.cfg.mode, self.variant
        if mode == "ckpt" and torch.is_grad_enabled():
            for blk in self.blocks:
                x = checkpoint(_euler_step, blk, x, use_reentrant=False)
            return x
        state = var.init(x)
        if mode in REVERSIBLE and self.reversible_backward and torch.is_grad_enabled():
            state = RevStack.apply(self.blocks, var, *state)
        else:
            for blk in self.blocks:
                state = var.step(blk, state)
        return var.readout(state)

    def hidden(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        return self.trunk(self.wte(idx) + self.wpe(pos))

    def logits(self, h):
        return F.linear(self.ln_f(h), self.wte.weight)  # tied head

    def _ce_sum(self, h, t):
        logits = self.logits(h)
        # at least fp32 (fp16/bf16 logits are upcast), but never demote fp64
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        return F.cross_entropy(logits, t, reduction="sum")

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:
            return self.logits(h)
        h = h.reshape(-1, h.shape[-1])
        t = targets.reshape(-1)
        n, chunk = t.numel(), self.cfg.ce_chunk or t.numel()
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, chunk):
            hc, tc = h[i:i + chunk], t[i:i + chunk]
            if torch.is_grad_enabled():
                total = total + checkpoint(self._ce_sum, hc, tc, use_reentrant=False)
            else:
                total = total + self._ce_sum(hc, tc)
        return total / n

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 0.8, top_k: int = 40):
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1, :].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
