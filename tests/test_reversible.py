"""Correctness of the reversible backward, checked in float64 on CPU.

For each exactly-invertible variant, the gradients from the reconstruct-in-backward path
must equal the gradients of the same network run under plain autograd, to float64
round-off. The fixed-point Euler variant must approach them as iterations grow.

    python tests/test_reversible.py       (or: pytest tests)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revlm.model import GPT, Config  # noqa: E402

TINY = dict(vocab_size=64, block_size=16, n_layer=6, n_head=4, n_embd=32, ce_chunk=24)


def _grads(model, idx, tgt):
    model.zero_grad(set_to_none=True)
    loss = model(idx, tgt)
    loss.backward()
    return loss.detach(), torch.cat([p.grad.flatten() for p in model.parameters()])


def _compare(mode, dtype=torch.float64, scale=8.0, **kw):
    torch.manual_seed(0)
    model = GPT(Config(mode=mode, **TINY, **kw)).to(dtype)
    # random init is too gentle to exercise anything; make the blocks do real work
    with torch.no_grad():
        for p in model.blocks.parameters():
            p.mul_(scale) if p.dim() > 1 else None
    idx = torch.randint(0, 64, (3, 16))
    tgt = torch.randint(0, 64, (3, 16))
    l_rev, g_rev = _grads(model, idx, tgt)
    model.reversible_backward = False
    l_ref, g_ref = _grads(model, idx, tgt)
    rel = ((g_rev - g_ref).norm() / g_ref.norm()).item()
    return (l_rev - l_ref).abs().item(), rel


def test_exact_variants_match_autograd():
    for mode in ("revnet", "midpoint", "leapfrog"):
        dl, rel = _compare(mode)
        print(f"{mode:9s} |loss diff| {dl:.1e}   grad rel err {rel:.1e}")
        assert dl == 0.0 and rel < 1e-10, (mode, dl, rel)


def test_euler_fixed_point_converges_only_for_contractions():
    # at init scale each block is a contraction and the inversion converges geometrically
    errs = []
    for it in (1, 3, 10, 40):
        _, rel = _compare("euler", scale=1.0, euler_iters=it)
        errs.append(rel)
        print(f"euler init-scale  iters={it:2d}   grad rel err {rel:.1e}")
    assert errs[-1] < errs[0] and errs[-1] < 1e-10, errs
    # with blocks that are not contractions, more iterations do not help at all
    _, rel = _compare("euler", scale=8.0, euler_iters=40)
    print(f"euler weights x8  iters=40   grad rel err {rel:.1e}  (expected: wrong)")
    assert rel > 1e-2


def test_ckpt_and_chunked_ce_are_exact():
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (3, 16))
    tgt = torch.randint(0, 64, (3, 16))
    out = {}
    for mode, chunk in (("standard", 0), ("standard", 24), ("ckpt", 24)):
        torch.manual_seed(1)
        m = GPT(Config(mode=mode, **{**TINY, "ce_chunk": chunk})).double()
        out[(mode, chunk)] = _grads(m, idx, tgt)
    ref_l, ref_g = out[("standard", 0)]
    for k, (l, g) in out.items():
        rel = ((g - ref_g).norm() / ref_g.norm()).item()
        print(f"{k!s:18s} |loss diff| {(l - ref_l).abs().item():.1e}  grad rel err {rel:.1e}")
        assert (l - ref_l).abs() < 1e-12 and rel < 1e-12


def test_forward_matches_plain_loop():
    # the no-grad eval path and the RevStack path compute the same loss
    for mode in ("revnet", "midpoint", "leapfrog", "euler"):
        torch.manual_seed(0)
        m = GPT(Config(mode=mode, **TINY)).double()
        idx = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            a = m(idx, idx)
        b = m(idx, idx)
        assert (a - b).abs() < 1e-12, mode


if __name__ == "__main__":
    test_exact_variants_match_autograd()
    test_euler_fixed_point_converges_only_for_contractions()
    test_ckpt_and_chunked_ce_are_exact()
    test_forward_matches_plain_loop()
    print("all reversible tests passed")
