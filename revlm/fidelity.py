"""How close are the reversible gradients to the true ones, in the real training precision?

    python -m revlm.fidelity --ckpt_dir runs/ckpt --out runs/fidelity.json

The fp64 unit tests prove the algebra. This measures what actually reaches the optimizer
on the GPU: bf16/fp16 autocast, fused attention, fp32 residual stream, and weights that
have been *trained* (a trained block can be far less invertible than a freshly
initialised one). For each mode the same batch is backpropagated twice, once through the
reconstruct-in-backward path and once under plain autograd, and the two gradient vectors
are compared.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import Batches
from .model import GPT, REVERSIBLE, Config
from .train import amp_setup


def flat_grad(model, x, y, amp, scale):
    model.zero_grad(set_to_none=True)
    with amp:
        loss = model(x, y)
    (loss * scale).backward()
    return torch.cat([p.grad.double().flatten() for p in model.parameters()]) / scale


def measure(model, x, y, amp, scale):
    model.reversible_backward = True
    g_rev = flat_grad(model, x, y, amp, scale)
    model.reversible_backward = False
    g_ref = flat_grad(model, x, y, amp, scale)
    model.reversible_backward = True
    # float64: a float32 dot product over 20M entries drifts past 1.0
    return dict(cos=float(g_rev @ g_ref / (g_rev.norm() * g_ref.norm())),
                rel_err=float((g_rev - g_ref).norm() / g_ref.norm()),
                finite=bool(torch.isfinite(g_rev).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt_dir", default="runs/ckpt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp, scaler, amp_name = amp_setup(dev)
    scale = 1024.0 if amp_name == "fp16" else 1.0  # keep fp16 grads out of underflow
    x, y = Batches(os.path.join(a.data, "val.bin"), a.batch_size, a.seq, 0, dev).next()
    vocab = json.load(open(os.path.join(a.data, "meta.json")))["vocab_size"]
    out = {"amp": amp_name, "batch_size": a.batch_size, "results": {}}
    for mode in REVERSIBLE:
        r = {}
        torch.manual_seed(0)
        m = GPT(Config(vocab_size=vocab, block_size=a.seq, mode=mode)).to(dev)
        r["init"] = measure(m, x, y, amp, scale)
        ck = os.path.join(a.ckpt_dir, f"{mode}.pt")
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=dev)["model"])
            r["trained"] = measure(m, x, y, amp, scale)
        out["results"][mode] = r
        print(f"[fidelity] {mode:9s} {r}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
