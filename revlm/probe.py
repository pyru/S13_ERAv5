"""Find the largest batch size that trains without CUDA OOM, for one mode.

    python -m revlm.probe --mode revnet --out runs/probe_revnet.json

Every attempt runs in a fresh subprocess (a caught OOM can leave the allocator fragmented
and make the next attempt fail for the wrong reason). An attempt is 3 full training steps
(forward, backward, AdamW) at the real model size with random tokens. The search doubles
until OOM, then bisects to a multiple of `--granularity`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from .model import GPT, Config
from .train import amp_setup


def one(mode: str, B: int, T: int, vocab: int) -> None:
    dev = "cuda"
    amp, scaler, _ = amp_setup(dev)
    model = GPT(Config(vocab_size=vocab, block_size=T, mode=mode)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = None
        for i in range(3):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            x = torch.randint(0, vocab, (B, T + 1), device=dev)
            with amp:
                loss = model(x[:, :-1], x[:, 1:])
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 2
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"ok": False}))
        sys.exit(3)
    print(json.dumps({"ok": True, "peak": torch.cuda.max_memory_allocated(),
                      "tok_per_s": B * T / dt}))


def attempt(mode, B, T, vocab):
    r = subprocess.run([sys.executable, "-m", "revlm.probe", "--one", "--mode", mode,
                        "--batch_size", str(B), "--seq", str(T), "--vocab", str(vocab)],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    last = (r.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        res = json.loads(last)
    except json.JSONDecodeError:
        res = {"ok": False, "error": (r.stderr or r.stdout)[-400:]}
    status = "ok " if res.get("ok") else "OOM"
    extra = f"peak {res['peak'] / 2**30:6.2f} GiB  {res['tok_per_s']:9,.0f} tok/s" if res.get("ok") else ""
    print(f"[probe] {mode:9s} B={B:5d}  {status} {extra}", flush=True)
    return res


def search(mode, T, vocab, start, cap, gran):
    tried = {}
    lo, hi, B = 0, None, start
    while B <= cap:
        tried[B] = attempt(mode, B, T, vocab)
        if not tried[B].get("ok"):
            hi = B
            break
        lo, B = B, B * 2
    if hi is None:
        hi = cap + gran
    while hi - lo > gran:
        mid = (lo + hi) // 2 // gran * gran
        if mid <= lo:
            break
        tried[mid] = attempt(mode, mid, T, vocab)
        lo, hi = (mid, hi) if tried[mid].get("ok") else (lo, mid)
    best = tried.get(lo, {})
    return dict(mode=mode, max_batch=lo, seq=T, peak_at_max=best.get("peak"),
                tok_per_s_at_max=best.get("tok_per_s"),
                gpu_total=torch.cuda.get_device_properties(0).total_memory,
                tried={str(k): v for k, v in sorted(tried.items())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--granularity", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.one:
        one(a.mode, a.batch_size, a.seq, a.vocab)
    else:
        res = search(a.mode, a.seq, a.vocab, a.start, a.cap, a.granularity)
        print(f"[probe] {a.mode}: max batch {res['max_batch']}", flush=True)
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
