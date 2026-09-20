"""Train one model for a fixed token budget and write everything measured to a JSON file.

    python -m revlm.train --mode standard --batch_size 64 --tokens 50e6 --out runs/x.json

Speed is train-step wall time only (evaluation excluded, CUDA synchronised at every
boundary). "steady" excludes the first 10 steps (allocator warm-up, kernel autotuning).
Peak memory is torch.cuda.max_memory_allocated over the whole run: parameters, grads,
AdamW state, activations and temporaries. Exit code 3 means CUDA OOM.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .data import Batches
from .model import GPT, REVERSIBLE, Config

WARM_STEPS = 10


def amp_setup(dev: str):
    if dev != "cuda":
        return contextlib.nullcontext(), None, "fp32"
    # T4 reports bf16 "supported" through emulation; only use it on Ampere or newer
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.autocast("cuda", dtype=torch.bfloat16), None, "bf16"
    return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda"), "fp16"


def eval_set(path: str, T: int, n_batches: int, B: int = 32, seed: int = 0):
    b = Batches(path, B, T, seed, "cpu")
    return [b.next() for _ in range(n_batches)]


@torch.no_grad()
def evaluate(model, batches, dev, amp):
    model.eval()
    tot = 0.0
    for x, y in batches:
        with amp:
            tot += model(x.to(dev), y.to(dev)).item()
    model.train()
    return tot / len(batches)


def lr_at(step, steps, lr, warmup, min_frac):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--euler_iters", type=int, default=6)
    ap.add_argument("--n_evals", type=int, default=10)
    ap.add_argument("--eval_batches", type=int, default=12)
    ap.add_argument("--final_eval_batches", type=int, default=120)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_ckpt", default="")
    ap.add_argument("--no_sample", action="store_true")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp, scaler, amp_name = amp_setup(dev)

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    cfg = Config(vocab_size=meta["vocab_size"], block_size=a.seq, mode=a.mode,
                 euler_iters=a.euler_iters)
    model = GPT(cfg).to(dev)
    n_params = model.num_params()
    n_nonemb = n_params - model.wte.weight.numel() - model.wpe.weight.numel()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=(dev == "cuda"))

    B, T = a.batch_size, a.seq
    steps = math.ceil(a.tokens / (B * T))
    warmup = max(10, int(a.warmup_frac * steps))
    eval_every = max(1, steps // a.n_evals)
    train = Batches(os.path.join(a.data, "train.bin"), B, T, a.seed, dev)
    val_path = os.path.join(a.data, "val.bin")
    ev_small = eval_set(val_path, T, a.eval_batches)

    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rec = dict(mode=a.mode, batch_size=B, seq=T, tokens_target=a.tokens, steps=steps,
               lr=a.lr, warmup=warmup, amp=amp_name, n_params=n_params, n_nonemb=n_nonemb,
               euler_iters=a.euler_iters if a.mode == "euler" else None,
               device=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
               curve=[], evals=[], status="running")
    print(f"[train] {a.mode} B={B} T={T} steps={steps} lr={a.lr} amp={amp_name} "
          f"params={n_params:,} on {rec['device']}", flush=True)

    t_train, n_tok, steady_t0, steady_tok0, model_state = 0.0, 0, None, None, None
    loss_acc, loss_n = torch.zeros((), device=dev), 0
    euler_resid = []
    sync()
    t_mark = time.perf_counter()
    try:
        for step in range(steps):
            lr = lr_at(step, steps, a.lr, warmup, a.min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = train.next()
            with amp:
                loss = model(x, y)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if model_state is None and dev == "cuda":
                model_state = torch.cuda.memory_allocated()  # params + grads + AdamW
            opt.zero_grad(set_to_none=True)
            loss_acc += loss.detach()
            loss_n += 1
            n_tok += B * T

            last = step == steps - 1
            if (step + 1) % a.log_every == 0 or last or step + 1 == WARM_STEPS:
                sync()
                now = time.perf_counter()
                t_train += now - t_mark
                t_mark = now
                if step + 1 == WARM_STEPS:
                    steady_t0, steady_tok0 = t_train, n_tok
                tl = (loss_acc / loss_n).item()
                loss_acc.zero_()
                loss_n = 0
                r = model.variant.pop_resid() if a.mode == "euler" else None
                if r is not None:
                    euler_resid.append((step + 1, r))
                rec["curve"].append((step + 1, n_tok, tl, lr))
                if not math.isfinite(tl):
                    rec["status"] = "diverged"
                    print(f"[train] step {step + 1}: loss {tl} -> diverged", flush=True)
                    break
            if (step + 1) % eval_every == 0 or last:
                sync()
                t_train += time.perf_counter() - t_mark
                vl = evaluate(model, ev_small, dev, amp)
                rec["evals"].append((step + 1, n_tok, vl))
                print(f"[train] step {step + 1:5d}/{steps}  tok {n_tok / 1e6:6.1f}M  "
                      f"train {rec['curve'][-1][2]:.4f}  val {vl:.4f}  "
                      f"{n_tok / t_train:,.0f} tok/s", flush=True)
                sync()
                t_mark = time.perf_counter()
    except torch.cuda.OutOfMemoryError:
        rec["status"] = "oom"
        rec["peak_mem_allocated"] = torch.cuda.max_memory_allocated()
        _write(a.out, rec)
        print(f"[train] CUDA OOM at B={B}", flush=True)
        sys.exit(3)

    if rec["status"] == "running":
        rec["status"] = "ok"
    rec["tokens_seen"] = n_tok
    rec["train_time_s"] = t_train
    rec["tok_per_s"] = n_tok / t_train
    if steady_t0 is not None and t_train > steady_t0:
        rec["tok_per_s_steady"] = (n_tok - steady_tok0) / (t_train - steady_t0)
    flops_per_tok = 6 * n_nonemb + 6 * cfg.n_layer * T * cfg.n_embd  # 6N + causal attention
    rec["model_tflops"] = flops_per_tok * rec.get("tok_per_s_steady", rec["tok_per_s"]) / 1e12
    tail = [c[2] for c in rec["curve"][-max(1, len(rec["curve"]) // 20):]]
    rec["final_train_loss"] = float(np.mean(tail))
    if euler_resid:
        rec["euler_resid"] = euler_resid
    if dev == "cuda":
        rec["peak_mem_allocated"] = torch.cuda.max_memory_allocated()
        rec["peak_mem_reserved"] = torch.cuda.max_memory_reserved()
        rec["model_state_mem"] = model_state
    if rec["status"] == "ok":
        rec["final_val_loss"] = evaluate(model, eval_set(val_path, T, a.final_eval_batches), dev, amp)
        if not a.no_sample:
            rec["sample"] = _sample(model, a.data, dev, amp)
    if a.save_ckpt:
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, a.save_ckpt)
    _write(a.out, rec)
    print(f"[train] done: {json.dumps({k: v for k, v in rec.items() if not isinstance(v, list)})}",
          flush=True)


def _sample(model, data_dir, dev, amp) -> str:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    idx = torch.tensor([tok.encode("Once upon a time").ids], device=dev)
    torch.manual_seed(0)
    with amp:
        out = model.generate(idx, 120)
    return tok.decode(out[0].tolist())


def _write(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
