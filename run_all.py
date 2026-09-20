"""The whole assignment, end to end, resumable.

    python run_all.py                 # everything
    python run_all.py --stage sweep   # one stage
    python run_all.py --root /content/drive/MyDrive/s13   # keep runs/ on Drive

Stages (each skips work whose JSON already exists, so a Colab disconnect costs one run):

  data      TinyStories -> 8192-token BPE -> 60M train tokens
  test      fp64 proof that the reversible backward equals autograd
  sweep     short (5M-token) run of every integrator at the fixed batch -> pick a variant
  fidelity  reversible vs autograd gradients in the real AMP precision, at init and trained
  run1      baseline, fixed batch, 50M tokens
  run2      chosen reversible variant, same batch, same data order, 50M tokens
  probe     largest batch that fits: baseline, activation checkpointing, reversible
  run3      reversible at its maximum batch, 50M tokens
  report    report/results.md + plots
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_MODES = ["standard", "revnet", "midpoint", "leapfrog", "euler"]
EXACT = ["revnet", "midpoint", "leapfrog"]


def sh(args, check=True):
    print("$", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=HERE)
    if check and r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(args)}")
    return r.returncode


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def done(path):
    r = load(path)
    return r is not None and r.get("status") in ("ok", "diverged")


def train(a, mode, B, tokens, out, lr=None, extra=()):
    if done(out):
        print(f"[run_all] {out} exists, skipping", flush=True)
        return load(out)
    args = [sys.executable, "-m", "revlm.train", "--mode", mode, "--batch_size", str(B),
            "--tokens", str(tokens), "--lr", str(lr or a.lr), "--data", a.data, "--out", out,
            *extra]
    return sh(args, check=False)


def choose(a):
    """Lowest sweep val loss among variants whose gradients are trustworthy."""
    fid = (load(f"{a.runs}/fidelity.json") or {}).get("results", {})
    cands, why = [], {}
    for m in EXACT + ["euler"]:
        r = load(f"{a.runs}/sweep/{m}.json")
        if not r or r.get("status") != "ok":
            why[m] = f"excluded: status {r and r.get('status')}"
            continue
        src = "trained" if "trained" in fid.get(m, {}) else "init"
        f = fid.get(m, {}).get(src) or {}
        if f and f.get("cos", 0) < 0.999:
            why[m] = f"excluded: grad cosine vs autograd ({src} weights) {f['cos']:.4f} < 0.999"
            continue
        cands.append((r["final_val_loss"], -r.get("tok_per_s_steady", 0), m))
        why[m] = f"val {r['final_val_loss']:.4f}, {r.get('tok_per_s_steady', 0):,.0f} tok/s"
    if a.rev_mode:
        best = a.rev_mode
    elif cands:
        best = min(cands)[2]
    else:
        best = "revnet"
    res = {"winner": best, "forced": bool(a.rev_mode), "why": why}
    json.dump(res, open(f"{a.runs}/winner.json", "w"), indent=1)
    print(f"[run_all] reversible variant: {best}   {why}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--sweep_tokens", type=float, default=5e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_cap", type=float, default=3e-3)
    ap.add_argument("--rev_mode", default="", help="skip the choice and force a variant")
    a = ap.parse_args()
    a.data = os.path.join(a.root, "data")
    a.runs = os.path.join(a.root, "runs")
    os.makedirs(f"{a.runs}/sweep", exist_ok=True)
    os.makedirs(f"{a.runs}/ckpt", exist_ok=True)
    want = lambda s: a.stage in ("all", s)  # noqa: E731
    B = a.batch_size

    if want("data"):
        sh([sys.executable, "-m", "revlm.data", "--out", a.data])
    if want("test"):
        sh([sys.executable, "tests/test_reversible.py"])
    if want("sweep"):
        for m in SWEEP_MODES:
            train(a, m, B, a.sweep_tokens, f"{a.runs}/sweep/{m}.json",
                  extra=("--save_ckpt", f"{a.runs}/ckpt/{m}.pt", "--no_sample"))
    if want("fidelity") and not os.path.exists(f"{a.runs}/fidelity.json"):
        sh([sys.executable, "-m", "revlm.fidelity", "--data", a.data,
            "--ckpt_dir", f"{a.runs}/ckpt", "--out", f"{a.runs}/fidelity.json"])
    rev = (load(f"{a.runs}/winner.json") or {}).get("winner")
    if want("fidelity") or a.rev_mode or (rev is None and a.stage in ("run2", "probe", "run3")):
        rev = choose(a)
    if want("run1"):
        train(a, "standard", B, a.tokens, f"{a.runs}/run1_baseline.json")
    if want("run2"):
        train(a, rev, B, a.tokens, f"{a.runs}/run2_reversible.json")
    if want("probe"):
        for m in ("standard", "ckpt", rev):
            out = f"{a.runs}/probe_{m}.json"
            if not os.path.exists(out):
                sh([sys.executable, "-m", "revlm.probe", "--mode", m, "--out", out])
    if want("run3"):
        out = f"{a.runs}/run3_reversible_maxbatch.json"
        Bmax = load(f"{a.runs}/probe_{rev}.json")["max_batch"]
        oom_at = []
        for _ in range(5):
            lr = min(a.lr * math.sqrt(Bmax / B), a.lr_cap)  # sqrt scaling for Adam
            train(a, rev, Bmax, a.tokens, out, lr=lr, extra=("--warmup_frac", "0.05"))
            # decide from what the run recorded, not from the exit code
            if (load(out) or {}).get("status") != "oom":
                break
            oom_at.append(Bmax)
            Bmax = int(Bmax * 0.95) // 16 * 16
            print(f"[run_all] OOM in the real run, retrying at B={Bmax}", flush=True)
        r = load(out)
        if r is not None and oom_at:
            r["oom_retries_at"] = oom_at  # the probe's max that did not survive a full run
            json.dump(r, open(out, "w"), indent=1)
    if want("report"):
        sh([sys.executable, "make_report.py", "--runs", a.runs])


if __name__ == "__main__":
    main()
