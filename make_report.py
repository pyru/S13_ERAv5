"""runs/*.json -> report/results.md + report/*.png. Missing runs are simply left out."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (colour follows the entity, never its rank)
COL = {"standard": "#2a78d6", "revnet": "#eb6834", "midpoint": "#1baf7a",
       "leapfrog": "#eda100", "euler": "#e87ba4", "ckpt": "#4a3aa7",
       "run1": "#2a78d6", "run2": "#eb6834", "run3": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GiB = 2**30


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def fmt_mem(b):
    return f"{b / GiB:.2f} GiB" if b else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(a.runs)), "report")
    os.makedirs(out, exist_ok=True)
    md = ["# Results — 20M LLM, 50M tokens, with and without reversibility\n"]

    runs = {k: load(f"{a.runs}/{f}.json") for k, f in
            (("run1", "run1_baseline"), ("run2", "run2_reversible"),
             ("run3", "run3_reversible_maxbatch"))}
    runs = {k: v for k, v in runs.items() if v}
    sweep = {m: load(f"{a.runs}/sweep/{m}.json") for m in
             ("standard", "revnet", "midpoint", "leapfrog", "euler")}
    sweep = {k: v for k, v in sweep.items() if v}
    win = load(f"{a.runs}/winner.json") or {}
    fid = (load(f"{a.runs}/fidelity.json") or {})
    probes = {m: load(f"{a.runs}/probe_{m}.json") for m in
              ("standard", "ckpt", "revnet", "midpoint", "leapfrog", "euler")}
    probes = {k: v for k, v in probes.items() if v}

    any_run = next(iter(runs.values()), None) or next(iter(sweep.values()), None)
    if any_run:
        md.append(f"GPU **{any_run['device']}**, AMP **{any_run['amp']}**, "
                  f"{any_run['n_params']:,} parameters ({any_run['n_nonemb']:,} non-embedding), "
                  f"seq len {any_run['seq']}.\n")

    # ---------------------------------------------------------------- main table
    if runs:
        md += ["## The three runs\n",
               "| run | mode | batch (seqs) | tokens/step | steps | lr | final train loss | "
               "final val loss | tok/s (steady) | peak mem | model state | wall (train) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        names = {"run1": "1 · baseline", "run2": "2 · reversible", "run3": "3 · reversible, max batch"}
        for k, r in runs.items():
            if r.get("status") != "ok":
                md.append(f"| {names[k]} | {r['mode']} | {r['batch_size']} | "
                          f"{r['batch_size'] * r['seq']:,} | {r['steps']} | {r['lr']:.2g} | "
                          f"**{r['status'].upper()}** | | | {fmt_mem(r.get('peak_mem_allocated'))} | | |")
                continue
            md.append(
                f"| {names[k]} | {r['mode']} | {r['batch_size']} | {r['batch_size'] * r['seq']:,} | "
                f"{r['steps']} | {r['lr']:.2g} | {r.get('final_train_loss', float('nan')):.4f} | "
                f"{r.get('final_val_loss', float('nan')):.4f} | "
                f"{r.get('tok_per_s_steady', r.get('tok_per_s', 0)):,.0f} | "
                f"{fmt_mem(r.get('peak_mem_allocated'))} | {fmt_mem(r.get('model_state_mem'))} | "
                f"{r.get('train_time_s', 0) / 60:.1f} min |")
        if "run1" in runs and "run2" in runs:
            r1, r2 = runs["run1"], runs["run2"]
            act1 = r1["peak_mem_allocated"] - r1["model_state_mem"]
            act2 = r2["peak_mem_allocated"] - r2["model_state_mem"]
            md.append(
                f"\nAt the same batch, reversibility cut peak memory "
                f"{r1['peak_mem_allocated'] / r2['peak_mem_allocated']:.2f}x "
                f"(activations+temporaries {act1 / GiB:.2f} → {act2 / GiB:.2f} GiB, "
                f"{act1 / max(act2, 1):.1f}x) and ran at "
                f"{r2['tok_per_s_steady'] / r1['tok_per_s_steady']:.2f}x the baseline speed. "
                f"Val loss moved by {r2['final_val_loss'] - r1['final_val_loss']:+.4f}.\n")

        ok = {k: r for k, r in runs.items() if r.get("status") == "ok"}
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
        for k, r in ok.items():
            ev = r["evals"]
            ax.plot([e[1] / 1e6 for e in ev], [e[2] for e in ev], color=COL[k], lw=2,
                    marker="o", ms=4, label=f"{names[k]} (B={r['batch_size']})")
        style(ax, "Validation loss vs tokens seen", "tokens (M)", "val loss")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out}/runs_val_loss.png")
        md.append("![val loss](runs_val_loss.png)\n")

        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), dpi=130)
        labels = [f"run {k[-1]}\nB={r['batch_size']}" for k, r in ok.items()]
        cols = [COL[k] for k in ok]
        for ax, key, title, scale, unit in (
                (axs[0], "peak_mem_allocated", "Peak GPU memory", GiB, "GiB"),
                (axs[1], "tok_per_s_steady", "Throughput", 1e3, "k tok/s")):
            vals = [r.get(key, 0) / scale for r in ok.values()]
            bars = ax.bar(labels, vals, color=cols, width=0.55)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}" if unit == "GiB" else f"{v:.0f}",
                            (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")
            style(ax, title, "", unit)
        fig.tight_layout()
        fig.savefig(f"{out}/runs_mem_speed.png")
        md.append("![memory and speed](runs_mem_speed.png)\n")

    # ---------------------------------------------------------------- variant sweep
    if sweep:
        base = sweep.get("standard")
        md += ["## Which reversible variant worked\n",
               f"Every integrator trained for {next(iter(sweep.values()))['tokens_target'] / 1e6:.0f}M tokens "
               f"at batch {next(iter(sweep.values()))['batch_size']}, same seed, same batches.\n",
               "| mode | exact inverse? | status | final val loss | Δ vs standard | tok/s (steady) | "
               "speed vs standard | peak mem | grad cosine (init) | grad cosine (trained) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        fr = fid.get("results", {})
        for m, r in sweep.items():
            f = fr.get(m, {})
            ci = f"{f['init']['cos']:.6f}" if "init" in f else "—"
            ct = f"{f['trained']['cos']:.6f}" if "trained" in f else "—"
            vl = r.get("final_val_loss")
            dv = f"{vl - base['final_val_loss']:+.4f}" if (vl and base and base.get("final_val_loss")) else "—"
            sp = (f"{r['tok_per_s_steady'] / base['tok_per_s_steady']:.2f}x"
                  if base and r.get("tok_per_s_steady") else "—")
            exact = {"standard": "(no inverse needed)", "euler": "no (fixed point)"}.get(m, "yes")
            md.append(f"| {m} | {exact} | {r['status']} | {vl if vl is None else f'{vl:.4f}'} | {dv} | "
                      f"{r.get('tok_per_s_steady', 0):,.0f} | {sp} | {fmt_mem(r.get('peak_mem_allocated'))} | "
                      f"{ci} | {ct} |")
        md.append("")
        if win:
            md.append(f"\n**Chosen variant: `{win['winner']}`**"
                      f"{' (forced)' if win.get('forced') else ''}. Rule: lowest sweep val loss among "
                      "variants whose trained-weight gradient cosine vs autograd is ≥ 0.999.\n")
            md += [f"- `{m}`: {w}" for m, w in win.get("why", {}).items()]
            md.append("")
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
        for m, r in sweep.items():
            c = r["curve"]
            ax.plot([p[1] / 1e6 for p in c], [p[2] for p in c], color=COL[m], lw=2, label=m)
        style(ax, "Variant sweep — training loss", "tokens (M)", "train loss")
        ax.legend(frameon=False)
        lo = min(min(p[2] for p in r["curve"]) for r in sweep.values())
        ax.set_ylim(lo - 0.1, lo + 3.0)
        fig.tight_layout()
        fig.savefig(f"{out}/sweep_loss.png")
        md.append("![variant sweep](sweep_loss.png)\n")

    # ---------------------------------------------------------------- probe
    if probes:
        md += ["## Maximum batch size that fits\n",
               f"3 real training steps per attempt, seq len {next(iter(probes.values()))['seq']}, "
               f"GPU memory {next(iter(probes.values()))['gpu_total'] / GiB:.1f} GiB.\n",
               "| mode | max batch (seqs) | tokens/step | peak at max | tok/s at max |",
               "|---|---|---|---|---|"]
        for m, p in probes.items():
            md.append(f"| {m} | {p['max_batch']} | {p['max_batch'] * p['seq']:,} | "
                      f"{fmt_mem(p.get('peak_at_max'))} | {(p.get('tok_per_s_at_max') or 0):,.0f} |")
        md.append("")
        fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
        for m, p in probes.items():
            pts = sorted((int(b), v["peak"] / GiB) for b, v in p["tried"].items() if v.get("ok"))
            if pts:
                ax.plot(*zip(*pts), color=COL[m], lw=2, marker="o", ms=5, label=f"{m} (max {p['max_batch']})")
        gt = next(iter(probes.values()))["gpu_total"] / GiB
        ax.axhline(gt, color=MUTED, lw=1, ls="--")
        ax.annotate("GPU capacity", (ax.get_xlim()[0], gt), color=MUTED, fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
        style(ax, "Peak memory vs batch size", "batch size (sequences)", "peak GiB")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out}/probe_memory.png")
        md.append("![probe](probe_memory.png)\n")

    samples = [(k, r["sample"]) for k, r in runs.items() if r.get("sample")]
    if samples:
        md.append("## Samples (prompt: \"Once upon a time\", T=0.8, top-k 40)\n")
        for k, s in samples:
            md.append(f"**{k}** — {s.strip()}\n")

    with open(f"{out}/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[report] wrote {out}/results.md")


if __name__ == "__main__":
    main()
