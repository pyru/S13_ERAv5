"""Build S13_Reversible_LM.ipynb: a self-contained Colab notebook.

The package source is embedded with %%writefile cells generated from the files in this
folder, so the notebook can never drift from the code. Upload the .ipynb to Colab, pick a
GPU runtime, Run all.

    python build_notebook.py
"""
from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "S13_Reversible_LM.ipynb")
SOURCES = ["revlm/__init__.py", "revlm/reversible.py", "revlm/model.py", "revlm/data.py",
           "revlm/train.py", "revlm/probe.py", "revlm/fidelity.py",
           "tests/test_reversible.py", "run_all.py", "make_report.py"]

cells = []
md = lambda t: cells.append(new_markdown_cell(t.strip()))  # noqa: E731
code = lambda t: cells.append(new_code_cell(t.strip()))  # noqa: E731

md("""
# S13 · A 20M LLM trained for 50M tokens, with and without reversibility

1. **Baseline**: standard pre-LN GPT, 20.0M params, fixed batch 64 × 512 tokens, 50M tokens.
2. **Reversible**: same model, same batch, same data order, with activations rebuilt in the
   backward pass. Four integrators are tried first (RevNet coupling, midpoint, leapfrog,
   Euler + fixed-point inversion) and the best one that is trustworthy is kept.
3. **Reversible at maximum batch**: same variant, largest batch that fits on this GPU.

Reported: final train/val loss, tokens/s, peak GPU memory, plus the variant sweep, gradient
fidelity vs autograd, and max-batch probes for baseline / activation checkpointing /
reversible.

**Runtime → Change runtime type → GPU.** On a T4 expect roughly 1.5–2.5 h for everything;
on an L4/A100 well under an hour. Every stage writes JSON and skips finished work, so
after a disconnect just run all cells again (mount Drive below to survive a VM reset).
""")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())
""")
md("Data and runs go to Google Drive (approve the mount pop-up), so a disconnect or runtime reset loses nothing. Set `USE_DRIVE = False` to keep everything on the VM instead.")
code("""
USE_DRIVE = True
ROOT = "/content/s13"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/s13_reversible"
import os
os.makedirs(ROOT, exist_ok=True)
os.makedirs("/content/s13_code/revlm", exist_ok=True)
os.makedirs("/content/s13_code/tests", exist_ok=True)
%cd /content/s13_code
!pip -q install datasets tokenizers
import subprocess, sys

def stage(name):
    # A failing `!python ...` does not stop "Run all"; this does, so the first error is
    # the one you see instead of a cascade of empty stages.
    p = subprocess.Popen([sys.executable, "-u", "run_all.py", "--root", ROOT, "--stage", name],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in p.stdout:  # relay live; child output is otherwise invisible in Colab
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"stage {name!r} failed (exit {p.returncode}); see output above")
""")
md("## Source\nGenerated from the repo; the reversible machinery is in `revlm/reversible.py`.")
for rel in SOURCES:
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        code(f"%%writefile {rel}\n{f.read()}")

stages = [
    ("data", "TinyStories → 8192-token BPE → 60M train tokens (≈5 min)."),
    ("test", "fp64 proof: reversible gradients equal autograd gradients."),
    ("sweep", "Every integrator for 5M tokens at batch 64, same batches."),
    ("fidelity", "Reversible vs autograd gradients in real AMP precision, at init and after the sweep; picks the variant."),
    ("run1", "**Run 1**: baseline, batch 64, 50M tokens."),
    ("run2", "**Run 2**: chosen reversible variant, batch 64, 50M tokens."),
    ("probe", "Largest batch that fits: baseline, activation checkpointing, reversible."),
    ("run3", "**Run 3**: reversible at its maximum batch, 50M tokens (lr sqrt-scaled, capped at 3e-3)."),
    ("report", "Tables and plots → `report/`."),
]
for s, desc in stages:
    md(f"### `{s}`\n{desc}")
    code(f'stage("{s}")')

md("## Results")
code("""
from IPython.display import Markdown, Image, display
import re
rep = f"{ROOT}/report"
text = open(f"{rep}/results.md", encoding="utf-8").read()
for part in re.split(r"!\\[[^\\]]*\\]\\(([^)]+)\\)", text):
    if part.endswith(".png"):
        display(Image(f"{rep}/{part}"))
    elif part.strip():
        display(Markdown(part))
""")

md("## Download the results\nZips the JSON results and the report (not the data or checkpoints). Bring `s13_results.zip` back to the repo.")
code("""
import shutil
from google.colab import files
os.makedirs("/content/s13_results", exist_ok=True)
for f in os.listdir(f"{ROOT}/runs"):
    if f.endswith(".json"):
        shutil.copy(f"{ROOT}/runs/{f}", "/content/s13_results/")
shutil.copytree(f"{ROOT}/runs/sweep", "/content/s13_results/sweep", dirs_exist_ok=True)
shutil.copytree(f"{ROOT}/report", "/content/s13_results/report", dirs_exist_ok=True)
shutil.make_archive("/content/s13_results", "zip", "/content/s13_results")
files.download("/content/s13_results.zip")
""")

nb = new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
