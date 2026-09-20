"""TinyStories -> 8192-token byte-level BPE -> flat uint16 token files.

    python -m revlm.data --out data --train_tokens 60e6

Writes data/train.bin, data/val.bin (uint16, stories joined by <|endoftext|>),
data/tokenizer.json and data/meta.json. Stories are streamed from the Hugging Face hub,
so only what is needed is downloaded. 60M train tokens are prepared so that a 50M-token
run draws its windows from more text than it consumes.

A small vocabulary is what makes "20M parameters" mean a real transformer: with GPT-2's
50,257 tokens the tied embedding alone would be 16M of the 20M.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

EOT = "<|endoftext|>"


def _stories(split: str):
    from datasets import load_dataset

    for row in load_dataset("roneneldan/TinyStories", split=split, streaming=True):
        t = row["text"].strip()
        if t:
            yield t


def _train_tokenizer(texts, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def _encode(tok, texts, max_tokens: int, path: str, batch: int = 4096) -> tuple[int, int]:
    eot = tok.token_to_id(EOT)
    parts, n_tok, n_docs, buf = [], 0, 0, []

    def flush():
        nonlocal n_tok, n_docs
        for enc in tok.encode_batch(buf):
            parts.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
            n_tok += len(enc.ids) + 1
            n_docs += 1
        buf.clear()

    for t in texts:
        buf.append(t)
        if len(buf) == batch:
            flush()
            if n_tok >= max_tokens:
                break
    if buf and n_tok < max_tokens:
        flush()
    arr = np.concatenate(parts)[:max_tokens]
    arr.tofile(path)
    return len(arr), n_docs


def prepare(out: str, train_tokens: float, val_tokens: float, vocab_size: int, tok_stories: int):
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "meta.json")):
        print(f"[data] {out}/meta.json exists, skipping")
        return
    stream = _stories("train")
    head = [next(stream) for _ in range(tok_stories)]
    print(f"[data] training {vocab_size}-token BPE on {len(head):,} stories")
    tok = _train_tokenizer(head, vocab_size)
    tok.save(os.path.join(out, "tokenizer.json"))

    def train_texts():
        yield from head
        yield from stream

    n_train, d_train = _encode(tok, train_texts(), int(train_tokens), os.path.join(out, "train.bin"))
    n_val, d_val = _encode(tok, _stories("validation"), int(val_tokens), os.path.join(out, "val.bin"))
    meta = dict(dataset="roneneldan/TinyStories", vocab_size=tok.get_vocab_size(),
                train_tokens=n_train, train_stories=d_train, val_tokens=n_val, val_stories=d_val)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] {meta}")


class Batches:
    """Random (B, T+1) windows from a token file; seeded, so two runs with the same seed
    and batch size see exactly the same batches in the same order."""

    def __init__(self, path: str, B: int, T: int, seed: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T, self.device = B, T, device
        self.rng = np.random.default_rng(seed)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.T - 1, size=self.B)
        x = torch.from_numpy(np.stack([self.data[i:i + self.T + 1] for i in ix]).astype(np.int64))
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        return x[:, :-1], x[:, 1:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train_tokens", type=float, default=60e6)
    ap.add_argument("--val_tokens", type=float, default=4e6)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--tok_stories", type=int, default=200_000)
    a = ap.parse_args()
    prepare(a.out, a.train_tokens, a.val_tokens, a.vocab_size, a.tok_stories)
