#!/usr/bin/env python3
"""
Download the NLP-ADBench files Meta-RAAD needs for one dataset.

Source: https://huggingface.co/datasets/kendx/NLP-ADBench (MIT licence; the N24News text
is New York Times articles, so do not redistribute it beyond the benchmark).

For a dataset <D> (e.g. N24News) this fetches, per split (train / test):
    datasets/<D>/<D>_<split>_data.jsonl                              article text + labels
    embeddings/<D>/<D>_<split>_data_bert_base_uncased_feature.npy    precomputed BERT vectors
and optionally the OpenAI text-embedding-3-large vectors (--openai, ~1.5 GB for N24).

Downloads resume if interrupted, skip files that are already complete, and the script checks
that each vector file has exactly one row per article.

Usage:
    python scripts/download_data.py                       # N24News, train + test, BERT vectors
    python scripts/download_data.py --splits train        # train only
    python scripts/download_data.py --dataset N24News --out-dir data/n24

Only N24News has been verified; other dataset folder names are taken from the repository
listing and are checked with a HEAD request before downloading.
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/kendx/NLP-ADBench/resolve/main"
DEFAULT_OUT = {"N24News": "data/n24"}  # other datasets default to data/<name lowercased>
CHUNK = 1 << 20


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # optional; fixes missing CA bundles on some Python installs

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ssl_context()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, context=CTX, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(url: str, dest: Path) -> None:
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  have  {dest.name} ({size / 1e6:.1f} MB)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    request = urllib.request.Request(url, headers=headers)
    print(f"  fetch {dest.name} ({size / 1e6:.1f} MB)" + (f", resuming at {done / 1e6:.1f} MB" if done else ""))
    with urllib.request.urlopen(request, context=CTX, timeout=60) as response:
        if done and response.status != 206:  # server ignored the range: start over
            done = 0
        with open(part, "ab" if done else "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                print(f"\r    {done / size:6.1%}", end="", flush=True)
    print()
    if part.stat().st_size != size:
        raise SystemExit(f"Incomplete download of {dest.name}: {part.stat().st_size} of {size} bytes")
    part.replace(dest)


def npy_rows(path: Path) -> int:
    """Row count from the .npy header, without loading the array."""
    import numpy as np

    return int(np.load(path, mmap_mode="r").shape[0])


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="N24News", help="folder name in the repository")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--out-dir", help="default: data/n24 for N24News, else data/<dataset>")
    parser.add_argument("--openai", action="store_true", help="also fetch text-embedding-3-large vectors (large)")
    args = parser.parse_args()

    d = args.dataset
    out = Path(args.out_dir or DEFAULT_OUT.get(d, f"data/{d.lower()}"))
    out.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, Path]] = []
    for split in args.splits:
        plan.append((f"{BASE}/datasets/{d}/{d}_{split}_data.jsonl", out / f"{d}_{split}_data.jsonl"))
        plan.append(
            (
                f"{BASE}/embeddings/{d}/{d}_{split}_data_bert_base_uncased_feature.npy",
                out / f"{d}_{split}_data_bert_base_uncased_feature.npy",
            )
        )
        if args.openai:
            name = f"{d}_{split}_data_gpt_text-embedding-3-large_feature.npy"
            plan.append((f"{BASE}/embeddings/{d}/{name}", out / name))

    print(f"Downloading {d} into {out}/")
    try:
        for url, dest in plan:
            download(url, dest)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} for {exc.url} (is '{d}' a dataset folder in the repository?)")

    try:  # consistency check: one vector per article
        for split in args.splits:
            n_text = count_lines(out / f"{d}_{split}_data.jsonl")
            n_vec = npy_rows(out / f"{d}_{split}_data_bert_base_uncased_feature.npy")
            status = "ok" if n_text == n_vec else "MISMATCH"
            print(f"  {split}: {n_text} articles, {n_vec} vectors -> {status}")
            if n_text != n_vec:
                raise SystemExit("Row counts differ; do not use these files together.")
        if "train" in args.splits:
            with open(out / f"{d}_train_data.jsonl", encoding="utf-8") as f:
                labels = {json.loads(line)["label"] for line in f}
            print(f"  train labels: {sorted(labels)} (expected [0], normal only)")
    except ImportError:
        print("  (numpy not installed: skipped the row-count check)")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
