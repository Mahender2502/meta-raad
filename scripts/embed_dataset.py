#!/usr/bin/env python3
"""
Embed articles with bert-base-uncased, reproducing the NLP-ADBench vectors.

Reads every .jsonl file in an input folder (one JSON object per line with a "text" field),
embeds each article and writes the vectors to an output folder, keeping the file stem and
the row order. Output name: <stem>_bert_base_uncased_feature.npy (the benchmark's naming).

Recipe (verified): CLS token of the last hidden layer of bert-base-uncased, input truncated
to 512 tokens, float32, one row per article. Against the benchmark's own N24News train
vectors this gives cosine 1.000000 (use --reference-dir to check any dataset you have).

Provenance: Meta-RAAD uses the benchmark's precomputed vectors, downloaded with
scripts/download_data.py; they were NOT generated with this script. This script reproduces
and checks them, embeds text that has no precomputed vectors (the demo uses the same recipe
in backend/app/services/bert_embedder.py), and can embed other datasets. On a CPU it runs at
about 4 articles/s, so the full N24 train split (40,595 articles) takes hours; use --limit
for a check.

Optional chunking (--chunk-chars N): additionally splits every article into windows of about
N characters (--chunk-overlap overlap, cut at spaces), embeds each chunk, and writes
<stem>_chunks_bert_base_uncased_feature.npy plus <stem>_chunks.jsonl (one line per chunk:
article row, chunk index, character span, text, and the article's other fields). The project's
method is article-level, so chunking is an optional extra for chunk-level retrieval.

Usage:
    python scripts/embed_dataset.py --input-dir data/in --output-dir data/out
    python scripts/embed_dataset.py --input-dir data/n24 --output-dir /tmp/out --limit 200 \\
        --reference-dir data/n24                      # compare with the benchmark's vectors
    python scripts/embed_dataset.py --input-dir data/in --output-dir data/out --chunk-chars 1500
Needs: torch, transformers, numpy (all in the backend Docker image).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL = "bert-base-uncased"
MAX_LEN = 512
SUFFIX = "_bert_base_uncased_feature.npy"


def load_records(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue  # tolerate blank lines
            record = json.loads(line)
            if "text" not in record:
                raise SystemExit(f"{path.name} line {line_no}: no 'text' field")
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of windows of about `size` chars, overlapping by `overlap`, cut at spaces."""
    if size <= overlap:
        raise SystemExit("--chunk-chars must be larger than --chunk-overlap")
    spans, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # prefer to cut at the last space inside the window
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut != -1 else end
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


class Embedder:
    def __init__(self, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(device).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        # Batch texts of similar length together (padding is masked, so the CLS vector is
        # unaffected), then restore the input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        start = time.time()
        for b in range(0, len(order), self.batch_size):
            idx = order[b : b + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", truncation=True,
                                 max_length=MAX_LEN, padding=True).to(self.device)
            with self.torch.no_grad():
                out[idx] = self.model(**enc).last_hidden_state[:, 0].cpu().numpy()
            done = b + len(idx)
            print(f"\r    {done}/{len(texts)}  ({done / (time.time() - start):.1f}/s)", end="", flush=True)
        print()
        return out


def compare(vectors: np.ndarray, ref_path: Path) -> None:
    ref = np.load(ref_path, mmap_mode="r")[: len(vectors)].astype(np.float32)
    if len(ref) < len(vectors):
        print(f"    reference has only {len(ref)} rows; comparison skipped")
        return
    cos = (vectors * ref).sum(1) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"    vs {ref_path.name}: min cosine {cos.min():.6f}, mean {cos.mean():.6f}, "
          f"max abs difference {np.abs(vectors - ref).max():.4f}")


def process(path: Path, out_dir: Path, embedder: Embedder, args: argparse.Namespace) -> None:
    records = load_records(path, args.limit)
    print(f"  {path.name}: {len(records)} articles")
    texts = [r["text"] for r in records]
    vectors = embedder.embed(texts)
    np.save(out_dir / f"{path.stem}{SUFFIX}", vectors)
    print(f"    wrote {path.stem}{SUFFIX} {vectors.shape}")
    if args.reference_dir:
        ref = args.reference_dir / f"{path.stem}{SUFFIX}"
        if ref.exists():
            compare(vectors, ref)
        else:
            print(f"    no reference file {ref.name} in {args.reference_dir}")

    if args.chunk_chars:
        chunk_rows, chunk_texts = [], []
        for row, record in enumerate(records):
            for ci, (a, b) in enumerate(chunk_text(record["text"], args.chunk_chars, args.chunk_overlap)):
                meta = {k: v for k, v in record.items() if k != "text"}
                chunk_rows.append({"article_row": row, "chunk_index": ci, "char_start": a,
                                   "char_end": b, "text": record["text"][a:b], **meta})
                chunk_texts.append(record["text"][a:b])
        print(f"    chunking: {len(chunk_texts)} chunks of ~{args.chunk_chars} chars")
        cvec = embedder.embed(chunk_texts)
        np.save(out_dir / f"{path.stem}_chunks{SUFFIX}", cvec)
        with open(out_dir / f"{path.stem}_chunks.jsonl", "w", encoding="utf-8") as f:
            for r in chunk_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    wrote {path.stem}_chunks{SUFFIX} {cvec.shape} and {path.stem}_chunks.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="folder of .jsonl files with a 'text' field")
    parser.add_argument("--output-dir", required=True, type=Path, help="folder to write the .npy (and chunk) files to")
    parser.add_argument("--pattern", default="*.jsonl", help="which files in the input folder (default *.jsonl)")
    parser.add_argument("--limit", type=int, help="only the first N articles of each file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dir", type=Path, help="folder holding reference .npy files to compare with")
    parser.add_argument("--chunk-chars", type=int, help="also embed chunks of about this many characters")
    parser.add_argument("--chunk-overlap", type=int, default=250)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Embedding {len(files)} file(s) from {args.input_dir} with {MODEL} (CLS, max {MAX_LEN} tokens)")
    embedder = Embedder(args.device, args.batch_size)
    for path in files:
        process(path, args.output_dir, embedder, args)
    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
