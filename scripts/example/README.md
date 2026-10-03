# Example: input folder → output folder

A tiny, runnable reference for `scripts/embed_dataset.py`. The four articles in
`input/sample_articles.jsonl` are short texts written for this example (not dataset text).

```
scripts/example/
├── input/
│   └── sample_articles.jsonl        one JSON object per line; "text" is required, other fields are kept
└── output/                          produced by the command below
    ├── sample_articles_bert_base_uncased_feature.npy          (4, 768)  one vector per article, same row order
    ├── sample_articles_chunks_bert_base_uncased_feature.npy   (8, 768)  one vector per chunk
    └── sample_articles_chunks.jsonl                           one line per chunk: article_row, chunk_index,
                                                               char_start, char_end, text, and the article's other fields
```

Regenerate (needs torch + transformers, e.g. the backend Docker image):

```bash
docker run --rm -v "$PWD":/work -w /work meta-raad-backend:latest \
  python scripts/embed_dataset.py --input-dir scripts/example/input \
  --output-dir scripts/example/output --chunk-chars 250 --chunk-overlap 50
```

Without `--chunk-chars` only the article-level `.npy` is written. The project's retrieval is
article-level.

**Provenance.** The vectors the project actually uses are NLP-ADBench's, downloaded with
`scripts/download_data.py` into `data/` (git-ignored). This script reproduces them (cosine 1.000000 on the
rows checked) and handles text that has no precomputed vectors; it was not used to produce the project's data.
