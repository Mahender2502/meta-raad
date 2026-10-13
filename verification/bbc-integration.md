# BBC integration

Both `n24_news` and `bbc` are registered. BBC source: https://huggingface.co/datasets/kendx/NLP-ADBench/tree/main/datasets/bbc

BBC: 1,206 normal training articles; 579 test articles (517 normal, 62 entertainment anomalies). BERT vectors are row-aligned and 768-dimensional. Normal classes: business, sport, politics, tech. Source files are preserved.

One normalized exact train/test overlap was found; the CLI evaluation excludes it, leaving 578 test records. Query self-match exclusion and batch validation remain enabled. Typed-text detection is permitted even when the source split has overlaps, with a visible notice.

Full BBC local baseline evaluation (1,206 train / 578 eligible test):

| Detector | AUROC | Average precision |
|---|---:|---:|
| iforest | 0.7533 | 0.2546 |
| lof | 0.9395 | 0.6526 |
| knn | 0.9453 | 0.6548 |

These are classical-detector results, not measurements of RAG improvement. Model selection returned a valid eight-candidate ranking through the configured free Nemotron API. Numeric citation precision was 1.0 across eight supplied citations; this does not validate recommendation quality.

Validation: 32 backend tests; 14 frontend tests; production frontend build. Dataset switching remounts page state and clears shared detection results to avoid showing one dataset's result under another dataset.
