# Meta-RAAD — Project Spec

**Meta-RAAD**: *A Retrieval-Augmented and Feature-Grounded Framework for Explainable Text Anomaly Detection via LLM.*
4th-year final project. Source of truth: the project deck (`new meta-raad.pptx`). If this file and the deck disagree, the deck wins. Update this file whenever a decision changes.

Team: S. Kushal, G. Saharsh, V. Mahender, V. Abhinav Reddy. Mentor: Ms. P. Harika.

---

## 1. Problem

Text anomaly detection with LLMs (zero-shot classifier and model-selection "meta-reasoner") has three weaknesses:

| Problem | Meaning |
|---|---|
| **Context blindness** | Prompts carry only class names ("Sports", "Normal"), not the real data distribution, so the LLM makes false anomaly calls. |
| **Selection bias** | When asked to pick a detection algorithm, the LLM defaults to popular ones (e.g. Isolation Forest) regardless of the dataset. |
| **Vague rationales** | Explanations are canned ("effective for high-dimensional data") and cannot be verified against dataset facts. |

**Objective**: a lightweight orchestration layer (no fine-tuning) that grounds the LLM in real samples, removes selection bias, and measures how factual its rationales are.

## 2. Proposed system

Two modules and one metric.

### 2.1 RA-ZAD — Retrieval-Augmented Zero-Shot Anomaly Detection (context grounding)
- **Offline**: embed the (normal-only) training split into a vector index.
- **Online**: embed the test sample, retrieve top-k nearest real examples (K-NN, cosine similarity), and insert them into the prompt as dynamic few-shot examples.
- LLM returns structured JSON: a `reason` followed by an `anomaly_score` in [0, 1] (reason first, as in AD-LLM, so it acts as implicit chain-of-thought).
- Both AD-LLM settings are supported: **normal_only** and **normal_anomaly**.
- k = 0 is the plain AD-LLM zero-shot baseline.

### 2.2 FG-MOS — Feature-Grounded Model Selection (bias alleviation)
- **Offline**, compute geometric meta-features of the dataset in embedding space:
  - **C_intra**: intra-class compactness.
  - **S_inter**: inter-class separability.
- Standardize **model capability cards** (what each candidate detector is good at).
- The LLM selects a detection algorithm from the meta-features and the cards, not from pre-training habits, and must cite the meta-features in its rationale.

### 2.3 Structured schema and G_score (Grounding Rate)
- LLM output is forced into a JSON schema.
- A script computes **G_score**: how specifically and factually the rationale cites the actual meta-feature values (exact meta-feature citations). This replaces any LLM-as-judge idea. It is automated and verifiable.

### Trade-offs the deck acknowledges
One-time embedding precomputation; one extra vector lookup per query at inference.

## 3. Evaluation
- Detection: **AUROC** and **AUPRC** (same as AD-LLM, so results are comparable).
- Explanation: **G_score**.
- Key comparisons: k = 0 (AD-LLM zero-shot) vs RA-ZAD at k > 0 (ablate k, e.g. 1/2/3/5); FG-MOS selection vs LLM-default selection (bias and quality).
- Seed 42. Both settings: normal_only and normal_anomaly.

## 4. Datasets (AD-LLM / NLPADBench, 5 datasets)
Train split contains normal samples only; test split mixes normal + anomaly. Source: https://github.com/USC-FORTIS/AD-LLM. Configured in `config/datasets.yaml`.

| Dataset | Normal | Anomaly | Train | Test | Anomaly % |
|---|---|---|---|---|---|
| AG News | Sports, Business, Sci/Tech | World | 66,098 | 32,109 | 11.77 |
| BBC News | Business, Politics, Sport, Tech | Entertainment | 1,206 | 579 | 10.71 |
| IMDB Reviews | Positive | Negative | 17,417 | 8,952 | 16.61 |
| N24 News | 23 NYT categories | Food | 40,569 | 19,227 | 9.51 |
| SMS Spam | Ham | Spam | 3,162 | 1,510 | 10.20 |

## 5. Stack
- Python 3.10+; numpy, scikit-learn, faiss-cpu, pydantic.
- LLMs: **local open-source via Ollama / HuggingFace** (per deck).
- Vector DB: FAISS / ChromaDB. The implemented backend uses **ChromaDB**.
- Dev: VS Code / Jupyter.
- Hardware target: i5/i7 or Ryzen 5+, 16 GB RAM (32 recommended), NVIDIA GPU 8+ GB VRAM, 50 GB SSD.

## 6. Repository layout (current)

```
spec.md                     this file
docker-compose.yml          chromadb + backend services
.env.example                copy to .env
requirements.txt            research/experiment deps (torch, pyod, jupyter, ...)
config/
  datasets.yaml             the 5 datasets
  models.yaml               LLM + embedding config
  experiments.yaml          k values, meta-features, G_score, metrics
backend/                    FastAPI service (own Dockerfile + requirements.txt)
  app/main.py
  app/api/                  health.py, rag.py, schemas.py
  app/core/config.py        settings (env-driven)
  app/services/
    embedder.py             sentence-transformers wrapper (BGE query-instruction aware)
    retriever.py            ChromaDB retriever
    llm_client.py           provider-agnostic LLMClient (stub | openai | deepseek)
    rag_service.py          retrieve -> build prompt -> generate
  scripts/test_rag.py       smoke test for the RAG path
notes/
  basepaper.pdf             base paper (AD-LLM)
  AD-LLM_Reference.md       AD-LLM analysis: exact Setting 1/2 prompts, JSON format, results
```

## 7. Status

| Area | State |
|---|---|
| ChromaDB + FastAPI scaffold, Docker | Done |
| Embedder, retriever, `/rag/query`, `/rag/index` | Done (generic RAG prompt) |
| LLM client | Stub only; openai/deepseek branches exist; **Ollama/HF provider not written** |
| N24 News index | Prebuilt ChromaDB in `chroma_n24news.zip` (see 7.1). **Not yet loaded**; the other 4 datasets are not indexed |
| Datasets on disk (`data/` is git-ignored) | **Not downloaded** |
| AD-LLM prompts (Setting 1/2) and detector | Not started |
| RA-ZAD prompt with retrieved examples | Not started |
| Metrics (AUROC/AUPRC) and experiment runners | Not started |
| FG-MOS (C_intra, S_inter, capability cards, selection) | Not started |
| G_score | Not started |

### 7.1 `chroma_n24news.zip` (prebuilt N24 News index)
Git-ignored (735 MB, 1.2 GB unzipped). Contains a persisted ChromaDB directory `chroma_db/` (inspected read-only, not loaded into Docker):
- Collection `rad_llm_n24_news`, **768-dim** (matches BAAI/bge-base-en-v1.5), **76,206 chunks** from **19,227 documents**, average chunk about 239 words.
- Metadata per chunk: `doc_id`, `chunk_id`, `chunk_index`, `total_chunks`, `char_start`, `char_end`, `word_count`, `category`, `is_anomaly`, `is_lead_chunk`, `dataset`.
- 24 categories (23 normal + Food). 4,705 chunks have `is_anomaly=1` (all Food).
- **Caveat**: 19,227 documents equals the N24 **test** size, not train (40,569), and the index contains the anomaly class. RA-ZAD must retrieve only from the **normal-only train split**, so this index cannot be the retrieval corpus as is. Retrieving from test data would leak labels. Either it is the test set (useful as the query side, with a separate train index still needed) or it must be filtered/rebuilt. To be confirmed.
- Chunk-level index: detection works per document, so chunk retrieval must map back to `doc_id` and scores must be aggregated per document.
- Collection name uses the old `rad_llm` prefix (`chroma_collection_prefix` in `backend/app/core/config.py`).
- **Verified**: loads and serves all 76,206 chunks on `chromadb/chroma:1.4.4` (server API reports 1.0.0). The server image and the Python client are pinned to 1.4.4. Collection config: cosine space, HNSW.

### 7.2 Storing and sharing the index
Embeddings cannot go in git (GitHub's limit is 100 MB per file; the index is ~1 GB). Plan: bake the index into a Chroma image (`docker/chroma-n24/Dockerfile`) pushed to **ghcr.io as a public package** (free), tagged per data version (e.g. `n24-v1`). Keep a copy of the zip somewhere safe (Drive or Hugging Face). A rebuild script is still needed for reproducibility. **Before making the image public, confirm N24News (NYT text) allows redistribution**; push private first otherwise.

## 8. Plan
1. **RA-ZAD first**: dataset loader and indexers (train split to vector index) for all 5 datasets; AD-LLM Setting 1/2 prompt templates (`notes/AD-LLM_Reference.md`); detector with JSON parsing; RAG-enhanced prompt with top-k exemplars; AUROC/AUPRC runner; k ablation.
2. Real local LLM client (Ollama / HF).
3. FG-MOS: compute C_intra / S_inter, capability cards, constrained model-selection prompt.
4. G_score script and schema validation.
5. Experiments, tables, report.

## 9. Open decisions
- **Embedding model**: deck says `text-embedding-3-large`; code uses `BAAI/bge-base-en-v1.5` (local, free, fits the local-LLM stack). Keep BGE unless the team decides otherwise.
- **Which local LLM** (Llama 3.1 8B, Qwen, etc.) and runtime (Ollama vs HF).
- **Vector store**: ChromaDB (implemented) vs FAISS (deck lists both).
- **Candidate detector pool for FG-MOS**: likely PyOD-style (Isolation Forest, LOF, OCSVM, ...). Needs a final list.
- Exact **G_score formula**.

## 10. Out of scope (not in the deck; removed from the repo)
Self-consistency / multi-path scoring and uncertainty, selective prediction, LLM-as-judge explanation scoring, and paid-API model configs (GPT-4o, DeepSeek) as the primary models. These came from an earlier RAD-LLM draft that does not match the deck. It remains in git history (commit `37cacf7`, `notebooks/RAD-LLM_Framework_Specification.md`) if ever needed.

## 11. References
1. LLMs for Anomaly and OOD Detection: A Survey — arXiv 2409.01980
2. AD-LLM: Benchmarking LLMs for Anomaly Detection — arXiv 2412.11142 (base paper)
3. AAD-LLM: Adaptive Anomaly Detection Using LLMs — arXiv 2411.00914
4. Retrieval Augmented Deep Anomaly Detection for Tabular Data — arXiv 2401.17052
