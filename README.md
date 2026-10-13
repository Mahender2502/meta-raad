# Meta-RAAD

Two implemented workflows: retrieval-augmented text anomaly detection (RA-ZAD), and feature-grounded detector recommendation (FG-MOS/PAMS). A React research website brings both workflows, dataset inspection, retrieval evidence, and evaluation together. No paper or slide changes are part of this implementation.

## Run frontend and backend together with Docker

Start Docker Desktop, then run from the project directory:

```powershell
docker compose up -d --build
```

Open http://localhost:8180. Docker installs frontend dependencies, builds React, and packages it with FastAPI. No local Node.js, separate frontend terminal, or manual frontend build is required. The same origin serves all website routes and API calls.

Keep your existing `.env`, `data/` (N24 and BBC), and `runs/` in the project folder. Keys are supplied only at container runtime and excluded from the image build. Dataset files and saved experiments are mounted read-only; BERT downloads persist in a Docker volume.

```powershell
docker compose logs -f backend  # logs for the combined app
docker compose down             # stop the app
docker compose up -d            # start the previously built image
```

After code changes use `docker compose up -d --build` again. After `.env` changes use `docker compose up -d --force-recreate`. Stop a terminal backend using port 8180 before starting Docker. Alternatively set `$env:META_RAAD_PORT=8181` before the Compose command and open http://localhost:8181.

## Run on this Windows machine

From the project directory, build the frontend once (Node.js 22.12+):

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

Then start the application:

```powershell
./scripts/start.ps1
```

Open http://127.0.0.1:8180 for the new research website. Use **Anomaly Detection** and **Model Selection** in the sidebar. The legacy selection page remains at `/model-selection`. API documentation is at `/docs`. The current `.env` selects NVIDIA Nemotron 3 Super (free) through OpenRouter for both detection variants and model selection. Gemini is not used by this configuration. Existing credentials remain in the ignored file.

For a fresh installation (Python 3.11+):

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
# Create .env using the OpenRouter configuration below before starting.
./.venv/Scripts/python.exe scripts/download_data.py
./.venv/Scripts/python.exe scripts/download_data.py --dataset bbc
./scripts/start.ps1
```

Set the provider, model name, and matching API key in `.env`. For a new installation, create `.env` from the configuration block below; preserve an existing `.env`. Docker users can use `docker compose up -d --build` after starting Docker Desktop.

## Free hosted inference through OpenRouter

Create an API key at https://openrouter.ai/settings/keys and put it in the project-root `.env` (never in frontend code):

```dotenv
LLM_PROVIDER=openrouter
LLM_MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_FREE_ONLY=true
OPENROUTER_API_KEY=your-key-here
LLM_TIMEOUT_SECONDS=60
```

Restart the backend with `./scripts/start.ps1` after saving. Stop the existing backend first if port 8180 is occupied. No Ollama installation or model download is needed. Both comparison variants and model selection use the configured Nemotron model; BERT still supplies retrieval embeddings. Missing keys and API errors are reported without falling back to Gemini. Historical experiments keep their original model names; rerun them for Nemotron results.

The selected endpoint has zero token prices; free quotas and availability still apply. Free-only mode rejects non-free model IDs and caps provider prompt/completion prices at zero. Both comparison calls use the same fixed model. NVIDIA logs this free endpoint for security and service improvement: use public/non-sensitive text only. See https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free for terms. Each paired detection uses two LLM calls.

## Research website

Eight responsive pages cover Overview, Datasets, Vector Database, Anomaly Detection, Retrieval Explorer, Model Selection, Evaluation, and Settings. See [frontend/README.md](frontend/README.md) for development instructions and service boundaries.

Live mode is the default and reads the actual backend. **Settings → Illustrative preview** explicitly enables synthetic fixtures, with no silent fallback. File inspection, field mapping, validation, record previews, and exports work locally. Dataset registration and FAISS/Chroma index building currently have **preview-only** adapters; the live memory is NumPy exact cosine with BERT embeddings.

The new inference requests hide reference categories and exclude normalized exact query matches. Saved CLI runs using category-tagged references are marked legacy in Evaluation. Uploaded preview data and current results are session-only; export important results before reloading or changing modes.

## Anomaly detection

Enter a text, choose normal-only or normal-plus-anomaly category knowledge, and select the number of nearest training examples. Detection returns a score, threshold-based verdict, explanation, retrieved document IDs, and validated citation indices. The website always runs a paired comparison using the same LLM without retrieved examples. Both scores, classifications, explanations, and actual prompts are displayed side by side. Batch results and exports include both methods; the current-session Evaluation link evaluates the RAG scores.

The N24 retrieval memory contains only normal training articles. Anomaly category knowledge does not add anomalous examples to memory. BERT query encoding uses the CLS vector with a 512-token limit. Stored and query vectors are L2-normalized for cosine retrieval. Zero, nonfinite, or incorrectly sized vectors are rejected. Malformed, nonnumeric, nonfinite, and out-of-range LLM scores remain errors; they are never clamped into valid predictions. A 0.5 demo threshold is not a calibrated operating point.

RA-ZAD uses in-context examples and no weight updates. Despite the inherited name, it is not strict example-free zero-shot inference. The no-retrieval condition is the zero-shot baseline.

## Model selection

Open **Model selection**, load the training profile, select candidate detectors and a condition, then click **Recommend model**. The LLM returns a complete ranking, a brief explanation, structured feature citations, and the top recommendation.

Eight cards cover Isolation Forest, LOF, ECOD, Autoencoder, VAE, DeepSVDD, LUNAR, and SO-GAAL. Four conditions expose basic descriptions, dataset features only, model cards only, or both. The basic-description condition is an internal ablation, not an exact reproduction of the upstream AD-LLM model-selection prompt. Capability cards cite research sources and express hypotheses, not guaranteed performance rules.

Profiles compute text-length and vocabulary statistics, category entropy, normal-category compactness and separation, silhouette, effective dimension, and neighbor-distance variation. Expensive calculations use a reproducible sample of up to 1,000 training records by default. Embedding dimensions come from the actual files. Normal-versus-anomaly separation and test anomaly prevalence are withheld because normal-only training data cannot establish them.

G-score is the fraction of supplied structured feature citations that match available numeric measurements (1% relative tolerance or 0.0001 absolute tolerance). Duplicate, unavailable, unknown, or incorrect citations do not count. Empty evidence scores zero. Feature coverage is reported separately. This checks numeric citation precision, not the logical validity or causal faithfulness of an explanation. A well-cited recommendation can still choose a worse detector.

## Reproducible evaluation

The app registers N24 News and BBC News. Both BERT and precomputed OpenAI embeddings are supported by the experiment and selection pipelines; the OpenAI files must be downloaded with `scripts/download_data.py --openai` before use. Interactive typed-text retrieval uses BERT.

Install evaluation dependencies:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-research.txt
```

Local CPU baselines:

```powershell
./.venv/Scripts/python.exe scripts/run_experiment.py baselines --out runs/local --max-train 2000 --max-test 1000 --models iforest lof knn
```

LLM anomaly detection, interleaving baseline and retrieval calls for each row:

```powershell
./.venv/Scripts/python.exe scripts/run_experiment.py detect --out runs/detection --max-train 2000 --max-test 100 --ks 0 3 --setting normal_anomaly --allow-api --max-calls 20
```

Repeat the identical command to continue pending calls. `--max-calls` caps logical calls per invocation; a detection call may retry a rate limit twice. Saved errors remain recorded rather than silently retried or discarded. Provider billing and rate limits depend on your account. Inspect the saved errors and use a new run after changing configuration. `--max-train 0 --max-test 0` uses full splits and can be expensive.

Evaluate model recommendations against the same candidate training configuration:

```powershell
./.venv/Scripts/python.exe scripts/run_experiment.py select --out runs/selection --max-train 2000 --models iforest lof --conditions baseline dataset_only models_only full --candidate-results runs/local/summary.json --allow-api --max-calls 4
```

To rank all eight candidates, pass all their IDs with `--models iforest lof ecod autoencoder vae deepsvdd lunar so_gaal`. Their rankings work independently of local detector training. Benchmarking the six PyOD candidates requires a working PyOD/Numba/PyTorch installation. On this machine Windows Application Control blocked a Numba DLL, so those local training adapters have not been executed here. The app and the sklearn baselines work without importing PyOD; security settings were not changed.

Each run saves a manifest with configuration, data SHA-256 hashes, source hashes and library versions; exact sampled rows; the training profile; predictions or recommendation responses; errors; and a summary. Reusing an output directory with changed data, code, or configuration is rejected. Use a new directory after edits. Candidate-result evaluation verifies an identical training-data signature before computing rank and AUROC regret.

Evaluation removes normalized exact-text overlap with training data and duplicate test texts before stratified sampling. AUROC and average precision (reported as AUPRC/AP) are calculated only on finite scores with coverage and failure counts. One-class subsets correctly have undefined AUROC/AP. Detection comparisons additionally report metrics on common successful rows. Confidence intervals use a seeded stratified bootstrap. They describe sampling uncertainty for that run, not variation across models, prompts or seeds. No test score is supplied to the selection prompt.

## Tests and implementation map

```powershell
./.venv/Scripts/python.exe -m pytest -q
```

- `backend/app/services/detector.py`, `memory.py`, `bert_embedder.py`: anomaly detection.
- `profiling.py`, `model_selection.py`: training features, cards, recommendations, and evidence checks.
- `research_data.py`, `baselines.py`, `evaluation.py`: validated data and held-out evaluation.
- `backend/app/api/selection.py`: profile, cards, and recommendation endpoints.
- `scripts/run_experiment.py`: local and LLM experiment commands.

Data source: [NLP-ADBench](https://huggingface.co/datasets/kendx/NLP-ADBench). Base work: [AD-LLM code](https://github.com/USC-FORTIS/AD-LLM) and [published paper](https://aclanthology.org/2025.findings-acl.79/). Downloaded benchmark text is not committed or redistributed by this project.

## BBC and N24 dataset selection

Choose **N24 News** or **BBC News** using the website dataset selector. Each dataset has separate normal-training reference memory, profiles, prompts, and evaluation runs. Switching clears current predictions and recommendations.

| Dataset | Normal categories | Anomaly category | Training | Test |
|---|---|---|---:|---:|
| N24 News | 23 news categories | Food | 40,595 | 19,227 |
| BBC News | business, sport, politics, tech | entertainment | 1,206 | 579 |

BBC test has 517 normal and 62 anomalous examples. The original benchmark has one exact normalized train/test overlap. Source files remain unchanged; the experiment runner removes it (578 eligible test records). Interactive retrieval excludes exact query matches, and batch-upload overlap checks remain enforced.

Download BBC text and BERT vectors with `./.venv/Scripts/python.exe scripts/download_data.py --dataset bbc`. No Hugging Face key is required. Run BBC evaluation with:

```powershell
./.venv/Scripts/python.exe scripts/run_experiment.py baselines --dataset bbc --out runs/bbc_local_v1 --max-train 0 --max-test 0 --models iforest lof knn
```

Source: https://huggingface.co/datasets/kendx/NLP-ADBench/tree/main/datasets/bbc
