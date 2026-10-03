# Setup guide

How to run the Meta-RAAD demo and dashboard on your own machine. Takes about 15 minutes, most of it downloading.

## What you need
- **Docker Desktop**, running.
- **Python 3.9 or newer** (only for the data download script).
- **A free Gemini API key**: https://aistudio.google.com/apikey (needs a Google account).
- Internet access, and about **3 GB** of free disk space (Docker image + data).

## Steps

### 1. Clone
```bash
git clone https://github.com/Mahender2502/meta-raad.git
cd meta-raad
git checkout feature/meta-raad-spec        # the branch with this work
```

### 2. Create your `.env` file
`.env` holds your API key and is **not** in the repo (it is git-ignored), so every person creates their own.
```bash
cp .env.example .env
```
Open `.env` and set these two lines (leave the rest as it is):
```
GEMINI_API_KEY=your-key-here
LLM_MODEL_NAME=gemini-3.5-flash-lite
```
Never commit `.env` or paste your key into chat or issues.

### 3. Download the data (about 450 MB)
```bash
python3 scripts/download_data.py
```
This fetches the N24 News articles and their precomputed BERT vectors from Hugging Face
([kendx/NLP-ADBench](https://huggingface.co/datasets/kendx/NLP-ADBench)) into `data/n24/`.
It resumes if interrupted and skips files you already have. At the end it should print:
```
  train: 40595 articles, 40595 vectors -> ok
  test: 19227 articles, 19227 vectors -> ok
  train labels: [0] (expected [0], normal only)
```
`data/` is git-ignored, so it is never committed.

### 4. Build and start
```bash
docker compose up -d --build
```
The first build takes a few minutes (it installs PyTorch and other packages).

### 5. Open it
| Page | Address |
|---|---|
| Demo | http://localhost:8180 |
| Dashboard | http://localhost:8180/dashboard |
| Health check | http://localhost:8180/health |

**The first "Detect" click is slow** (about 30 to 60 seconds). The app downloads the BERT model (about 440 MB,
once, it is kept in a Docker volume) and loads the training data into memory. After that, a query takes a few seconds.
Run one query before showing the demo to anyone.

## Using it
- **Demo page:** type a text, choose a setting and k, click **Detect**. You get a verdict, a score, the reason, and the real training articles it was compared against (with ids like `trn_05590`). **Compare with baseline** runs the same text without retrieval.
- **Dashboard:** statistics of the N24 data (real), live statistics of the queries you run (they start at zero and update by themselves), and results panels that show "pending" until the evaluation runs exist.

## Everyday commands
```bash
docker compose logs -f backend      # watch the app's log
docker compose up -d --force-recreate backend   # after editing .env (a plain restart does not re-read it)
docker compose up -d --build        # after changing requirements.txt or the Dockerfile
docker compose down                 # stop (your data/ folder is kept)
```
Code changes in `backend/` reload automatically; no rebuild needed.

## Troubleshooting

| What you see | Cause and fix |
|---|---|
| `env file .env not found` when running `docker compose up` | You skipped step 2. Run `cp .env.example .env`. |
| Page opens, but Detect says **"LLM not configured"** | `GEMINI_API_KEY` is empty in `.env`. Set it, then run `docker compose up -d --force-recreate backend` (a plain `restart` does not re-read `.env`). |
| Detect says **"Retrieval data not found"** (HTTP 503) | `data/n24/` is empty. Run step 3. |
| Detect says **"Rate limit reached (free tier)"** | The free Gemini tier allows only a few requests per minute. Wait a minute. Each Detect uses 1 request (2 with the baseline comparison). |
| The result says the model name is not found | Use a model id that your key offers: set `LLM_MODEL_NAME` (see AI Studio's model list). |
| `port is already allocated` for 8180 | Another program uses that port. Change `"8180:8080"` in `docker-compose.yml`, e.g. to `"8190:8080"`. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it is running. |
| `docker compose` looks for a file such as `compose-override.yml` | A `COMPOSE_FILE` variable is set in your shell by another project. Run `unset COMPOSE_FILE` first. |
| The first query fails while downloading the BERT model | You were offline or Hugging Face was unreachable. Retry; once downloaded it works offline. |
| The BERT/PyTorch download times out during the build | A network hiccup. Run `docker compose up -d --build` again. |
| Download script: `CERTIFICATE_VERIFY_FAILED` | Your Python lacks CA certificates. Run `pip install certifi`, or on macOS run the "Install Certificates.command" that comes with python.org Python. |

## Using another LLM provider
Set `LLM_PROVIDER` in `.env` to `gemini` (default), `groq`, `cerebras`, `openai`, `deepseek` or `stub`, put the matching key in `.env`
(see the comments in `.env.example`) and set `LLM_MODEL_NAME` to a model that provider offers.
`stub` makes no LLM call, so the retrieval still works but answers show an error.

## Extra scripts (optional)
| Script | What it does |
|---|---|
| `scripts/download_data.py` | Fetches the N24 text and BERT vectors (step 3). `--splits train` for the train split only; `--openai` also fetches the large OpenAI vectors (not used by the app). |
| `scripts/embed_dataset.py` | Turns a folder of `.jsonl` files into BERT vectors (optional chunking). It reproduces the benchmark's vectors and does **not** produce the data the project uses. See `scripts/example/README.md`. Needs PyTorch, so run it in the backend image. |
| `scripts/make_dashboard_data.py` | Rebuilds the dashboard's real statistics from `data/n24/` (needs `numpy` and `scikit-learn`). |

## Where things are
```
data/n24/            downloaded data (git-ignored)
backend/app/         the FastAPI app (demo, retrieval, dashboard)
backend/data/        dashboard JSON files
scripts/             download, embedding and dashboard-data scripts
notes/               the base paper (AD-LLM) and notes on its prompts
```
For the background of the method, read the base paper in `notes/`.
