# =============================================================================
# Meta-RAAD Backend: FastAPI Service
#
# This is the backend service's own image — separate from the root
# Dockerfile (which builds the heavier research/experiment environment with
# torch, jupyter, pyod, etc.). This image only installs what the FastAPI
# service needs to run: fastapi/uvicorn, NumPy, BERT (transformers + torch),
# and the OpenAI SDK (used for any OpenAI-compatible LLM provider).
#
# Build (standalone):  docker build -t rad-llm-backend ./backend
# Build (via compose):  docker compose build backend
# Run (via compose):    docker compose up backend
# =============================================================================

FROM python:3.11-slim

# ---- System dependencies ----------------------------------------------------
# build-essential is needed for some packages that compile C extensions
# in some dependencies; curl is handy for container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies -----------------------------------------------------
# Copied first so Docker can cache this layer independently of app code
# changes.
COPY requirements.txt .
# CPU-only PyTorch first: the default wheel pulls several GB of CUDA libraries
# that this container (bert_device=cpu) never uses. transformers then sees
# torch as already installed.
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code --------------------------------------------------------
COPY . .

# ---- Networking ---------------------------------------------------------------
EXPOSE 8080

# ---- Entrypoint ----------------------------------------------------------------
# --reload is intended for development (paired with the bind-mounted volume
# in docker-compose.yml). Remove --reload for a production image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
