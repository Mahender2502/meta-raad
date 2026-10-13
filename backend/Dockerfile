# Build both frontend and backend from the repository root.
# docker build -f backend/Dockerfile -t meta-raad .

FROM node:22-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

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
COPY backend/requirements.txt .
# CPU-only PyTorch first: the default wheel pulls several GB of CUDA libraries
# that this container (bert_device=cpu) never uses. transformers then sees
# torch as already installed.
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code --------------------------------------------------------
COPY backend/ .
COPY --from=frontend-build /build/frontend/dist /frontend/dist

# ---- Networking ---------------------------------------------------------------
EXPOSE 8080

# ---- Entrypoint ----------------------------------------------------------------
# Serve the compiled website and API from one production process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://127.0.0.1:8080/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
