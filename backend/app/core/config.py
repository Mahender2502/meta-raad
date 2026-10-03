"""
Application settings for the Meta-RAAD backend.

Values are loaded from environment variables (populated via the .env file
referenced in docker-compose.yml). This gives the retrieval memory, the LLM
client factory, and later FG-MOS and G_score logic a single place to read
configuration from.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service metadata ---
    app_name: str = "Meta-RAAD Backend"
    app_version: str = "0.1.0"

    # --- Retrieval memory: the benchmark's precomputed BERT vectors + train text ---
    # Files live in <data_dir>/<dataset subdir>/ (see app/core/datasets.py); in the
    # container ./data is mounted at /data. Typed text is embedded with the same
    # model and recipe that produced the stored vectors (CLS token, max 512 tokens).
    data_dir: str = "/data"
    bert_model: str = "bert-base-uncased"
    bert_device: str = "cpu"

    # --- Dashboard data (JSON files, see app/services/stats.py) ---
    # Inside the container ./backend is mounted at /app, so /app/data is backend/data on the host.
    stats_dir: str = "/app/data"

    # --- LLM client (provider-agnostic — see app/services/llm_client.py) ---
    # Which concrete LLMClient implementation the factory should build
    # ("stub", "gemini", "groq", "openai", ...); detection logic does not change
    # when this is swapped.
    llm_provider: str = "stub"
    llm_model_name: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # --- LLM API keys (read by whichever provider client is selected) ---
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    huggingface_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    cerebras_api_key: str | None = None


settings = Settings()
