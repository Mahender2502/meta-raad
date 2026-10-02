"""
Application settings for the RAD-LLM backend.

Values are loaded from environment variables (populated via the .env file
referenced in docker-compose.yml). This gives the RAG endpoint, the LLM
client factory, and later self-consistency / explanation-judge logic a
single place to read configuration from.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service metadata ---
    app_name: str = "RAD-LLM Backend"
    app_version: str = "0.1.0"

    # --- ChromaDB connection (overridden to 'chromadb' inside docker-compose) ---
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_prefix: str = "rad_llm"

    # --- Embedding model for retrieval (RAG-AD) ---
    # BAAI/bge-base-en-v1.5 via sentence-transformers. BGE models are trained
    # with an instruction prefix on the *query* side only (not on documents) —
    # see EMBEDDING_QUERY_INSTRUCTION below, used by app/services/embedder.py.
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_device: str = "cpu"

    # --- LLM client (provider-agnostic — see app/services/llm_client.py) ---
    # Which concrete LLMClient implementation the factory should build. Left
    # unset ("stub") until we decide on GPT-4o / DeepSeek / Llama etc.; the
    # RAG pipeline itself does not change when this is swapped.
    llm_provider: str = "stub"
    llm_model_name: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # --- LLM API keys (read by whichever provider client is selected) ---
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    huggingface_api_key: str | None = None


settings = Settings()
