"""
Provider-agnostic LLM client for the RAD-LLM backend.

We haven't decided which model(s) we'll ultimately use (GPT-4o, DeepSeek-V3,
Llama 3.1, etc.), so nothing outside this file should ever import a
provider SDK directly. Every other module (rag_service.py, and later
self-consistency / explanation-judge logic) talks only to the LLMClient
interface below and calls get_llm_client() to obtain an instance. Adding a
new provider means adding one class here and one branch in the factory —
it never touches RAG or detection logic.

Currently wired up:
    - "stub"   : deterministic, no external calls — lets the RAG endpoint
                 and its tests run end-to-end before a real model is chosen.
    - "openai" : minimal implementation using the OpenAI SDK (works for any
                 OpenAI-compatible chat-completions endpoint, including
                 DeepSeek's API, by overriding base_url).

To add a real provider client, implement generate() and register it in
get_llm_client() — no other file needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class LLMResponse:
    """Normalized response shape returned by every LLMClient implementation."""

    text: str
    model: str
    raw: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Common interface every concrete LLM provider client must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a single prompt to the LLM and return its response."""
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """
    No-op client used until a real model provider is chosen.

    Echoes back a canned response so the RAG endpoint is exercisable
    end-to-end (retrieval + prompt construction + "generation" + response
    shape) without any API key or model decision being required yet.
    """

    def __init__(self, model_name: str = "stub-echo") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        preview = prompt if len(prompt) <= 500 else f"{prompt[:500]}..."
        return LLMResponse(
            text=(
                "[stub-llm-client] No LLM provider configured yet "
                "(settings.llm_provider='stub'). This is the exact prompt "
                f"that would have been sent:\n\n{preview}"
            ),
            model=self.model_name,
            raw={"prompt_length": len(prompt)},
        )


class OpenAICompatibleClient(LLMClient):
    """
    Chat-completions client for OpenAI and any OpenAI-API-compatible
    provider (e.g. DeepSeek), selected via base_url/api_key.

    Kept intentionally minimal — real prompt/response handling for
    detection (score + explanation JSON parsing, retries, etc.) belongs in
    the detection module that calls this client, not here.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # local import: keeps the SDK optional for "stub" mode

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        )
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            raw=response.model_dump(),
        )


def get_llm_client() -> LLMClient:
    """
    Factory returning the configured LLMClient implementation.

    Selection is driven entirely by settings.llm_provider (env var
    LLM_PROVIDER), so switching models/providers is a config change, not a
    code change in the RAG pipeline or detection logic.
    """
    provider = settings.llm_provider.lower()

    if provider == "stub":
        return StubLLMClient(model_name=settings.llm_model_name or "stub-echo")

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAICompatibleClient(
            model_name=settings.llm_model_name or "gpt-4o",
            api_key=settings.openai_api_key,
        )

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set")
        return OpenAICompatibleClient(
            model_name=settings.llm_model_name or "deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected one of: stub, openai, deepseek "
        "(add new providers in app/services/llm_client.py)."
    )
