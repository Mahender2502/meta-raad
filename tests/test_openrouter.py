import pytest

from app.services import llm_client


def test_openrouter_uses_llama_and_dedicated_key(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "openrouter_free_only", False)
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(llm_client.settings, "llm_model_name", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setattr(llm_client.settings, "openrouter_api_key", "test-openrouter-key")
    # A previous endpoint must never receive the OpenRouter credential.
    monkeypatch.setattr(llm_client.settings, "llm_base_url", "https://example.invalid")
    monkeypatch.setattr(llm_client, "OpenAICompatibleClient", lambda **kwargs: kwargs)
    client = llm_client.get_llm_client()
    assert client == {
        "model_name": "meta-llama/llama-3.1-8b-instruct",
        "api_key": "test-openrouter-key",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_missing_openrouter_key_does_not_fall_back_to_gemini(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(llm_client.settings, "openrouter_api_key", None)
    monkeypatch.setattr(llm_client.settings, "gemini_api_key", "test-existing-key")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        llm_client.get_llm_client()


def test_free_only_rejects_paid_model(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(llm_client.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(llm_client.settings, "openrouter_free_only", True)
    monkeypatch.setattr(llm_client.settings, "llm_model_name", "qwen/qwen3-32b")
    with pytest.raises(RuntimeError, match="fixed model ID"):
        llm_client.get_llm_client()


def test_free_only_caps_price(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(llm_client.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(llm_client.settings, "openrouter_free_only", True)
    monkeypatch.setattr(llm_client.settings, "llm_model_name", "nvidia/nemotron-3-super-120b-a12b:free")
    monkeypatch.setattr(llm_client, "OpenAICompatibleClient", lambda **kwargs: kwargs)
    client = llm_client.get_llm_client()
    assert client["extra_body"]["provider"]["max_price"] == {"prompt": 0, "completion": 0}
