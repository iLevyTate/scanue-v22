"""Tests for `agents.factory.LLMFactory`."""

import sys
from unittest.mock import patch

import pytest

from agents.factory import LLMFactory


def test_ollama_timeout_reaches_the_http_client():
    """C13 regression: `timeout=` was silently dropped.

    ChatOllama declares no `timeout` field and its model_config sets
    extra="ignore", so the kwarg vanished and local runs had no client-side
    timeout whatsoever -- only the 45s outer node guard. It must go through
    client_kwargs, which the underlying ollama client forwards to httpx.
    """
    llm = LLMFactory.create_llm(
        {"provider": "ollama", "name": "llama3.2", "timeout": 7.5}
    )

    assert llm.client_kwargs == {"timeout": 7.5}
    # Assert on the real client, not just the pydantic field.
    assert llm._client._client.timeout.read == 7.5
    assert llm._async_client._client.timeout.read == 7.5


def test_ollama_timeout_defaults_to_120s():
    llm = LLMFactory.create_llm({"provider": "ollama", "name": "llama3.2"})
    assert llm.client_kwargs == {"timeout": 120.0}


def test_ollama_base_url_default():
    llm = LLMFactory.create_llm({"provider": "ollama", "name": "llama3.2"})
    assert llm.base_url == "http://localhost:11434"


def test_openai_provider_is_the_default():
    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        LLMFactory.create_llm({"name": "gpt-4o-mini", "api_key": "k"})

    assert chat_openai.call_args.kwargs["model"] == "gpt-4o-mini"
    assert chat_openai.call_args.kwargs["timeout"] == 30.0
    assert chat_openai.call_args.kwargs["max_retries"] == 3


def test_missing_model_name_is_rejected():
    with pytest.raises(ValueError, match="Model name must be specified"):
        LLMFactory.create_llm({"provider": "ollama"})


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider: banana"):
        LLMFactory.create_llm({"provider": "banana", "name": "x"})


def test_provider_sdks_are_imported_lazily():
    """C14: importing all three SDKs at module scope forced an Ollama-only user
    to install langchain-openai and langchain-huggingface as well."""
    import agents.factory as factory

    for attr in ("ChatOpenAI", "ChatOllama", "HuggingFaceEndpoint", "ChatHuggingFace"):
        assert not hasattr(factory, attr), f"{attr} should not be a module-level import"


def test_ollama_config_does_not_need_huggingface_installed(monkeypatch):
    """Building an Ollama model must not touch the HuggingFace SDK."""
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)

    llm = LLMFactory.create_llm({"provider": "ollama", "name": "llama3.2"})
    assert llm.model == "llama3.2"
