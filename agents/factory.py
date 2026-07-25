import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on configuration.

    Supports multiple providers including OpenAI, Ollama, and HuggingFace.

    Provider SDKs are imported lazily inside each branch. Importing all three at
    module scope forced an Ollama-only user to install langchain-openai and
    langchain-huggingface (and its transformers/tokenizers dependency chain)
    just to start the app.
    """

    @staticmethod
    def wrap_with_retry(runnable: Any, config: dict[str, Any]) -> Any:
        """Wrap a runnable in retry-with-exponential-backoff.

        `max_retries` is an OpenAI-only *constructor* argument -- neither
        ChatOllama nor HuggingFaceEndpoint declares such a field -- so with the
        shipped all-Ollama config the app had NO retry on any failure: a
        one-second blip on a local server failed every stage of the run in turn.
        `.with_retry()` is a Runnable-level feature that works for all providers.

        Applied at invocation rather than inside `create_llm` on purpose:
        `.with_retry()` returns a `RunnableRetry`, which is not a chat model and
        would break `model_name`/`temperature` introspection and, critically,
        `with_structured_output()`. Callers keep the real model and wrap only the
        thing they are about to invoke.

        OpenAI is skipped because its client already retries natively (with
        rate-limit awareness); wrapping it too would multiply the attempts.
        Set `max_retries: 0` to disable.
        """
        if (config or {}).get("provider", "openai").lower() == "openai":
            return runnable

        attempts = (config or {}).get("max_retries", 3)
        if not attempts or attempts < 1:
            return runnable

        try:
            return runnable.with_retry(
                # `attempts` counts RETRIES, so add the initial try.
                stop_after_attempt=attempts + 1,
                wait_exponential_jitter=True,
            )
        except Exception as e:
            # Any object that is not a full LangChain Runnable. Retrying is a
            # nice-to-have; returning something unusable here would break the
            # call entirely, which is strictly worse than not retrying.
            logger.warning("Could not apply retry policy (%s); invoking without it", e)
            return runnable

    @staticmethod
    def create_llm(config: dict[str, Any]) -> Any:
        """Create an LLM instance based on the provided configuration.

        Args:
            config: Dictionary containing model configuration:
                - provider: 'openai', 'ollama', or 'huggingface'
                - name: Model identifier (e.g., 'gpt-4', 'llama3')
                - temperature: Model temperature (default: 0.7)
                - base_url: Optional base URL for local providers
                - api_key: Optional API key (overrides env vars)
                - timeout: Request timeout in seconds
                - max_retries: Retry count (OpenAI only)

        Returns:
            LangChain ChatModel instance
        """
        config = config or {}
        provider = config.get("provider", "openai").lower()
        model_name = config.get("name")
        temperature = config.get("temperature", 0.7)

        if not model_name:
            raise ValueError("Model name must be specified in configuration")

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            # The ignores below cover max_tokens/api_key: both are accepted at
            # runtime (verified against the pinned langchain-openai); the
            # published signature just does not reflect it.
            return ChatOpenAI(  # type: ignore[call-arg]
                model=model_name,
                temperature=temperature,
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),  # type: ignore[arg-type]
                timeout=config.get("timeout", 30.0),
                max_tokens=config.get("max_tokens"),
                max_retries=config.get("max_retries", 3),
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            # ChatOllama has no `timeout` field and its model_config sets
            # extra="ignore", so passing timeout= was silently dropped and local
            # runs had no client-side timeout at all. The underlying ollama
            # client takes it via client_kwargs.
            kwargs = {
                "model": model_name,
                "temperature": temperature,
                "base_url": config.get("base_url", "http://localhost:11434"),
                # Local models might be slower.
                "client_kwargs": {"timeout": config.get("timeout", 120.0)},
            }

            # num_ctx defaults to None, which means the Ollama server applies its
            # own default (commonly 2048 tokens) and silently DROPS anything past
            # it -- no error, no log line, just a truncated prompt. Setting it
            # explicitly is the only way to know the model saw what we sent.
            if config.get("num_ctx") is not None:
                kwargs["num_ctx"] = config["num_ctx"]
            if config.get("max_tokens") is not None:
                # Ollama's name for max output tokens.
                kwargs["num_predict"] = config["max_tokens"]

            return ChatOllama(**kwargs)

        elif provider == "huggingface":
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            # Use HuggingFaceEndpoint for inference API or local TGI
            # `model` is declared required but a validator populates it from
            # repo_id; repo_id-only construction is verified to work.
            llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
                repo_id=model_name,
                temperature=temperature,
                huggingfacehub_api_token=config.get("api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                timeout=config.get("timeout", 120.0)
            )
            return ChatHuggingFace(llm=llm)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
