from typing import Dict, Any, Optional
import os


class LLMFactory:
    """Factory for creating LLM instances based on configuration.

    Supports multiple providers including OpenAI, Ollama, and HuggingFace.

    Provider SDKs are imported lazily inside each branch. Importing all three at
    module scope forced an Ollama-only user to install langchain-openai and
    langchain-huggingface (and its transformers/tokenizers dependency chain)
    just to start the app.
    """

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> Any:
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

            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                timeout=config.get("timeout", 30.0),
                max_tokens=config.get("max_tokens"),
                max_retries=config.get("max_retries", 3)
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            # ChatOllama has no `timeout` field and its model_config sets
            # extra="ignore", so passing timeout= was silently dropped and local
            # runs had no client-side timeout at all. The underlying ollama
            # client takes it via client_kwargs.
            kwargs = dict(
                model=model_name,
                temperature=temperature,
                base_url=config.get("base_url", "http://localhost:11434"),
                client_kwargs={"timeout": config.get("timeout", 120.0)}  # Local models might be slower
            )

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
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            # Use HuggingFaceEndpoint for inference API or local TGI
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=temperature,
                huggingfacehub_api_token=config.get("api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                timeout=config.get("timeout", 120.0)
            )
            return ChatHuggingFace(llm=llm)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
