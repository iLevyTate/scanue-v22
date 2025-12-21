from typing import Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

class LLMFactory:
    """Factory for creating LLM instances based on configuration.
    
    Supports multiple providers including OpenAI, Ollama, and HuggingFace.
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
                
        Returns:
            LangChain ChatModel instance
        """
        provider = config.get("provider", "openai").lower()
        model_name = config.get("name")
        temperature = config.get("temperature", 0.7)
        
        if not model_name:
            raise ValueError("Model name must be specified in configuration")
            
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                timeout=config.get("timeout", 30.0),
                max_retries=config.get("max_retries", 3)
            )
            
        elif provider == "ollama":
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=config.get("base_url", "http://localhost:11434"),
                timeout=config.get("timeout", 120.0)  # Local models might be slower
            )
            
        elif provider == "huggingface":
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

