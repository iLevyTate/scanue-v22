import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader for SCANUE-V agents.
    
    Handles loading agent configurations from YAML files and managing
    environment variable fallbacks for backward compatibility.
    """
    
    _config: Optional[Dict[str, Any]] = None
    _config_path: Path = Path("config/agents.yaml")
    
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Load the agent configuration.
        
        Returns:
            Dict containing the full agent configuration
        """
        if cls._config is None:
            if cls._config_path.exists():
                with open(cls._config_path, 'r') as f:
                    cls._config = yaml.safe_load(f)
            else:
                cls._config = {"agents": {}}
        
        return cls._config

    @classmethod
    def get_agent_config(cls, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "DLPFC", "VMPFC")
            
        Returns:
            Dict containing model configurations for the agent
        """
        config = cls.load_config()
        agent_config = config.get("agents", {}).get(agent_name, {})
        
        # Ensure 'models' key exists
        if "models" not in agent_config:
            agent_config["models"] = {}
            
        return agent_config

    @classmethod
    def get_model_config(cls, agent_name: str, model_type: str = "primary", env_var_fallback: str = None) -> Dict[str, Any]:
        """Get specific model configuration with environment variable fallback.
        
        Args:
            agent_name: Name of the agent
            model_type: Type of model (e.g., "primary", "fast")
            env_var_fallback: Legacy environment variable name to check if config is missing
            
        Returns:
            Dict with 'provider', 'name', and other model settings
        """
        agent_config = cls.get_agent_config(agent_name)
        model_config = agent_config.get("models", {}).get(model_type)
        
        # If configuration exists, return it
        if model_config:
            return model_config
            
        # Fallback to environment variable if provided
        if env_var_fallback:
            env_model = os.getenv(env_var_fallback)
            if env_model:
                # Default to OpenAI for backward compatibility if using env vars
                return {
                    "provider": "openai",
                    "name": env_model,
                    "temperature": 0.7  # Default temperature
                }
                
        # Final fallback default
        return {
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "temperature": 0.7
        }

