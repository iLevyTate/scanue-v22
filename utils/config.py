import os
import copy
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Configuration loader for SCANUE-V agents.

    Handles loading agent configurations from YAML files and managing
    environment variable fallbacks for backward compatibility.
    """

    _config: Optional[Dict[str, Any]] = None
    # Resolve relative to this module, not the current working directory, so the
    # config is found regardless of where the process is launched from.
    _config_path: Path = Path(__file__).resolve().parent.parent / "config" / "agents.yaml"

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Load the agent configuration.

        Returns:
            Dict containing the full agent configuration

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if cls._config is None:
            if not cls._config_path.exists():
                raise FileNotFoundError(
                    f"Agent configuration file not found at: {cls._config_path}. "
                    "Create config/agents.yaml (see config/agents.example.yaml)."
                )
            with open(cls._config_path, 'r') as f:
                cls._config = yaml.safe_load(f) or {"agents": {}}

        return cls._config

    @classmethod
    def reset(cls) -> None:
        """Clear the cached configuration.

        Primarily useful for tests that patch the config path or contents and
        need a clean load on the next access.
        """
        cls._config = None

    @classmethod
    def get_agent_config(cls, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "DLPFC", "VMPFC")

        Returns:
            Dict containing model configurations for the agent. This is a deep
            copy of the cached config so callers can never mutate shared state.
        """
        config = cls.load_config()
        agent_config = copy.deepcopy(config.get("agents", {}).get(agent_name, {}))

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

        Raises:
            ValueError: If no configuration exists for the model and no env-var
                fallback is available. (There is intentionally no silent fallback
                to a paid OpenAI model -- a missing config is a hard error.)
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

        raise ValueError(
            f"No model configuration found for agent '{agent_name}' model '{model_type}'. "
            f"Add it to config/agents.yaml"
            + (f" or set the {env_var_fallback} environment variable." if env_var_fallback else ".")
        )
