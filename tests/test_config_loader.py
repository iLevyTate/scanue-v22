"""Tests for `utils.config.ConfigLoader`.

Focus on shapes that hand-edited YAML actually produces -- in particular a key
that is present but null, which is what commenting out a block leaves behind.
"""

import pytest
from unittest.mock import patch

from utils.config import ConfigLoader


def _with_config(config):
    return patch.object(ConfigLoader, "load_config", return_value=config)


def test_null_models_block_is_coerced():
    """C4 regression: `models:` present-but-null must not crash.

    Commenting out an agent's model block leaves `models: None`. The old guard
    (`if "models" not in agent_config`) only filled the key when it was absent,
    so the next `.get()`/`.items()` raised AttributeError.
    """
    with _with_config({"agents": {"DLPFC": {"description": "x", "models": None}}}):
        assert ConfigLoader.get_agent_config("DLPFC")["models"] == {}


def test_null_agents_block_is_coerced():
    with _with_config({"agents": None}):
        assert ConfigLoader.get_agent_config("DLPFC")["models"] == {}


def test_null_agent_entry_is_coerced():
    with _with_config({"agents": {"DLPFC": None}}):
        assert ConfigLoader.get_agent_config("DLPFC")["models"] == {}


def test_missing_model_raises_actionable_error_not_attributeerror():
    """A null models block should surface the helpful ValueError, not a crash."""
    with _with_config({"agents": {"DLPFC": {"models": None}}}):
        with pytest.raises(ValueError, match="No model configuration found"):
            ConfigLoader.get_model_config("DLPFC", "primary")


def test_env_var_fallback_still_works():
    with _with_config({"agents": {"DLPFC": {"models": None}}}), \
         patch.dict("os.environ", {"DLPFC_MODEL": "gpt-4o-mini"}):
        cfg = ConfigLoader.get_model_config("DLPFC", "primary", env_var_fallback="DLPFC_MODEL")

    assert cfg == {"provider": "openai", "name": "gpt-4o-mini", "temperature": 0.7}


def test_get_agent_config_returns_a_deep_copy():
    """Callers must not be able to mutate the cached config."""
    config = {"agents": {"DLPFC": {"models": {"primary": {"name": "a"}}}}}
    with _with_config(config):
        got = ConfigLoader.get_agent_config("DLPFC")
        got["models"]["primary"]["name"] = "mutated"

    assert config["agents"]["DLPFC"]["models"]["primary"]["name"] == "a"
