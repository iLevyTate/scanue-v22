"""Shared fixtures.

`mock_env_vars` was redefined in five test modules with near-identical bodies and
`TEST_CONFIG` in two; the shared ones live here.
"""

import pytest
from dotenv import load_dotenv

from utils.config import ConfigLoader

# A config where every agent uses a cheap, offline-safe OpenAI stub. Individual
# modules override this when they need provider-specific behaviour.
AGENT_NAMES = ("DLPFC", "VMPFC", "OFC", "ACC", "MPFC")
TEST_CONFIG = {
    "agents": {
        name: {"models": {"primary": {"provider": "openai", "name": "test-model"}}}
        for name in AGENT_NAMES
    }
}


@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables before each test"""
    load_dotenv()


@pytest.fixture(autouse=True)
def mock_openai_key(monkeypatch):
    """Ensure OPENAI_API_KEY is available for tests.

    Uses monkeypatch rather than assigning os.environ directly: the previous
    version permanently mutated the process environment for the whole session
    with no teardown, unlike every other env fixture in the suite.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the ConfigLoader cache around each test.

    Tests patch the config path/contents in different ways; clearing the cached
    config before and after keeps them isolated from one another.
    """
    ConfigLoader.reset()
    yield
    ConfigLoader.reset()
