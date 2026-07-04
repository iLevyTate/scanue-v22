import pytest
import os
from dotenv import load_dotenv
from utils.config import ConfigLoader

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables before each test"""
    load_dotenv()

@pytest.fixture(autouse=True)
def mock_openai_key():
    """Ensure OPENAI_API_KEY is available for tests"""
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test-key"

@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the ConfigLoader cache around each test.

    Tests patch the config path/contents in different ways; clearing the cached
    config before and after keeps them isolated from one another.
    """
    ConfigLoader.reset()
    yield
    ConfigLoader.reset()
