import pytest
import os
from dotenv import load_dotenv
from agents.specialized import VMPFCAgent

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables before each test"""
    load_dotenv()

@pytest.fixture(autouse=True)
def mock_openai_key():
    """Ensure OPENAI_API_KEY is available for tests"""
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test-key"

@pytest.fixture
def vmpfc_agent(mock_env_vars):
    """Fixture for VMPFCAgent"""
    agent = VMPFCAgent()
    return agent
