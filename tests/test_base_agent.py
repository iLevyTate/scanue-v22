import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.base import BaseAgent, AGENT_LLM_TIMEOUT_SECONDS
from workflow import NODE_TIMEOUT_SECONDS
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import asyncio

class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    __test__ = False  # Not a pytest test class; opts out of test collection.

    def __init__(self):
        super().__init__(agent_name="TEST", model_env_key="TEST_MODEL")
        
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("Test prompt: {task}")
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state)

@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {
        "TEST_MODEL": "test-model",
        "OPENAI_API_KEY": "test-key"
    }):
        yield

@pytest.fixture
def test_agent(mock_env_vars):
    return TestAgent()

@pytest.fixture
def test_state():
    return {
        "task": "test task",
        "stage": "test_stage",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }

@pytest.mark.asyncio
async def test_base_agent_initialization(test_agent):
    """Test base agent initialization"""
    assert isinstance(test_agent, BaseAgent)
    assert hasattr(test_agent, 'process')
    assert hasattr(test_agent, '_create_prompt')

@pytest.mark.asyncio
async def test_base_agent_process(test_agent, test_state):
    """Test base agent process method"""
    mock_response = AsyncMock()
    mock_response.content = "test response"
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_response)):
        result = await test_agent.process(test_state)
        assert isinstance(result, dict)
        # Handle structured response
        response_text = result["response"]["content"] if isinstance(result["response"], dict) else str(result["response"])
        assert response_text == "test response"
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_base_agent_validation(test_agent):
    """Test state validation"""
    invalid_state = {"invalid": "state"}
    with patch("langchain_openai.ChatOpenAI.ainvoke") as mock_ainvoke:
        mock_ainvoke.side_effect = ValueError("Invalid state format")
        result = await test_agent.process(invalid_state)
        assert result["error"]
        # Handle structured response
        response_text = result["response"]["content"] if isinstance(result["response"], dict) else str(result["response"])
        assert "Invalid state format" in response_text

@pytest.mark.asyncio
async def test_base_agent_timeout(test_agent, test_state):
    """Test timeout handling"""
    async def mock_process(*args, **kwargs):
        await asyncio.sleep(1)
        return None

    test_agent.process = mock_process
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):
            await test_agent.process(test_state)

@pytest.mark.asyncio
async def test_base_agent_error_handling(test_agent, test_state):
    """Test error handling"""
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=ValueError("Test error")):
        result = await test_agent.process(test_state)
        assert result["error"]
        # Handle structured response
        response_text = result["response"]["content"] if isinstance(result["response"], dict) else str(result["response"])
        assert "Test error" in response_text

@pytest.mark.asyncio
async def test_base_agent_cancellation_propagates(test_agent, test_state):
    """CancelledError is a BaseException and must propagate for cooperative
    cancellation to work -- it is no longer swallowed into an error result."""
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            await test_agent.process(test_state)


def test_outer_node_timeout_exceeds_inner_llm_timeout():
    """C4 guard: the outer per-node timeout must stay strictly greater than the
    inner LLM timeout so they never race."""
    assert NODE_TIMEOUT_SECONDS > AGENT_LLM_TIMEOUT_SECONDS
