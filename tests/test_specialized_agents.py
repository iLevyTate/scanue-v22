import pytest
from unittest.mock import patch, AsyncMock
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
from typing import Dict, Any, Type
import asyncio
from agents.base import BaseAgent # Import BaseAgent for type hinting

@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {
        "VMPFC_MODEL": "vmpfc-model",
        "OFC_MODEL": "ofc-model",
        "ACC_MODEL": "acc-model",
        "MPFC_MODEL": "mpfc-model",
        "OPENAI_API_KEY": "test-key"
    }):
        yield

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

@pytest.fixture
def mock_llm():
    async def mock_ainvoke(*args, **kwargs):
        mock_response = AsyncMock()
        mock_response.content = "test response"
        return mock_response
    
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=mock_ainvoke):
        yield

@pytest.mark.parametrize("agent_class", [
    VMPFCAgent,
    OFCAgent,
    ACCAgent,
    MPFCAgent,
])
@pytest.mark.asyncio
async def test_specialized_agent_process(agent_class: Type[BaseAgent], mock_env_vars, test_state, mock_llm):
    """Test specialized agent processing using mock_llm fixture"""
    agent = agent_class()
    # mock_llm fixture is automatically used here due to dependency injection
    result = await agent.process(test_state)
    assert isinstance(result, dict)
    assert "response" in result
    # Ensure the mock response content is checked
    assert result["response"] == "test response"
    assert not result.get("error", False)

@pytest.mark.asyncio
async def test_agent_error_handling(mock_env_vars, test_state):
    """Test error handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]

    for agent in agents:
        # Remove the try...except block, rely on agent's internal handling
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=ValueError("Test error")):
            result = await agent.process(test_state)
            assert result["error"] # Check if the agent correctly flagged the error
            # Optionally, check if the error message is propagated
            assert "error" in result["response"].lower()
            assert "Test error" in result["response"] # Be more specific if possible

@pytest.mark.asyncio
async def test_agent_timeout_handling(mock_env_vars, test_state):
    """Test timeout handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]

    for agent in agents:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.TimeoutError("Request timed out. Please try again.")):
            result = await agent.process(test_state)
            assert result["error"]
            assert "timed out" in result["response"].lower()
            assert "Request timed out" in result["response"]

@pytest.mark.asyncio
async def test_agent_cancellation_handling(mock_env_vars, test_state):
    """Test cancellation handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]

    for agent in agents:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.CancelledError("Test cancellation")):
            result = await agent.process(test_state)
            assert result["error"]
            assert "cancelled" in result["response"].lower()

@pytest.mark.asyncio
async def test_agent_initialization(mock_env_vars):
    test_cases = [
        (VMPFCAgent(), "VMPFC_MODEL", "vmpfc-model"),
        (OFCAgent(), "OFC_MODEL", "ofc-model"),
        (ACCAgent(), "ACC_MODEL", "acc-model"),
        (MPFCAgent(), "MPFC_MODEL", "mpfc-model")
    ]
    
    for agent, env_key, expected_model in test_cases:
        assert agent.llm.model_name == expected_model
