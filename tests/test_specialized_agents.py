import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents.base import BaseAgent  # Import BaseAgent for type hinting
from agents.specialized import ACCAgent, MPFCAgent, OFCAgent, VMPFCAgent


@pytest.fixture
def mock_env_vars():
    # Mock ConfigLoader to avoid reading real config file and return expected test models
    def mock_get_agent_config(agent_name):
        return {
            "models": {
                "primary": {"provider": "openai", "name": f"{agent_name.lower()}-model"}
            }
        }

    with patch.dict("os.environ", {
        "VMPFC_MODEL": "vmpfc-model",
        "OFC_MODEL": "ofc-model",
        "ACC_MODEL": "acc-model",
        "MPFC_MODEL": "mpfc-model",
        "OPENAI_API_KEY": "test-key"
    }), patch("utils.config.ConfigLoader.get_agent_config", side_effect=mock_get_agent_config):
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
async def test_specialized_agent_process(agent_class: type[BaseAgent], mock_env_vars, test_state, mock_llm):
    """Test specialized agent processing using mock_llm fixture"""
    agent = agent_class()
    # mock_llm fixture is automatically used here due to dependency injection
    result = await agent.process(test_state)
    assert isinstance(result, dict)
    assert "response" in result
    # Ensure the mock response content is checked
    # response is now a dict: {'role': 'assistant', 'content': 'test response'}
    assert result["response"]["content"] == "test response"
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
            # Handle structured response
            response_text = result["response"]["content"] if isinstance(result["response"], dict) else str(result["response"])
            assert "error" in response_text.lower()
            assert "Test error" in response_text # Be more specific if possible

@pytest.mark.asyncio
async def test_agent_timeout_handling(mock_env_vars, test_state):
    """Test timeout handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]

    for agent in agents:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=TimeoutError("Request timed out. Please try again.")):
            result = await agent.process(test_state)
            assert result["error"]
            # Handle structured response
            response_text = result["response"]["content"] if isinstance(result["response"], dict) else str(result["response"])
            assert "timed out" in response_text.lower()
            # The exact message might be wrapped or changed by BaseAgent error handler
            # BaseAgent returns: "Request timed out. Please try again."
            assert "Request timed out" in response_text

@pytest.mark.asyncio
async def test_agent_cancellation_propagates(mock_env_vars, test_state):
    """CancelledError (a BaseException) must propagate out of specialist agents
    for cooperative cancellation -- it is no longer swallowed into a result."""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]

    for agent in agents:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.CancelledError("Test cancellation")):
            with pytest.raises(asyncio.CancelledError):
                await agent.process(test_state)

@pytest.mark.asyncio
async def test_agent_initialization(mock_env_vars):
    test_cases = [
        (VMPFCAgent(), "VMPFC_MODEL", "vmpfc-model"),
        (OFCAgent(), "OFC_MODEL", "ofc-model"),
        (ACCAgent(), "ACC_MODEL", "acc-model"),
        (MPFCAgent(), "MPFC_MODEL", "mpfc-model")
    ]

    for agent, _env_key, expected_model in test_cases:
        assert agent.llm.model_name == expected_model
