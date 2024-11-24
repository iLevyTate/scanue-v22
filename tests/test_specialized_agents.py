import pytest
from unittest.mock import patch, AsyncMock
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
from typing import Dict, Any
import asyncio

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

@pytest.mark.asyncio
async def test_vmpfc_agent_initialization(mock_env_vars):
    """Test VMPFC agent initialization"""
    agent = VMPFCAgent()
    assert agent.llm.model_name == "vmpfc-model"

@pytest.mark.asyncio
async def test_vmpfc_agent_process(mock_env_vars, test_state):
    """Test VMPFC agent processing"""
    agent = VMPFCAgent()
    mock_response = AsyncMock()
    mock_response.content = "test response"
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_response)):
        result = await agent.process(test_state)
        assert isinstance(result, dict)
        assert "response" in result
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_ofc_agent_initialization(mock_env_vars):
    """Test OFC agent initialization"""
    agent = OFCAgent()
    assert agent.llm.model_name == "ofc-model"

@pytest.mark.asyncio
async def test_ofc_agent_process(mock_env_vars, test_state):
    """Test OFC agent processing"""
    agent = OFCAgent()
    mock_response = AsyncMock()
    mock_response.content = "test response"
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_response)):
        result = await agent.process(test_state)
        assert isinstance(result, dict)
        assert "response" in result
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_acc_agent_initialization(mock_env_vars):
    """Test ACC agent initialization"""
    agent = ACCAgent()
    assert agent.llm.model_name == "acc-model"

@pytest.mark.asyncio
async def test_acc_agent_process(mock_env_vars, test_state):
    """Test ACC agent processing"""
    agent = ACCAgent()
    mock_response = AsyncMock()
    mock_response.content = "test response"
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_response)):
        result = await agent.process(test_state)
        assert isinstance(result, dict)
        assert "response" in result
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_mpfc_agent_initialization(mock_env_vars):
    """Test MPFC agent initialization"""
    agent = MPFCAgent()
    assert agent.llm.model_name == "mpfc-model"

@pytest.mark.asyncio
async def test_mpfc_agent_process(mock_env_vars, test_state):
    """Test MPFC agent processing"""
    agent = MPFCAgent()
    mock_response = AsyncMock()
    mock_response.content = "test response"
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_response)):
        result = await agent.process(test_state)
        assert isinstance(result, dict)
        assert "response" in result
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_agent_error_handling(mock_env_vars, test_state):
    """Test error handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]
    
    for agent in agents:
        try:
            with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=ValueError("Test error")):
                result = await agent.process(test_state)
                assert result["error"]
                assert "error" in result["response"].lower()
        except ValueError as e:
            assert str(e) == "Test error"

@pytest.mark.asyncio
async def test_agent_timeout_handling(mock_env_vars, test_state):
    """Test timeout handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]
    
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(2)
        mock_response = AsyncMock()
        mock_response.content = "test response"
        return mock_response
    
    for agent in agents:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=slow_response):
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(1):
                    await agent.process(test_state)

@pytest.mark.asyncio
async def test_agent_cancellation_handling(mock_env_vars, test_state):
    """Test cancellation handling in specialized agents"""
    agents = [VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]
    
    for agent in agents:
        try:
            with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.CancelledError()):
                result = await agent.process(test_state)
                assert result["error"]
                assert "cancelled" in result["response"].lower()
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_vmpfc_agent(mock_env_vars, mock_llm):
    agent = VMPFCAgent()
    result = await agent.process({"task": "test"})
    assert result["response"] == "test response"

@pytest.mark.asyncio
async def test_ofc_agent(mock_env_vars, mock_llm):
    agent = OFCAgent()
    result = await agent.process({"task": "test"})
    assert result["response"] == "test response"

@pytest.mark.asyncio
async def test_acc_agent(mock_env_vars, mock_llm):
    agent = ACCAgent()
    result = await agent.process({"task": "test"})
    assert result["response"] == "test response"

@pytest.mark.asyncio
async def test_mpfc_agent(mock_env_vars, mock_llm):
    agent = MPFCAgent()
    result = await agent.process({"task": "test"})
    assert result["response"] == "test response"

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
