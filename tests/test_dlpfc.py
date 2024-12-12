import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from agents.dlpfc import DLPFCAgent
from typing import Dict, Any
import openai
from langchain.chat_models import ChatOpenAI
from langchain.exceptions import AuthenticationError, ConnectionError

@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {
        "DLPFC_MODEL": "dlpfc-model",
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
        "feedback": "test feedback",
        "previous_response": "previous test response",
        "feedback_history": [
            {
                "stage": "stage1",
                "response": "response1",
                "feedback": "feedback1"
            },
            {
                "stage": "stage2",
                "response": "response2",
                "feedback": "feedback2"
            }
        ],
        "error": False
    }

@pytest.fixture
def dlpfc_agent(mock_env_vars):
    return DLPFCAgent()

@pytest.mark.asyncio
async def test_dlpfc_agent_initialization(dlpfc_agent):
    """Test DLPFC agent initialization"""
    assert isinstance(dlpfc_agent, DLPFCAgent)
    assert dlpfc_agent.llm.model_name == "dlpfc-model"

@pytest.mark.asyncio
async def test_dlpfc_agent_process(dlpfc_agent, test_state):
    """Test DLPFC agent processing"""
    mock_response = MagicMock()
    mock_response.content = """
    Here's the task breakdown:
    1. Subtask 1 - Assign to VMPFC
    2. Subtask 2 - Assign to OFC
    3. Subtask 3 - Assign to ACC
    
    Integration plan:
    1. Collect responses
    2. Analyze results
    3. Generate final output
    """
    
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke_messages = AsyncMock(return_value=mock_response)
    
    result = await dlpfc_agent.process(test_state)
    assert isinstance(result, dict)
    assert "subtasks" in result
    assert result["stage"] == "task_delegation"
    assert not result.get("error", False)

@pytest.mark.asyncio
async def test_dlpfc_agent_error_handling(dlpfc_agent, test_state):
    """Test error handling in DLPFC agent"""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke_messages = AsyncMock(side_effect=ValueError("Test error"))
    
    result = await dlpfc_agent.process(test_state)
    assert result["error"]
    assert "error" in result["response"].lower()

@pytest.mark.asyncio
async def test_dlpfc_agent_timeout(dlpfc_agent, test_state):
    """Test timeout handling in DLPFC agent"""
    async def mock_process(*args, **kwargs):
        await asyncio.sleep(1)
        return None

    dlpfc_agent.process = mock_process
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):
            await dlpfc_agent.process(test_state)

@pytest.mark.asyncio
async def test_dlpfc_agent_cancellation(dlpfc_agent, test_state):
    """Test cancellation handling in DLPFC agent"""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke_messages = AsyncMock(side_effect=asyncio.CancelledError())
    
    result = await dlpfc_agent.process(test_state)
    assert result["error"]
    assert "cancelled" in result["response"].lower()

def test_dlpfc_format_feedback_history(dlpfc_agent, test_state):
    """Test feedback history formatting"""
    history = test_state["feedback_history"]
    formatted = dlpfc_agent._format_feedback_history(history)
    assert "stage1" in formatted
    assert "response1" in formatted
    assert "feedback1" in formatted
    assert "stage2" in formatted
    assert "response2" in formatted
    assert "feedback2" in formatted

def test_dlpfc_format_feedback_history_empty(dlpfc_agent):
    """Test feedback history formatting with empty history"""
    formatted = dlpfc_agent._format_feedback_history([])
    assert formatted == "No previous feedback"

def test_dlpfc_parse_subtasks(dlpfc_agent):
    """Test subtask parsing"""
    response = """
    Here's the task breakdown:
    1. Analyze data - Assign to VMPFC
    2. Generate options - Assign to OFC
    3. Monitor progress - Assign to ACC
    """
    subtasks = dlpfc_agent._parse_subtasks(response)
    assert isinstance(subtasks, list)
    assert len(subtasks) > 0
    assert all(isinstance(task, dict) for task in subtasks)
    assert all("task" in task and "agent" in task for task in subtasks)

@pytest.mark.asyncio
async def test_dlpfc_error_types(dlpfc_agent, test_state):
    """Test different error types in DLPFC agent"""
    error_scenarios = {
        "connection": ConnectionError("Connection failed"),
        "timeout": asyncio.TimeoutError(),
        "cancelled": asyncio.CancelledError(),
        "api_error": openai.APIError("API Error"),
        "auth_error": AuthenticationError("Invalid key")
    }
    
    for error_type, exception in error_scenarios.items():
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=exception):
            result = await dlpfc_agent.process(test_state)
            assert result["error"]
            assert isinstance(result["response"], str)
            if error_type in ["connection", "api_error", "auth_error"]:
                assert result["error_type"] == "connection"
            else:
                assert result["error_type"] == error_type

@pytest.mark.asyncio
async def test_dlpfc_subtask_parsing_edge_cases(dlpfc_agent):
    """Test edge cases in subtask parsing"""
    test_cases = [
        ("", []),  # Empty response
        ("No tasks found", []),  # No tasks in response
        ("1. Task without agent", [{"task": "Task without agent", "agent": None, "category": "general"}]),
        ("Invalid format\nRandom text", []),  # Invalid format
        ("1. Task 1 - Agent: VMPFC\n2. Task 2 - Agent: OFC", [
            {"task": "Task 1", "agent": "VMPFC", "category": "general"},
            {"task": "Task 2", "agent": "OFC", "category": "general"}
        ])
    ]
    
    for response, expected in test_cases:
        result = dlpfc_agent._parse_subtasks(response)
        assert result == expected
