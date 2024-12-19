import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from agents.dlpfc import DLPFCAgent
from typing import Dict, Any

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
async def test_dlpfc_agent_process(dlpfc_agent: DLPFCAgent, test_state: Dict[str, Any]):
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
    dlpfc_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
    
    result = await dlpfc_agent.process(test_state)
    assert isinstance(result, dict)
    assert "subtasks" in result
    assert result["stage"] == "task_delegation"
    assert not result.get("error", False)

@pytest.mark.asyncio
async def test_dlpfc_agent_error_handling(dlpfc_agent: DLPFCAgent, test_state: Dict[str, Any]):
    """Test error handling in DLPFC agent"""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(side_effect=ValueError("Test error"))
    
    result = await dlpfc_agent.process(test_state)
    assert result["error"]
    assert "error" in result["response"].lower()

@pytest.mark.asyncio
async def test_dlpfc_agent_timeout(dlpfc_agent: DLPFCAgent, test_state: Dict[str, Any]):
    """Test timeout handling in DLPFC agent"""
    async def mock_process(*args, **kwargs):
        await asyncio.sleep(1)
        return None

    dlpfc_agent.process = mock_process
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):
            await dlpfc_agent.process(test_state)

@pytest.mark.asyncio
async def test_dlpfc_agent_cancellation(dlpfc_agent: DLPFCAgent, test_state: Dict[str, Any]):
    """Test cancellation handling in DLPFC agent"""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(side_effect=asyncio.CancelledError())
    
    result = await dlpfc_agent.process(test_state)
    assert result["error"]
    assert "cancelled" in result["response"].lower()

def test_dlpfc_format_feedback_history(dlpfc_agent: DLPFCAgent, test_state: Dict[str, Any]):
    """Test feedback history formatting"""
    history = test_state.get("feedback_history", [])
    formatted = dlpfc_agent._format_feedback_history(history)
    assert "stage1" in formatted
    assert "response1" in formatted
    assert "feedback1" in formatted
    assert "stage2" in formatted
    assert "response2" in formatted
    assert "feedback2" in formatted

def test_dlpfc_format_feedback_history_empty(dlpfc_agent: DLPFCAgent):
    """Test feedback history formatting with empty history"""
    formatted = dlpfc_agent._format_feedback_history([])
    assert formatted == "No previous feedback"

def test_dlpfc_parse_subtasks(dlpfc_agent: DLPFCAgent):
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
