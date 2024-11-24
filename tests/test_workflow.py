import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from workflow import (
    create_workflow, process_hitl_feedback, AgentState, END,
    timeout_context, process_task_delegation, process_emotional_regulation,
    process_reward_processing, process_conflict_detection, process_value_assessment
)
import asyncio
from langchain.prompts import ChatPromptTemplate
from agents.base import BaseAgent

# Mock ChatOpenAI at import time
mock_chat_openai = AsyncMock()
mock_chat_openai.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))

@pytest.fixture
def mock_env_vars():
    with patch.dict('os.environ', {
        'DLPFC_MODEL': 'dlpfc-model',
        'VMPFC_MODEL': 'vmpfc-model',
        'OFC_MODEL': 'ofc-model',
        'ACC_MODEL': 'acc-model',
        'MPFC_MODEL': 'mpfc-model',
    }), patch('agents.base.ChatOpenAI', return_value=mock_chat_openai):
        yield

@pytest.fixture
def mock_llm():
    async def mock_ainvoke(*args, **kwargs):
        return MagicMock(content="test response")
    
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=mock_ainvoke):
        yield

@pytest.fixture
def mock_state():
    return {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }

@pytest.mark.asyncio
async def test_workflow_creation(mock_env_vars, mock_llm):
    """Test workflow creation and structure"""
    workflow = create_workflow()
    assert workflow is not None

@pytest.mark.asyncio
async def test_hitl_feedback_processing(mock_env_vars):
    """Test HITL feedback processing"""
    initial_state = {
        "task": "test task",
        "stage": "value_assessment",
        "response": "test response",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    feedback = "Test feedback"
    updated_state = process_hitl_feedback(initial_state.copy(), feedback)
    
    assert updated_state["feedback"] == feedback
    assert len(updated_state["feedback_history"]) == 1
    assert updated_state["previous_response"] == "test response"
    assert id(updated_state) != id(initial_state)  # Ensure we got a new state object

@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_env_vars, mock_llm):
    """Test workflow state transitions"""
    workflow = create_workflow()
    
    # Test initial state
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Mock agent process functions to return proper state
    async def mock_process(*args, **kwargs):
        state = args[1] if len(args) > 1 else kwargs.get('state')
        stage_map = {
            "task_delegation": {"stage": "emotional_regulation"},
            "emotional_regulation": {"stage": "reward_processing"},
            "reward_processing": {"stage": "conflict_detection"},
            "conflict_detection": {"stage": "value_assessment"},
            "value_assessment": {"stage": END}
        }
        next_stage = stage_map.get(state["stage"])
        return {
            "response": "test response",
            **next_stage,
            "error": False
        }
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_process), \
         patch("agents.specialized.VMPFCAgent.process", new=mock_process), \
         patch("agents.specialized.OFCAgent.process", new=mock_process), \
         patch("agents.specialized.ACCAgent.process", new=mock_process), \
         patch("agents.specialized.MPFCAgent.process", new=mock_process):
        
        final_state = await workflow.ainvoke(initial_state)
        assert not final_state.get("error"), f"Workflow failed with error: {final_state.get('response')}"
        assert final_state["stage"] == END

@pytest.mark.asyncio
async def test_error_handling(mock_env_vars, mock_llm):
    """Test error handling in workflow"""
    workflow = create_workflow()
    
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Mock process to simulate an error
    async def mock_error_process(*args, **kwargs):
        return {
            "response": "Error occurred",
            "error": True
        }
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_error_process):
        final_state = await workflow.ainvoke(initial_state)
        assert final_state["error"]
        assert "Error occurred" in final_state["response"]

@pytest.mark.asyncio
async def test_timeout_handling(mock_env_vars, mock_llm):
    """Test timeout handling in workflow"""
    workflow = create_workflow()
    
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Mock process to simulate a timeout
    async def mock_timeout_process(*args, **kwargs):
        await asyncio.sleep(60)  # Sleep longer than the timeout
        return {"response": "Should timeout"}
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_timeout_process):
        final_state = await workflow.ainvoke(initial_state)
        assert final_state["error"]
        assert "timed out" in final_state["response"].lower()

@pytest.mark.asyncio
async def test_cancellation_handling(mock_env_vars, mock_llm):
    """Test cancellation handling in workflow"""
    workflow = create_workflow()
    
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Mock process to simulate a cancellation
    async def mock_cancel_process(*args, **kwargs):
        raise asyncio.CancelledError()
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_cancel_process):
        final_state = await workflow.ainvoke(initial_state)
        assert final_state["error"]
        assert "cancelled" in final_state["response"].lower()

@pytest.mark.asyncio
async def test_timeout_context():
    """Test timeout context manager"""
    # Test normal execution
    async with timeout_context(1.0):
        await asyncio.sleep(0.1)  # Should complete normally
    
    # Test timeout
    with pytest.raises(TimeoutError):
        async with timeout_context(0.1):
            await asyncio.wait_for(asyncio.sleep(1.0), timeout=0.1)  # Should timeout
    
    # Test cancellation
    with pytest.raises(KeyboardInterrupt):
        async with timeout_context(1.0):
            raise asyncio.CancelledError()

@pytest.mark.asyncio
async def test_process_task_delegation(mock_env_vars, mock_state):
    """Test task delegation processing"""
    # Test successful processing
    with patch("agents.dlpfc.DLPFCAgent.process", new=AsyncMock(return_value={"response": "success", "stage": "next"})):
        result = await process_task_delegation(mock_state)
        assert result["stage"] == "emotional_regulation"
        assert not result["error"]
    
    # Test timeout
    async def timeout_process(*args, **kwargs):
        await asyncio.sleep(60)
        return {}
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=timeout_process):
        result = await process_task_delegation(mock_state)
        assert result["error"]
        assert "timed out" in result["response"].lower()
    
    # Test error
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_task_delegation(mock_state)
        assert result["error"]
        assert "test error" in result["response"]

@pytest.mark.asyncio
async def test_process_emotional_regulation(mock_env_vars, mock_state):
    """Test emotional regulation processing"""
    # Test successful processing
    with patch("agents.specialized.VMPFCAgent.process", new=AsyncMock(return_value={"response": "success", "stage": "next"})):
        result = await process_emotional_regulation(mock_state)
        assert result["stage"] == "reward_processing"
        assert not result["error"]
    
    # Test error
    with patch("agents.specialized.VMPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_emotional_regulation(mock_state)
        assert result["error"]
        assert "test error" in result["response"]

@pytest.mark.asyncio
async def test_process_reward_processing(mock_env_vars, mock_state):
    """Test reward processing"""
    # Test successful processing
    with patch("agents.specialized.OFCAgent.process", new=AsyncMock(return_value={"response": "success", "stage": "next"})):
        result = await process_reward_processing(mock_state)
        assert result["stage"] == "conflict_detection"
        assert not result["error"]
    
    # Test error
    with patch("agents.specialized.OFCAgent.process", side_effect=ValueError("test error")):
        result = await process_reward_processing(mock_state)
        assert result["error"]
        assert "test error" in result["response"]

@pytest.mark.asyncio
async def test_process_conflict_detection(mock_env_vars, mock_state):
    """Test conflict detection processing"""
    # Test successful processing
    with patch("agents.specialized.ACCAgent.process", new=AsyncMock(return_value={"response": "success", "stage": "next"})):
        result = await process_conflict_detection(mock_state)
        assert result["stage"] == "value_assessment"
        assert not result["error"]
    
    # Test error
    with patch("agents.specialized.ACCAgent.process", side_effect=ValueError("test error")):
        result = await process_conflict_detection(mock_state)
        assert result["error"]
        assert "test error" in result["response"]

@pytest.mark.asyncio
async def test_process_value_assessment(mock_env_vars, mock_state):
    """Test value assessment processing"""
    result = await process_value_assessment(mock_state)
    assert not result.get("error")
    assert result["stage"] == END
    assert "response" in result

@pytest.mark.asyncio
async def test_workflow_state_transitions_with_errors(mock_env_vars):
    """Test workflow state transitions with errors in different stages"""
    workflow = create_workflow()
    
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Mock error in task delegation
    async def mock_error(*args, **kwargs):
        raise ValueError("Test error")
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_error):
        final_state = await workflow.ainvoke(initial_state)
        assert final_state["error"]
        assert "Test error" in final_state["response"]
        assert final_state["stage"] == END

def test_hitl_feedback_history(mock_env_vars):
    """Test HITL feedback with multiple entries"""
    state = {
        "task": "test task",
        "stage": "value_assessment",
        "response": "test response 1",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Add first feedback
    state = process_hitl_feedback(state, "feedback 1")
    assert len(state["feedback_history"]) == 1
    assert state["feedback_history"][0]["feedback"] == "feedback 1"
    assert state["feedback_history"][0]["response"] == "test response 1"
    
    # Update response and add second feedback
    state["response"] = "test response 2"
    state = process_hitl_feedback(state, "feedback 2")
    assert len(state["feedback_history"]) == 2
    assert state["feedback_history"][1]["feedback"] == "feedback 2"
    assert state["feedback_history"][1]["response"] == "test response 2"
    
    # Verify previous response is updated
    assert state["previous_response"] == "test response 2"
