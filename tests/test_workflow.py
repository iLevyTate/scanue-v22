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
from utils.logging import InteractionLogger
import json

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
        'OPENAI_API_KEY': 'test-key'
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
        "error": False,
        "scanaq_results": "Test SCANAQ results"
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
        "stage": "task_delegation",
        "response": "initial response",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    state = await process_hitl_feedback(initial_state, "test feedback")
    assert state["feedback"] == "test feedback"
    assert state["previous_response"] == "initial response"
    assert len(state["feedback_history"]) == 1
    assert state["feedback_history"][0]["feedback"] == "test feedback"

@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_env_vars):
    """Test workflow state transitions"""
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
    
    workflow = create_workflow()
    final_state = await workflow.ainvoke(initial_state)
    
    assert final_state["stage"] == END
    assert "response" in final_state

@pytest.mark.asyncio
async def test_error_handling(mock_env_vars):
    """Test error handling in workflow"""
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False  # Set initial error state to False
    }
    
    with patch("workflow.process_task_delegation", side_effect=Exception("Simulated error")):
        workflow = create_workflow()
        final_state = await workflow.ainvoke(initial_state)
        
        assert final_state["error"] == True
        assert "simulated error" in final_state["response"].lower()

@pytest.mark.asyncio
async def test_timeout_handling(mock_env_vars):
    """Test timeout handling in workflow"""
    async def slow_task():
        await asyncio.sleep(1.0)
        
    with pytest.raises(asyncio.TimeoutError):
        async with timeout_context(0.1):
            await slow_task()

@pytest.mark.asyncio
async def test_cancellation_handling(mock_env_vars):
    """Test cancellation handling in workflow"""
    async def cancellable_task():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
            
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(0.1)
    task.cancel()
    
    with pytest.raises(asyncio.CancelledError):
        await task

@pytest.mark.asyncio
async def test_timeout_context(mock_env_vars):
    """Test timeout context"""
    async def slow_task():
        await asyncio.sleep(1.0)
        
    with pytest.raises(asyncio.TimeoutError):
        async with timeout_context(0.1):
            await slow_task()

@pytest.mark.asyncio
async def test_process_task_delegation(mock_env_vars, mock_state):
    """Test task delegation processing"""
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=asyncio.TimeoutError()):
        result = await process_task_delegation(mock_state)
        assert "task delegation timed out" in result["response"].lower()
    
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=Exception("Simulated error")):
        result = await process_task_delegation(mock_state)
        assert "error in task delegation" in result["response"].lower()

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
async def test_workflow_state_transitions_with_errors(mock_env_vars, mock_state):
    """Test workflow state transitions when errors occur"""
    # Simulate an error in task delegation
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=Exception("Simulated error")):
        final_state = await process_task_delegation(mock_state)
        assert final_state["error"] == True, "Error flag not set to True"
        assert "simulated error" in final_state["response"].lower(), "Error message does not contain expected text"
        assert final_state["stage"] == END, "Workflow did not transition to END stage"

@pytest.mark.asyncio
async def test_hitl_feedback_history(mock_env_vars):
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
    state = await process_hitl_feedback(state, "feedback 1")
    assert len(state["feedback_history"]) == 1
    assert state["feedback_history"][0]["feedback"] == "feedback 1"
    assert state["feedback_history"][0]["response"] == "test response 1"
    
    # Update response and add second feedback
    state["response"] = "test response 2"
    state = await process_hitl_feedback(state, "feedback 2")
    assert len(state["feedback_history"]) == 2
    assert state["feedback_history"][1]["feedback"] == "feedback 2"
    assert state["feedback_history"][1]["response"] == "test response 2"
    
    # Verify previous response is updated
    assert state["previous_response"] == "test response 2"

@pytest.mark.asyncio
async def test_hitl_feedback_integration(mock_env_vars):
    """Test complete HITL feedback integration"""
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": "initial response",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Test feedback collection
    state = await process_hitl_feedback(initial_state, "test feedback")
    assert len(state["feedback_history"]) == 1
    assert state["feedback_history"][0]["stage"] == "task_delegation"
    assert state["feedback_history"][0]["feedback"] == "test feedback"
    assert state["feedback_history"][0]["response"] == "initial response"
    assert "timestamp" in state["feedback_history"][0]
    
    # Test feedback persistence through workflow
    workflow = create_workflow()
    state = await workflow.ainvoke(state)
    assert len(state["feedback_history"]) == 1  # Feedback history preserved
    assert state["previous_response"] == "initial response"

@pytest.mark.asyncio
async def test_workflow_logging_integration(mock_env_vars, tmp_path):
    """Test logging integration with workflow"""
    logger = InteractionLogger(log_dir=str(tmp_path))
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
    
    # Log initial state
    logger.log_state(initial_state, "initial")
    
    # Process through workflow
    state = await workflow.ainvoke(initial_state)
    logger.log_state(state, state.get("stage", "unknown"))
    
    # Verify logs
    with open(logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) >= 2  # At least initial and final states
        assert log_data[0]["stage"] == "initial"
        assert "timestamp" in log_data[0]
        assert isinstance(log_data[0]["state"], dict)

@pytest.mark.asyncio
async def test_feedback_logging_integration(mock_env_vars, tmp_path):
    """Test feedback logging integration"""
    logger = InteractionLogger(log_dir=str(tmp_path))
    
    state = {
        "task": "test task",
        "stage": "value_assessment",
        "response": "test response",
        "feedback_history": []
    }
    
    # Log state and feedback
    logger.log_state(state, "value_assessment")
    state = await process_hitl_feedback(state, "test feedback")
    logger.log_feedback("test feedback", state)
    
    # Verify logs
    with open(logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 2
        assert log_data[1]["type"] == "feedback"
        assert log_data[1]["feedback"] == "test feedback"
