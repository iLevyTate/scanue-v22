import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from workflow import (
    create_workflow, process_hitl_feedback, AgentState, END,
    timeout_context, process_task_delegation, process_emotional_regulation,
    process_reward_processing, process_conflict_detection, process_value_assessment
)
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent

# Mock ChatOpenAI at import time
mock_chat_openai = AsyncMock()
mock_chat_openai.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))

@pytest.fixture
def mock_env_vars():
    # Mock ConfigLoader to return a consistent test configuration
    test_config = {
        "agents": {
            "DLPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "VMPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "OFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "ACC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "MPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        }
    }
    
    with patch.dict('os.environ', {
        'DLPFC_MODEL': 'dlpfc-model',
        'VMPFC_MODEL': 'vmpfc-model',
        'OFC_MODEL': 'ofc-model',
        'ACC_MODEL': 'acc-model',
        'MPFC_MODEL': 'mpfc-model',
        'OPENAI_API_KEY': 'test-key'
    }), patch('utils.config.ConfigLoader.load_config', return_value=test_config), \
       patch('agents.factory.ChatOpenAI', return_value=mock_chat_openai):
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
        "delegated_agents": ["emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"],
        "agent_responses": {},
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
        agent_responses = state.get("agent_responses", {})
        
        # Define structured response
        structured_response = {"role": "assistant", "content": "test response"}
        
        # Simulate agent execution adding to agent_responses
        current_stage = state["stage"]
        
        if current_stage == "task_delegation":
            return {
                "response": structured_response,
                "delegated_agents": ["emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"],
                "agent_responses": {},
            }
        else:
            current_stage_agent = {
                "emotional_regulation": "VMPFC",
                "reward_processing": "OFC",
                "conflict_detection": "ACC",
                "value_assessment": "MPFC"
            }.get(current_stage)
            
            # Need to create a NEW dict to avoid mutating shared state across recursions if runner reuses objects
            # Note: For LangGraph state updates to work correctly in loop, we must return the updated key.
            new_agent_responses = agent_responses.copy()
            if current_stage_agent:
                new_agent_responses[current_stage_agent] = structured_response
                
            return {
                "response": structured_response,
                "agent_responses": new_agent_responses
            }
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_process), \
         patch("agents.specialized.VMPFCAgent.process", new=mock_process), \
         patch("agents.specialized.OFCAgent.process", new=mock_process), \
         patch("agents.specialized.ACCAgent.process", new=mock_process), \
         patch("agents.specialized.MPFCAgent.process", new=mock_process):
        
        final_state = await workflow.ainvoke(initial_state)
        assert not final_state.get("error"), f"Workflow failed with error: {final_state.get('response')}"
        # With LangGraph, the state['stage'] might not update to END explicitly in the state dict
        # Instead, verify we reached the end by checking agent responses
        assert "MPFC" in final_state.get("agent_responses", {})

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
        structured_error = {
            "role": "assistant",
            "content": "Error occurred"
        }
        return {
            "response": structured_error,
            "error": True
        }
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_error_process):
        final_state = await workflow.ainvoke(initial_state)
        # Check for error in agent_errors
        assert final_state.get("agent_errors", {}).get("DLPFC") == "Error occurred"
        # Verify workflow continued despite error (resilience)
        assert not final_state.get("error")

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
        # Instead of sleeping, raise TimeoutError directly to be faster and more reliable
        raise TimeoutError("Operation timed out")
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_timeout_process):
        final_state = await workflow.ainvoke(initial_state)
        # Check for error in agent_errors
        assert "timed out" in final_state.get("agent_errors", {}).get("DLPFC", "").lower()
        # Verify workflow continued
        assert not final_state.get("error")

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
        raise asyncio.CancelledError("Operation was cancelled")
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_cancel_process):
        final_state = await workflow.ainvoke(initial_state)
        # Check for error in agent_errors
        assert "cancelled" in final_state.get("agent_errors", {}).get("DLPFC", "").lower()
        # Verify workflow continued
        assert not final_state.get("error")

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
    mock_response = {"response": {"role": "assistant", "content": "success"}, "stage": "task_delegation"}
    with patch("agents.dlpfc.DLPFCAgent.process", new=AsyncMock(return_value=mock_response)):
        result = await process_task_delegation(mock_state)
        # Check against expected next stage based on parsing
        # Note: If no agents are parsed, it defaults to value_assessment
        assert result["stage"] in ["value_assessment", "emotional_regulation"]
        assert not result.get("error")
    
    # Test timeout
    async def mock_timeout_process(*args, **kwargs):
        raise TimeoutError("Operation timed out")
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_timeout_process):
        result = await process_task_delegation(mock_state)
        # Check for error in agent_errors if error flag is not set
        assert result.get("error") or result.get("agent_errors", {}).get("DLPFC")
        if "response" in result:
             error_message = result["response"]["content"] if isinstance(result["response"], dict) and "content" in result["response"] else str(result["response"])
             assert "timed out" in error_message.lower()
    
    # Test error
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_task_delegation(mock_state)
        assert result.get("error") or result.get("agent_errors", {}).get("DLPFC")
        if "response" in result:
             error_message = result["response"]["content"] if isinstance(result["response"], dict) and "content" in result["response"] else str(result["response"])
             assert "test error" in error_message

@pytest.mark.asyncio
async def test_process_emotional_regulation(mock_env_vars, mock_state):
    """Test emotional regulation processing"""
    # Test successful processing
    mock_response = {"response": {"role": "assistant", "content": "success"}}
    with patch("agents.specialized.VMPFCAgent.process", new=AsyncMock(return_value=mock_response)):
        # Ensure delegated_agents is set correctly in mock_state
        mock_state["delegated_agents"] = ["emotional_regulation", "reward_processing"]
        result = await process_emotional_regulation(mock_state)
        assert not result.get("error")
        # Assert agent response is stored
        assert "VMPFC" in result["agent_responses"]
    
    # Test error
    with patch("agents.specialized.VMPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_emotional_regulation(mock_state)
        # Agent errors are now stored in "agent_errors" dict
        assert result.get("agent_errors", {}).get("VMPFC")
        error_message = result["response"]["content"] if isinstance(result["response"], dict) and "content" in result["response"] else result["response"]
        assert "test error" in error_message

@pytest.mark.asyncio
async def test_process_reward_processing(mock_env_vars, mock_state):
    """Test reward processing"""
    # Test successful processing
    mock_response = {"response": {"role": "assistant", "content": "success"}, "stage": "next"}
    with patch("agents.specialized.OFCAgent.process", new=AsyncMock(return_value=mock_response)):
        result = await process_reward_processing(mock_state)
        assert result["stage"] == "conflict_detection"
        assert not result["error"]
    
    # Test error
    with patch("agents.specialized.OFCAgent.process", side_effect=ValueError("test error")):
        result = await process_reward_processing(mock_state)
        # Check for error in return value which might be structured differently
        assert result.get("error") or result.get("agent_errors", {}).get("OFC")
        # If error is in response content
        if "response" in result:
             error_message = result["response"]["content"] if isinstance(result["response"], dict) and "content" in result["response"] else str(result["response"])
             assert "test error" in error_message

@pytest.mark.asyncio
async def test_process_conflict_detection(mock_env_vars, mock_state):
    """Test conflict detection processing"""
    # Test successful processing
    mock_response = {"response": {"role": "assistant", "content": "success"}}
    with patch("agents.specialized.ACCAgent.process", new=AsyncMock(return_value=mock_response)):
        result = await process_conflict_detection(mock_state)
        # process_conflict_detection does NOT set "stage" in the result.
        # It relies on LangGraph conditional edges.
        assert not result.get("error")
        assert "ACC" in result["agent_responses"]
    
    # Test error
    with patch("agents.specialized.ACCAgent.process", side_effect=ValueError("test error")):
        result = await process_conflict_detection(mock_state)
        # Check for error in return value which might be structured differently
        assert result.get("error") or result.get("agent_errors", {}).get("ACC")
        if "response" in result:
             error_message = result["response"]["content"] if isinstance(result["response"], dict) and "content" in result["response"] else str(result["response"])
             assert "test error" in error_message

@pytest.mark.asyncio
async def test_process_value_assessment(mock_env_vars, mock_state):
    """Test value assessment processing"""
    # Test successful processing
    mock_response = {"response": {"role": "assistant", "content": "test response"}}
    with patch("agents.specialized.MPFCAgent.process", new=AsyncMock(return_value=mock_response)):
        result = await process_value_assessment(mock_state)
        # process_value_assessment does NOT set "stage" in the result.
        assert not result.get("error")
        assert "MPFC" in result["agent_responses"]

@pytest.mark.asyncio
async def test_workflow_state_transitions_with_errors(mock_env_vars):
    """Test workflow state transitions with errors"""
    workflow = create_workflow()
    
    # Initial state
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
    async def mock_error(*args, **kwargs):
        structured_error = {
            "role": "assistant",
            "content": "Test error"
        }
        return {
            "response": structured_error,
            "error": True
        }
    
    with patch("agents.dlpfc.DLPFCAgent.process", new=mock_error):
        final_state = await workflow.ainvoke(initial_state)
        # Check if error is propagated
        has_error = final_state.get("error") or final_state.get("agent_errors", {}).get("DLPFC")
        assert has_error
        
        # Check error message content
        response = final_state.get("response")
        error_message = ""
        if isinstance(response, dict) and "content" in response:
            error_message = response["content"]
        elif isinstance(response, str):
            error_message = response
        elif "agent_errors" in final_state and "DLPFC" in final_state["agent_errors"]:
            error_message = final_state["agent_errors"]["DLPFC"]
            
        assert "Test error" in error_message or "Test error" in str(final_state)

def test_hitl_feedback_history(mock_env_vars):
    """Test HITL feedback with multiple entries"""
    state = {
        "task": "test task",
        "stage": "value_assessment",
        "response": {"role": "assistant", "content": "test response 1"},
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
    state["response"] = {"role": "assistant", "content": "test response 2"}
    state = process_hitl_feedback(state, "feedback 2")
    assert len(state["feedback_history"]) == 2
    assert state["feedback_history"][1]["feedback"] == "feedback 2"
    assert state["feedback_history"][1]["response"] == "test response 2"
    
    # Verify previous response is updated
    assert state["previous_response"] == "test response 2"
