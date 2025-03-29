import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from main import (
    print_thinking_animation,
    print_agent_transition,
    format_stage_name,
    main
)

@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
        "VMPFC_MODEL": "test-model",
        "OFC_MODEL": "test-model",
        "ACC_MODEL": "test-model",
        "MPFC_MODEL": "test-model",
        "DLPFC_MODEL": "test-model"
    }):
        yield

@pytest.fixture
def mock_workflow():
    """Mock workflow for testing."""
    workflow = AsyncMock()
    workflow.ainvoke = AsyncMock(return_value={
        "response": "Test response",
        "stage": "complete",
        "error": False
    })
    return workflow

def test_format_stage_name():
    """Test stage name formatting with emojis."""
    assert "📋 Task Delegation" == format_stage_name("task_delegation")
    assert "❤️ Emotional Regulation" == format_stage_name("emotional_regulation")
    assert "🎯 Reward Processing" == format_stage_name("reward_processing")
    assert "⚡ Conflict Detection" == format_stage_name("conflict_detection")
    assert "💡 Value Assessment" == format_stage_name("value_assessment")
    assert "✅ Complete" == format_stage_name("complete")
    assert "🔹 Unknown Stage" == format_stage_name("unknown_stage")

def test_print_agent_transition(capsys):
    """Test agent transition printing."""
    print_agent_transition("stage1", "stage2")
    captured = capsys.readouterr()
    expected_output = (
        "\n--------------------\n"
        "🔄 STAGE1 → STAGE2\n"
        "--------------------\n\n"
    )
    assert captured.out == expected_output

def test_print_thinking_animation(capsys):
    """Test thinking animation with reduced duration."""
    print_thinking_animation("Thinking", duration=1)
    captured = capsys.readouterr()
    assert "Thinking" in captured.out
    assert "..." in captured.out

@pytest.mark.asyncio
async def test_app_initialization(mock_env_vars, mock_workflow):
    """Test application initialization and configuration."""
    with patch("main.create_workflow", return_value=mock_workflow):
        with patch("builtins.input", side_effect=["test task", "n", "exit"]):
            await main()
            # Verify ainvoke was called with the correct task
            mock_workflow.ainvoke.assert_called_with({
                "task": "test task",
                "stage": "task_delegation",
                "response": "",
                "subtasks": [],
                "feedback": "",
                "previous_response": "",
                "feedback_history": [],
                "error": False
            })

@pytest.mark.asyncio
async def test_empty_task_handling(mock_env_vars, mock_workflow, capsys):
    """Test handling of empty task input."""
    with patch("main.create_workflow", return_value=mock_workflow):
        with patch("builtins.input", side_effect=["", "exit"]):
            await main()
            captured = capsys.readouterr()
            assert "Task cannot be empty" in captured.out
            mock_workflow.ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_feedback_processing(mock_workflow):
    """Test feedback collection and processing."""
    # First call returns a result that requires feedback
    mock_workflow.ainvoke.side_effect = [
        {"response": "Task completed", "feedback_required": True},
        {"response": "Feedback processed"}
    ]
    
    with patch('builtins.input', side_effect=['y', 'Test feedback']):
        await main(["test task"])
    
    # Verify ainvoke was called twice - once for task and once for feedback
    assert mock_workflow.ainvoke.call_count == 2
    
    # Verify the second call included the feedback
    feedback_call = mock_workflow.ainvoke.call_args_list[1][0][0]
    assert feedback_call["feedback"] == "Test feedback"
    assert len(feedback_call["feedback_history"]) == 1
    assert feedback_call["feedback_history"][0]["feedback"] == "Test feedback"

@pytest.mark.asyncio
async def test_error_handling(mock_workflow):
    """Test error handling during workflow execution."""
    mock_workflow.ainvoke.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as excinfo:
        await main(["test task"])
    
    assert "Test error" in str(excinfo.value)
    assert mock_workflow.ainvoke.call_count == 1

@pytest.mark.asyncio
async def test_keyboard_interrupt_handling(mock_env_vars, mock_workflow, capsys):
    """Test handling of keyboard interrupt."""
    with patch("main.create_workflow", return_value=mock_workflow):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await main()
            captured = capsys.readouterr()
            assert "interrupted" in captured.out.lower()

@pytest.mark.asyncio
async def test_missing_api_key(monkeypatch):
    """Test that the application exits when API key is missing."""
    # Remove API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    with pytest.raises(SystemExit) as excinfo:
        await main(["test task"])
    
    assert excinfo.value.code == 1 