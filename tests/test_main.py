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
        with patch("main.load_feedback_history", return_value=[]):
            with patch("builtins.input", side_effect=["test task", "n", "exit"]):
                await main()
                # Verify ainvoke was called with the correct task and empty feedback history
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
async def test_feedback_processing(mock_env_vars):
    """Test feedback collection and processing."""
    # Create a mock workflow that returns a success response
    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response": "Test response",
        "stage": "__end__",
        "error": False
    })
    
    # Mock load_feedback_history to return an empty list
    with patch("main.create_workflow", return_value=mock_workflow):
        with patch("main.load_feedback_history", return_value=[]):
            # Mock save_feedback_history
            with patch("main.save_feedback_history") as mock_save:
                with patch('builtins.input', side_effect=["test task", "y", "Test feedback", "exit"]):
                    await main()
        
                    # Verify ainvoke was called once for the task
                    assert mock_workflow.ainvoke.call_count >= 1
        
                    # Verify save_feedback_history was called once
                    assert mock_save.call_count == 1
                    
                    # Check the feedback data structure passed to save_feedback_history
                    feedback_history = mock_save.call_args[0][0]
                    assert len(feedback_history) == 1
                    assert feedback_history[0]['feedback'] == 'Test feedback'

@pytest.mark.asyncio
async def test_error_handling(mock_env_vars):
    """Test error handling during workflow execution."""
    # Create a mock workflow that raises an exception
    mock_workflow = MagicMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=Exception("Test error"))
    
    # Patch sys.exit to prevent test from exiting
    with patch("sys.exit"):
        with patch("main.create_workflow", return_value=mock_workflow):
            with patch("main.load_feedback_history", return_value=[]):
                with patch('builtins.input', side_effect=["test task"]):
                    with patch('builtins.print') as mock_print:
                        # We need to catch the exception that main will re-raise
                        try:
                            await main()
                        except Exception as e:
                            # Verify it was our test error
                            assert "Test error" in str(e)
                        
                        # Check if error message was printed
                        error_calls = [
                            call_args for call_args in mock_print.call_args_list 
                            if isinstance(call_args[0][0], str) and "error" in call_args[0][0].lower()
                        ]
                        assert len(error_calls) > 0
                    
                    # Verify ainvoke was called once
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