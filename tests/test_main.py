import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langgraph.errors import GraphRecursionError
from main import main


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
    }):
        yield


@pytest.fixture
def mock_workflow():
    """Mock workflow for testing."""
    workflow = AsyncMock()
    workflow.ainvoke = AsyncMock(return_value={
        "response": "Test response",
        "stage": "complete",
        "error": False,
    })
    return workflow


MOCK_SESSION = {
    "task": "test task",
    "timestamp": "2023-01-01T00:00:00.000000",
    "session_id": "test-session-id",
    "stages": [],
    "final_response": None,
    "user_feedback": None,
    "error": None,
    "completed": False,
}


@pytest.mark.asyncio
async def test_app_initialization(mock_env_vars, mock_workflow):
    """The initial workflow state carries the new keys and ainvoke gets the
    recursion_limit config."""
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("builtins.input", side_effect=["test task", "n", "exit"]), \
         patch("main.create_session_log", return_value=MOCK_SESSION), \
         patch("main.save_session_log", return_value="test_log_file.json"):
        await main()

        expected_state = {
            "task": "test task",
            "stage": "task_delegation",
            "response": "",
            "subtasks": [],
            "feedback": "",
            "previous_response": "",
            "feedback_history": [],
            "session_log": MOCK_SESSION,
            "completed_stages": [],
            "error": False,
        }
        mock_workflow.ainvoke.assert_called_with(
            expected_state, config={"recursion_limit": 50}
        )


@pytest.mark.asyncio
async def test_one_shot_argv(mock_env_vars, mock_workflow):
    """Non-interactive one-shot mode runs a single task from argv and exits
    without ever reading stdin."""
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", return_value=MOCK_SESSION), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        await main(["one shot task"])

    assert mock_workflow.ainvoke.call_count == 1
    called_state = mock_workflow.ainvoke.call_args[0][0]
    assert called_state["task"] == "one shot task"


@pytest.mark.asyncio
async def test_one_shot_error_result_terminates(mock_env_vars):
    """A one-shot task whose workflow returns error=True must not loop back onto
    the same failing task -- it runs once and exits."""
    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response": {"role": "assistant", "content": "final synthesis failed"},
        "error": True,
    })

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", return_value=MOCK_SESSION), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        await main(["failing task"])

    assert mock_workflow.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_graph_recursion_error_is_survivable(mock_env_vars):
    """A GraphRecursionError is caught and does not crash the CLI."""
    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=GraphRecursionError("limit reached"))

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", return_value=MOCK_SESSION), \
         patch("main.save_session_log", return_value="test_log_file.json"):
        # Should return normally (one-shot mode breaks after handling).
        await main(["a task that never converges"])

    assert mock_workflow.ainvoke.call_count == 1


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
    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response": {"role": "assistant", "content": "Test response"},
        "stage": "__end__",
        "error": False,
    })

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.save_feedback_history") as mock_save, \
         patch("main.create_session_log", return_value=MOCK_SESSION), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=["test task", "y", "Test feedback", "exit"]):
        await main()

        assert mock_workflow.ainvoke.call_count >= 1
        assert mock_save.call_count == 1

        feedback_history = mock_save.call_args[0][0]
        assert len(feedback_history) == 1
        assert feedback_history[0]["feedback"] == "Test feedback"


@pytest.mark.asyncio
async def test_error_handling(mock_env_vars):
    """Test error handling during workflow execution."""
    mock_workflow = MagicMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("sys.exit"), \
         patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=["test task"]), \
         patch("builtins.print") as mock_print:
        try:
            await main()
        except Exception as e:
            assert "Test error" in str(e)

        error_calls = [
            call_args for call_args in mock_print.call_args_list
            if isinstance(call_args[0][0], str) and "error" in call_args[0][0].lower()
        ]
        assert len(error_calls) > 0

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
    """The app exits when an OpenAI-configured model is missing OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("main.ConfigLoader.load_config", return_value={
        "agents": {
            "DLPFC": {
                "models": {
                    "primary": {"provider": "openai", "name": "gpt-4o-mini"}
                }
            }
        }
    }):
        with pytest.raises(SystemExit) as excinfo:
            await main(["test task"])

    assert excinfo.value.code == 1
