import json
import pathlib

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langgraph.errors import GraphRecursionError

import main as main_mod
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


def _mock_session():
    """Build a fresh session log.

    `main()` mutates the session log it is given (`completed`, `final_response`,
    `error`), so a module-level dict leaked state between tests and made
    assert_called_with compare the mutated object against itself.
    """
    return {
        "task": "test task",
        "timestamp": "2023-01-01T00:00:00.000000",
        "session_id": "test-session-id",
        "stages": [],
        "final_response": None,
        "user_feedback": None,
        "error": None,
        "completed": False,
    }


@pytest.fixture
def mock_session():
    return _mock_session()


def test_session_log_survives_unserializable_values(tmp_path, monkeypatch):
    """A value json can't encode used to truncate the log file mid-write.

    json.dump() streams to the file handle, so it raised partway through and
    left an unparseable file behind while save_session_log swallowed the error
    and reported failure -- losing the whole run's diagnostics.
    """
    monkeypatch.setattr(main_mod, "LOGS_DIRECTORY", str(tmp_path / "logs"))

    log = _mock_session()
    log["stages"] = [{"stage": "task_delegation", "raw_llm_response": object()}]

    filename = main_mod.save_session_log(log)

    assert filename is not None
    written = json.loads(pathlib.Path(filename).read_text())  # valid JSON
    assert written["session_id"] == "test-session-id"
    assert written["stages"][0]["stage"] == "task_delegation"


def test_feedback_history_write_is_not_partial(tmp_path, monkeypatch):
    path = tmp_path / "feedback_history.json"
    monkeypatch.setattr(main_mod, "FEEDBACK_HISTORY_FILE", str(path))

    main_mod.save_feedback_history([{"feedback": "ok", "response": object()}])

    assert json.loads(path.read_text())[0]["feedback"] == "ok"


@pytest.mark.asyncio
async def test_app_initialization(mock_env_vars, mock_workflow):
    """The initial workflow state carries the new keys and ainvoke gets the
    recursion_limit config."""
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("builtins.input", side_effect=["test task", "n", "exit"]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", return_value="test_log_file.json"):
        await main()

    args, kwargs = mock_workflow.ainvoke.call_args
    state = args[0]

    # main() mutates the session log it is handed, so compare the rest of the
    # state by value and assert the session log separately by identity.
    assert {k: v for k, v in state.items() if k != "session_log"} == {
        "task": "test task",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "completed_stages": [],
        "error": False,
    }
    assert state["session_log"]["session_id"] == "test-session-id"
    assert kwargs == {"config": {"recursion_limit": 50}}


@pytest.mark.asyncio
async def test_one_shot_argv(mock_env_vars, mock_workflow):
    """Non-interactive one-shot mode runs a single task from argv and exits
    without ever reading stdin."""
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
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
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
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
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
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
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=["test task", "y", "Test feedback", "exit"]):
        await main()

        assert mock_workflow.ainvoke.call_count >= 1
        assert mock_save.call_count == 1

        feedback_history = mock_save.call_args[0][0]
        assert len(feedback_history) == 1
        assert feedback_history[0]["feedback"] == "Test feedback"


@pytest.mark.asyncio
async def test_error_handling_keeps_interactive_session_alive(mock_env_vars):
    """A workflow exception is reported but does not tear down the CLI.

    A transient provider error used to propagate out of `main()` and kill the
    whole interactive session; only GraphRecursionError was survivable. The user
    should be able to try another task instead.
    """
    mock_workflow = MagicMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=["test task", "exit"]), \
         patch("builtins.print") as mock_print:
        # Returns normally: the loop survives and the user types 'exit'.
        await main()

        error_calls = [
            call_args for call_args in mock_print.call_args_list
            if isinstance(call_args[0][0], str) and "error" in call_args[0][0].lower()
        ]
        assert len(error_calls) > 0

    assert mock_workflow.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_one_shot_workflow_exception_exits_nonzero(mock_env_vars):
    """The same failure in one-shot mode exits non-zero so scripts/CI detect it."""
    mock_workflow = MagicMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        with pytest.raises(SystemExit) as excinfo:
            await main(["a task"])

    assert excinfo.value.code == 1
    assert mock_workflow.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_empty_argv_task_exits_instead_of_looping(mock_env_vars, mock_workflow):
    """C2 regression: `python main.py ""` used to spin forever.

    One-shot mode re-reads argv[0] on every iteration, so an empty task hit
    `continue` with no exit condition and looped printing the same error.
    """
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        with pytest.raises(SystemExit) as excinfo:
            await main([""])

    assert excinfo.value.code == 1
    mock_workflow.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_one_shot_task_is_stripped(mock_env_vars, mock_workflow):
    """Whitespace-only argv is empty, and a padded task is trimmed.

    Only the interactive branch used to call .strip().
    """
    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        with pytest.raises(SystemExit):
            await main(["   "])
    mock_workflow.ainvoke.assert_not_called()

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", return_value="test_log_file.json"), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        await main(["  padded task  "])
    assert mock_workflow.ainvoke.call_args[0][0]["task"] == "padded task"


@pytest.mark.asyncio
async def test_errored_run_is_not_logged_as_completed(mock_env_vars):
    """C10: `completed` was set True before the error check, so failed runs were
    recorded in logs/ as successful."""
    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response": {"role": "assistant", "content": "final synthesis failed"},
        "error": True,
    })
    saved = {}

    with patch("main.create_workflow", return_value=mock_workflow), \
         patch("main.load_feedback_history", return_value=[]), \
         patch("main.create_session_log", side_effect=lambda task: _mock_session()), \
         patch("main.save_session_log", side_effect=lambda log: saved.update(log) or "f.json"), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read in one-shot mode")):
        await main(["failing task"])

    assert saved["completed"] is False
    assert saved["error"] == "final synthesis failed"


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
