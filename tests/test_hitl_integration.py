"""HITL (Human-In-The-Loop) integration tests.

These run fully offline: no workflow is invoked and no provider is contacted.
The feedback-history file is redirected to a temporary path so tests never write
the repo-root `feedback_history.json`.
"""

import pytest

import main
from main import load_feedback_history, save_feedback_history
from workflow import process_hitl_feedback


@pytest.fixture
def temp_feedback_file(tmp_path, monkeypatch):
    """Point the feedback-history persistence at a temp file for the test."""
    path = tmp_path / "feedback_history.json"
    monkeypatch.setattr(main, "FEEDBACK_HISTORY_FILE", str(path))
    return path


def test_feedback_persistence(temp_feedback_file):
    """Feedback saved to disk is loaded back intact."""
    test_feedback = [
        {
            "response": "Test response 1",
            "feedback": "This response was helpful but could be more detailed.",
            "stage": "value_assessment",
        },
        {
            "response": "Test response 2",
            "feedback": "Please provide more specific recommendations.",
            "stage": "conflict_detection",
        },
    ]

    save_feedback_history(test_feedback)
    loaded_feedback = load_feedback_history()

    assert loaded_feedback == test_feedback
    # And it really went to the temp file, not the repo root.
    assert temp_feedback_file.exists()


def test_feedback_processing():
    """process_hitl_feedback records feedback and updates previous_response."""
    test_state = {
        "task": "Test task",
        "stage": "emotional_regulation",
        "response": {"role": "assistant", "content": "Test response content"},
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False,
    }

    feedback_text = "This analysis needs more emotional context."
    updated_state = process_hitl_feedback(test_state, feedback_text)

    assert len(updated_state["feedback_history"]) == 1
    assert updated_state["feedback"] == feedback_text
    assert updated_state["previous_response"] == "Test response content"
    assert "timestamp" in updated_state["feedback_history"][0]


def test_agent_prompt_integration(mock_env_vars):
    """DLPFC formats feedback history and specialists include it in their prompts."""
    from agents.dlpfc import DLPFCAgent
    from agents.specialized import ACCAgent, MPFCAgent, VMPFCAgent

    test_feedback_history = [
        {
            "response": "Previous analysis was too general",
            "feedback": "Please be more specific about actionable steps",
            "stage": "value_assessment",
        }
    ]

    dlpfc = DLPFCAgent()
    formatted_feedback = dlpfc._format_feedback_history(test_feedback_history)
    assert "Previous analysis was too general" in formatted_feedback
    assert "Please be more specific" in formatted_feedback

    for agent_class in (VMPFCAgent, ACCAgent, MPFCAgent):
        agent = agent_class()
        prompt_messages = agent.prompt.messages
        template_content = str(prompt_messages[0].prompt.template) if prompt_messages else ""
        assert "Feedback History: {feedback_history}" in template_content


@pytest.fixture
def mock_env_vars():
    """Provide a consistent OpenAI test configuration for agent construction."""
    from unittest.mock import AsyncMock, MagicMock, patch

    test_config = {
        "agents": {
            "DLPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "VMPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "OFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "ACC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
            "MPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        }
    }
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))

    # LLMFactory imports provider SDKs lazily inside each branch, so patch the
    # source module rather than a factory-module attribute.
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("utils.config.ConfigLoader.load_config", return_value=test_config), \
         patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
        yield
