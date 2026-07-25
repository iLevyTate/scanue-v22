"""Tests for the bounds that keep a long-lived install from breaking itself.

The feedback history is persisted across sessions, grows monotonically, and was
injected in full into all 5-6 LLM calls of every run. At ~1,250 chars per entry,
25 entries filled ~95% of an 8k context and 50 exceeded it -- at which point
every agent fails at once with a generic provider error.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import main as main_mod
from agents.base import (
    FEEDBACK_CHAR_BUDGET,
    FEEDBACK_MAX_ENTRIES,
    format_feedback_history,
)
from agents.factory import LLMFactory


def _entry(n=0):
    return {
        "stage": "value_assessment",
        "response": f"Answer {n}. " + "Taking the role is likely better. " * 20,
        "feedback": f"Feedback number {n}.",
    }


# --------------------------------------------------------------------------- #
# Feedback history
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("count", [10, 50, 100, 500])
def test_feedback_history_is_bounded_regardless_of_size(count):
    """Previously linear in the number of entries, with no ceiling."""
    rendered = format_feedback_history([_entry(i) for i in range(count)])
    assert len(rendered) <= FEEDBACK_CHAR_BUDGET + 40  # + truncation marker


def test_only_the_most_recent_entries_are_kept():
    """Recent feedback is also the most relevant, so the window is not purely a
    cost measure."""
    rendered = format_feedback_history([_entry(i) for i in range(20)])

    assert "Feedback number 19." in rendered
    assert "Feedback number 0." not in rendered


def test_truncation_is_announced_not_silent():
    rendered = format_feedback_history([_entry(i) for i in range(20)])
    assert f"showing the {FEEDBACK_MAX_ENTRIES} most recent of 20" in rendered


def test_short_history_is_rendered_whole_without_a_notice():
    rendered = format_feedback_history([_entry(0)])
    assert "most recent" not in rendered
    assert "Feedback number 0." in rendered


def test_empty_history_still_returns_the_placeholder():
    assert format_feedback_history([]) == "No previous feedback"


def test_oversized_single_response_is_clipped():
    huge = {"stage": "s", "response": "x" * 50_000, "feedback": "ok"}
    rendered = format_feedback_history([huge])

    assert len(rendered) <= FEEDBACK_CHAR_BUDGET + 40
    assert "truncated" in rendered


def test_non_dict_entries_do_not_crash():
    assert "junk" in format_feedback_history(["junk"])


# --------------------------------------------------------------------------- #
# State paths
# --------------------------------------------------------------------------- #

def test_state_paths_are_anchored_to_the_project_root():
    """As bare relative paths these fragmented across directories: launching from
    elsewhere silently started an empty history and a second logs/ tree."""
    assert Path(main_mod.FEEDBACK_HISTORY_FILE).is_absolute()
    assert Path(main_mod.LOGS_DIRECTORY).is_absolute()
    assert Path(main_mod.FEEDBACK_HISTORY_FILE).parent == main_mod.STATE_DIR


def test_log_retention_prunes_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "LOGS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(main_mod, "LOG_RETENTION_COUNT", 3)

    for i in range(6):
        p = tmp_path / f"session_{i}.json"
        p.write_text("{}")
        # Ascending mtimes so ordering is deterministic.
        import os
        os.utime(p, (1_000_000 + i, 1_000_000 + i))

    main_mod._prune_old_logs()

    remaining = sorted(p.name for p in tmp_path.glob("session_*.json"))
    assert remaining == ["session_3.json", "session_4.json", "session_5.json"]


def test_log_retention_disabled_by_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "LOGS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(main_mod, "LOG_RETENTION_COUNT", 0)
    for i in range(4):
        (tmp_path / f"session_{i}.json").write_text("{}")

    main_mod._prune_old_logs()

    assert len(list(tmp_path.glob("session_*.json"))) == 4


def test_pruning_never_raises(monkeypatch):
    """Losing a run because cleanup failed would be worse than a stale file."""
    monkeypatch.setattr(main_mod, "LOGS_DIRECTORY", "/nonexistent/nope")
    monkeypatch.setattr(main_mod, "LOG_RETENTION_COUNT", 3)
    main_mod._prune_old_logs()  # must not raise


# --------------------------------------------------------------------------- #
# Context window
# --------------------------------------------------------------------------- #

def test_num_ctx_reaches_the_ollama_client():
    """num_ctx defaults to None, so the server applies its own default (~2048
    tokens) and silently DROPS anything past it -- no error, no log line."""
    llm = LLMFactory.create_llm(
        {"provider": "ollama", "name": "llama3.2", "num_ctx": 8192, "max_tokens": 512}
    )
    assert llm.num_ctx == 8192
    assert llm.num_predict == 512


def test_ollama_omits_context_settings_when_unconfigured():
    llm = LLMFactory.create_llm({"provider": "ollama", "name": "llama3.2"})
    assert llm.num_ctx is None
    assert llm.num_predict is None


def test_openai_max_tokens_is_configurable():
    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        LLMFactory.create_llm({"name": "gpt-4o-mini", "api_key": "k", "max_tokens": 1024})
    assert chat_openai.call_args.kwargs["max_tokens"] == 1024
