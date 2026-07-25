"""Tests for token/cost capture, model attribution, and run timing.

Nothing in the app recorded tokens or cost. Every LLM response carries
`usage_metadata` and `response_metadata`, and all of it was dropped at the point
the log record was built -- so there was no notion of spend, and a response cut
off mid-generation (finish_reason="length") was indistinguishable from a
complete one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from agents.base import extract_usage
from workflow import log_stage_start

TEST_CONFIG = {
    "agents": {"VMPFC": {"models": {"primary": {"provider": "ollama", "name": "llama3.2"}}}}
}


def _response(usage=None, metadata=None):
    return SimpleNamespace(
        content="text", usage_metadata=usage or {}, response_metadata=metadata or {}
    )


# --------------------------------------------------------------------------- #
# Usage extraction
# --------------------------------------------------------------------------- #

def test_openai_style_usage_is_captured():
    usage = extract_usage(_response(
        usage={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        metadata={"finish_reason": "stop"},
    ))
    assert usage == {
        "input_tokens": 120, "output_tokens": 40, "total_tokens": 160, "finish_reason": "stop",
    }


def test_ollama_style_usage_is_captured():
    """Ollama reports counts in response_metadata, not usage_metadata."""
    usage = extract_usage(_response(
        metadata={"prompt_eval_count": 300, "eval_count": 50, "done_reason": "stop"},
    ))
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 350
    assert usage["finish_reason"] == "stop"


def test_truncated_response_is_flagged():
    """A response cut off mid-generation reads exactly like a complete one."""
    usage = extract_usage(_response(metadata={"finish_reason": "length"}))
    assert usage["finish_reason"] == "length"


def test_missing_usage_is_not_an_error():
    assert extract_usage(_response()) == {}
    assert extract_usage(SimpleNamespace(content="x")) == {}


# --------------------------------------------------------------------------- #
# Model attribution
# --------------------------------------------------------------------------- #

def test_model_is_stamped_at_stage_start():
    """It used to live only inside raw_llm_response, which is null on failure --
    so the log could not say which model had failed."""
    stage_log = log_stage_start(
        {"session_log": {}}, "emotional_regulation", "VMPFC",
        {"model": "llama3.2", "provider": "ollama"},
    )
    assert stage_log["model"] == "llama3.2"
    assert stage_log["provider"] == "ollama"


def test_agent_reports_its_resolved_model():
    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="llama3.2")):
        from agents.specialized import VMPFCAgent
        descriptor = VMPFCAgent().model_descriptor()

    assert descriptor == {"model": "llama3.2", "provider": "ollama"}


@pytest.mark.asyncio
async def test_dlpfc_records_its_prompt():
    """The stage that decides all routing was the only one logging no prompt."""
    from agents.dlpfc import DLPFCAgent

    config = {"agents": {"DLPFC": {"models": {"primary": {"provider": "ollama", "name": "m"}}}}}
    with patch("utils.config.ConfigLoader.load_config", return_value=config), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")):
        agent = DLPFCAgent()

    llm = MagicMock(model_name="m")
    llm.with_structured_output = MagicMock(side_effect=NotImplementedError)
    llm.with_retry = MagicMock(return_value=llm)

    async def ainvoke(messages):
        return _response(usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    llm.ainvoke = ainvoke
    agent.llm = llm

    await agent.process({"task": "decide something"})

    raw = agent.last_raw_response
    assert raw["prompt"], "DLPFC must record the prompt it sent"
    assert raw["usage"]["total_tokens"] == 15
    assert raw["path"] == "free_text"
    # with_structured_output raised, so no structured call was issued and
    # nothing was billed for it.
    assert raw["structured_attempts"] == 0


@pytest.mark.asyncio
async def test_a_billed_structured_attempt_is_counted():
    """When the structured call is issued but fails validation, the fallback
    makes a SECOND full call. The first one is still billed, and used to produce
    no last_raw_response and no session-log entry at all -- invisible spend on
    every run against a schema-noncompliant model."""
    from agents.dlpfc import DLPFCAgent

    config = {"agents": {"DLPFC": {"models": {"primary": {"provider": "ollama", "name": "m"}}}}}
    with patch("utils.config.ConfigLoader.load_config", return_value=config), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")):
        agent = DLPFCAgent()

    failing = MagicMock()

    async def bad_json(messages):
        raise ValueError("model emitted invalid JSON")

    failing.ainvoke = bad_json
    failing.with_retry = MagicMock(return_value=failing)

    llm = MagicMock(model_name="m")
    llm.with_structured_output = MagicMock(return_value=failing)
    llm.with_retry = MagicMock(return_value=llm)

    async def ainvoke(messages):
        return _response()

    llm.ainvoke = ainvoke
    agent.llm = llm

    await agent.process({"task": "t"})

    assert agent.last_raw_response["structured_attempts"] == 1


# --------------------------------------------------------------------------- #
# Run summary
# --------------------------------------------------------------------------- #

def test_run_summary_aggregates_tokens_and_time():
    session_log = {"stages": [
        {"stage": "task_delegation", "duration_ms": 100,
         "raw_llm_response": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
        {"stage": "value_assessment", "duration_ms": 250,
         "raw_llm_response": {"usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50}}},
    ]}

    summary = main_mod.summarize_run(session_log)

    assert summary["stages_run"] == 2
    assert summary["stage_duration_ms"] == 350
    assert summary["tokens"] == {"input_tokens": 40, "output_tokens": 25, "total_tokens": 65}
    assert "truncated_stages" not in summary


def test_run_summary_flags_truncated_stages():
    session_log = {"stages": [
        {"stage": "value_assessment", "duration_ms": 10,
         "raw_llm_response": {"usage": {"finish_reason": "length"}}},
    ]}
    assert main_mod.summarize_run(session_log)["truncated_stages"] == ["value_assessment"]


def test_run_summary_handles_stages_without_usage():
    session_log = {"stages": [{"stage": "x", "duration_ms": None, "raw_llm_response": None}]}
    summary = main_mod.summarize_run(session_log)
    assert summary["tokens"]["total_tokens"] == 0
    assert summary["stage_duration_ms"] == 0


def test_run_summary_of_an_empty_log():
    assert main_mod.summarize_run({})["stages_run"] == 0
