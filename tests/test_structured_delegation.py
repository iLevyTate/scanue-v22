"""Tests for schema-constrained delegation and routing-source instrumentation.

Asking DLPFC for its routing decision as free text meant the decision had to
survive a reformatter and then a ladder of regex/keyword heuristics. Constraining
generation to a schema removes that entire class of failure; these tests pin the
preferred path, every fallback that keeps older/weaker models working, and the
source label that makes the fallback rate measurable from real runs.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.dlpfc import AgentDelegation, DLPFCAgent
from workflow import (
    parse_agent_assignments_with_source,
    process_task_delegation,
)

TEST_CONFIG = {
    "agents": {"DLPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}}}
}


@pytest.fixture
def agent():
    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")):
        yield DLPFCAgent()


def _structured_llm(returns):
    """A stub shaped like a real chat model: sync with_structured_output, async ainvoke."""
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(return_value=returns)
    llm = MagicMock(model_name="test-model")
    llm.with_structured_output = MagicMock(return_value=runnable)
    return llm


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_to_stages_orders_specialists_with_mpfc_last():
    d = AgentDelegation(vmpfc=True, ofc=True, acc=True)
    assert d.to_stages() == [
        "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment",
    ]


def test_to_stages_always_includes_mpfc():
    assert AgentDelegation(vmpfc=False, ofc=False, acc=False).to_stages() == ["value_assessment"]


def test_to_stages_selects_only_requested_specialists():
    d = AgentDelegation(vmpfc=True, ofc=False, acc=True)
    assert d.to_stages() == ["emotional_regulation", "conflict_detection", "value_assessment"]


# --------------------------------------------------------------------------- #
# Preferred path
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_structured_delegation_is_used_when_available(agent):
    agent.llm = _structured_llm(AgentDelegation(
        vmpfc=True, ofc=False, acc=True,
        reasoning="Emotionally loaded with competing goals.",
        subtasks=["Weigh the personal impact", "Surface the contradiction"],
    ))

    result = await agent.process({"task": "t"})

    assert result["delegation_source"] == "structured_output"
    assert result["delegated_agents"] == [
        "emotional_regulation", "conflict_detection", "value_assessment",
    ]
    assert [s["task"] for s in result["subtasks"]] == [
        "Weigh the personal impact", "Surface the contradiction",
    ]
    assert not result["error"]


@pytest.mark.asyncio
async def test_structured_result_renders_readable_content(agent):
    agent.llm = _structured_llm(AgentDelegation(
        vmpfc=True, ofc=False, acc=False, reasoning="Feelings dominate.", subtasks=["Do a thing"],
    ))

    content = (await agent.process({"task": "t"}))["response"]["content"]

    assert "Do a thing" in content
    assert "VMPFC, MPFC" in content
    assert "Feelings dominate." in content
    assert "OFC" not in content


@pytest.mark.asyncio
async def test_router_consumes_structured_decision_without_parsing(agent):
    """process_task_delegation must not re-derive routing from text when the
    agent already returned a schema-validated decision."""
    dlpfc_result = {
        "response": {"role": "assistant", "content": "irrelevant prose about rewards and costs"},
        "error": False,
        "delegated_agents": ["emotional_regulation", "value_assessment"],
        "delegation_source": "structured_output",
    }

    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock()), \
         patch("agents.dlpfc.DLPFCAgent.process", new=AsyncMock(return_value=dlpfc_result)):
        delta = await process_task_delegation({"task": "t", "completed_stages": []})

    # The content mentions rewards/costs; text parsing would have added OFC.
    assert delta["delegated_agents"] == ["emotional_regulation", "value_assessment"]
    assert delta["delegation_source"] == "structured_output"


# --------------------------------------------------------------------------- #
# Fallbacks -- older/weaker models must keep working
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_falls_back_to_text_when_structured_output_unsupported(agent):
    llm = MagicMock(model_name="m")
    llm.with_structured_output = MagicMock(side_effect=NotImplementedError("no structured output"))
    llm.ainvoke = AsyncMock(return_value=MagicMock(
        content="- VMPFC Agent: YES\n- OFC Agent: NO\n- ACC Agent: NO\n- MPFC Agent: YES"
    ))
    agent.llm = llm

    result = await agent.process({"task": "t"})

    assert "delegated_agents" not in result  # router falls back to parsing
    assert not result["error"]
    llm.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_falls_back_when_schema_validation_fails(agent):
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=ValueError("model emitted invalid JSON"))
    llm = MagicMock(model_name="m")
    llm.with_structured_output = MagicMock(return_value=runnable)
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="VMPFC Agent: YES"))
    agent.llm = llm

    result = await agent.process({"task": "t"})

    assert not result["error"]
    llm.ainvoke.assert_awaited()  # the free-text path ran


@pytest.mark.asyncio
async def test_falls_back_when_result_is_not_the_schema(agent):
    """Guards against a provider returning a dict/sentinel instead of the model."""
    agent_llm = _structured_llm({"vmpfc": True})  # a plain dict, not AgentDelegation
    agent_llm.ainvoke = AsyncMock(return_value=MagicMock(content="ACC Agent: YES"))
    agent.llm = agent_llm

    result = await agent.process({"task": "t"})

    assert not result["error"]
    agent_llm.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_structured_timeout_does_not_trigger_a_second_call(agent):
    """A timeout must propagate, not fall through to another 30s text call --
    that would blow the 45s outer node timeout."""
    async def hang(*a, **k):
        await asyncio.sleep(5)

    runnable = MagicMock()
    runnable.ainvoke = hang
    llm = MagicMock(model_name="m")
    llm.with_structured_output = MagicMock(return_value=runnable)
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="should not be reached"))
    agent.llm = llm

    agent.llm_timeout = 0.01
    result = await agent.process({"task": "t"})

    assert result["error"] is True
    assert "timed out" in result["response"]["content"].lower()
    llm.ainvoke.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("response,expected_source", [
    ("- VMPFC Agent: YES\n- MPFC Agent: YES", "structured_text"),
    ("This is about financial cost and profit.", "semantic"),
    ("", "heuristic"),
])
def test_parse_reports_which_strategy_decided_routing(response, expected_source):
    assert parse_agent_assignments_with_source(response)[1] == expected_source


@pytest.mark.asyncio
async def test_delegation_source_is_recorded_in_the_session_log():
    """The label lands in logs/ so the fallback rate is measurable from real runs."""
    dlpfc_result = {
        "response": {"role": "assistant", "content": "- VMPFC Agent: YES\n- MPFC Agent: YES"},
        "error": False,
    }
    state = {"task": "t", "completed_stages": [], "session_log": {"stages": []}}

    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock()), \
         patch("agents.dlpfc.DLPFCAgent.process", new=AsyncMock(return_value=dlpfc_result)):
        delta = await process_task_delegation(state)

    stage_log = delta["session_log"]["stages"][0]
    assert stage_log["delegation_source"] == "structured_text"
    assert stage_log["delegated_agents"] == ["emotional_regulation", "value_assessment"]


@pytest.mark.asyncio
async def test_resilient_fallback_is_annotated_in_the_session_log():
    """Only the success branch annotated the stage log, so in the exact scenario
    worth debugging -- DLPFC failed and routing was guessed -- the log said
    nothing about who ran or why."""
    state = {"task": "t", "completed_stages": [], "session_log": {"stages": []}}

    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock()), \
         patch("agents.dlpfc.DLPFCAgent.process", side_effect=RuntimeError("provider down")):
        delta = await process_task_delegation(state)

    stage_log = delta["session_log"]["stages"][0]
    assert stage_log["delegation_source"] == "resilient_fallback"
    assert stage_log["delegated_agents"] == [
        "emotional_regulation", "conflict_detection", "value_assessment",
    ]
    assert stage_log["error"]


@pytest.mark.asyncio
async def test_dlpfc_failure_is_labelled_resilient_fallback():
    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock()), \
         patch("agents.dlpfc.DLPFCAgent.process", side_effect=RuntimeError("provider down")):
        delta = await process_task_delegation({"task": "t", "completed_stages": []})

    assert delta["delegation_source"] == "resilient_fallback"
