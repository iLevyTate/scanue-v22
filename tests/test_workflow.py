import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from workflow import (
    create_workflow, process_hitl_feedback, NODE_TIMEOUT_SECONDS,
    parse_agent_assignments, _prepare_value_assessment_state, PEER_INSIGHT_CHAR_BUDGET,
    process_task_delegation, process_emotional_regulation,
    process_reward_processing, process_conflict_detection, process_value_assessment,
)
import asyncio
from agents.base import AGENT_LLM_TIMEOUT_SECONDS

# Mock ChatOpenAI at import time
mock_chat_openai = AsyncMock()
mock_chat_openai.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))

TEST_CONFIG = {
    "agents": {
        "DLPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        "VMPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        "OFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        "ACC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
        "MPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}},
    }
}


@pytest.fixture
def mock_env_vars():
    # LLMFactory imports provider SDKs lazily inside each branch, so patch the
    # source module rather than a factory-module attribute.
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
         patch('utils.config.ConfigLoader.load_config', return_value=TEST_CONFIG), \
         patch('langchain_openai.ChatOpenAI', return_value=mock_chat_openai):
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
        "agent_errors": {},
        "completed_stages": [],
        "error": False,
    }


# A full-delegation DLPFC response (structured YES/NO) so parse_agent_assignments
# selects all four specialist stages.
FULL_DELEGATION = "- VMPFC Agent: YES\n- OFC Agent: YES\n- ACC Agent: YES\n- MPFC Agent: YES"


def _ok_response(content="test response"):
    return {"response": {"role": "assistant", "content": content}, "error": False}


async def _dispatch_agent_process(self, state):
    """Stand-in agent.process: DLPFC emits a full delegation; specialists echo."""
    if self.agent_name == "DLPFC":
        return _ok_response(FULL_DELEGATION)
    return _ok_response()


def _patch_all_agents(fn):
    return [
        patch("agents.dlpfc.DLPFCAgent.process", new=fn),
        patch("agents.specialized.VMPFCAgent.process", new=fn),
        patch("agents.specialized.OFCAgent.process", new=fn),
        patch("agents.specialized.ACCAgent.process", new=fn),
        patch("agents.specialized.MPFCAgent.process", new=fn),
    ]


# --------------------------------------------------------------------------- #
# Structure / constants
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_workflow_creation(mock_env_vars, mock_llm):
    """Test workflow creation and structure"""
    workflow = create_workflow()
    assert workflow is not None


def test_outer_timeout_exceeds_inner_timeout():
    """C4: the outer per-node timeout must be strictly greater than the inner LLM
    timeout, otherwise they race and the timeout is reported ambiguously."""
    assert NODE_TIMEOUT_SECONDS > AGENT_LLM_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# HITL feedback (unchanged contract)
# --------------------------------------------------------------------------- #

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
        "error": False,
    }

    feedback = "Test feedback"
    updated_state = process_hitl_feedback(initial_state.copy(), feedback)

    assert updated_state["feedback"] == feedback
    assert len(updated_state["feedback_history"]) == 1
    assert updated_state["previous_response"] == "test response"
    assert id(updated_state) != id(initial_state)


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
        "error": False,
    }

    state = process_hitl_feedback(state, "feedback 1")
    assert len(state["feedback_history"]) == 1
    assert state["feedback_history"][0]["feedback"] == "feedback 1"
    assert state["feedback_history"][0]["response"] == "test response 1"

    state["response"] = {"role": "assistant", "content": "test response 2"}
    state = process_hitl_feedback(state, "feedback 2")
    assert len(state["feedback_history"]) == 2
    assert state["feedback_history"][1]["feedback"] == "feedback 2"
    assert state["feedback_history"][1]["response"] == "test response 2"
    assert state["previous_response"] == "test response 2"


# --------------------------------------------------------------------------- #
# Full-graph execution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_env_vars):
    """Happy path: every delegated specialist runs and MPFC integrates."""
    workflow = create_workflow()
    initial_state = {"task": "test task", "stage": "task_delegation", "completed_stages": []}

    patches = _patch_all_agents(_dispatch_agent_process)
    for p in patches:
        p.start()
    try:
        final_state = await workflow.ainvoke(initial_state, config={"recursion_limit": 50})
    finally:
        for p in patches:
            p.stop()

    assert not final_state.get("error")
    assert "MPFC" in final_state.get("agent_responses", {})
    # All four specialist stages plus task_delegation recorded exactly once.
    assert set(final_state["completed_stages"]) == {
        "task_delegation", "emotional_regulation", "reward_processing",
        "conflict_detection", "value_assessment",
    }


@pytest.mark.asyncio
async def test_c1_regression_always_failing_vmpfc_still_terminates(mock_env_vars):
    """C1: a specialist that ALWAYS raises must not loop forever -- the workflow
    still terminates and downstream stages (MPFC) still execute."""
    async def flaky_process(self, state):
        if self.agent_name == "DLPFC":
            return _ok_response(FULL_DELEGATION)
        if self.agent_name == "VMPFC":
            raise RuntimeError("VMPFC always fails")
        return _ok_response()

    workflow = create_workflow()
    initial_state = {"task": "test task", "stage": "task_delegation", "completed_stages": []}

    patches = _patch_all_agents(flaky_process)
    for p in patches:
        p.start()
    try:
        final_state = await workflow.ainvoke(initial_state, config={"recursion_limit": 50})
    finally:
        for p in patches:
            p.stop()

    # Terminated (no GraphRecursionError) and MPFC ran despite VMPFC failing.
    assert "MPFC" in final_state.get("agent_responses", {})
    assert "VMPFC" in final_state.get("agent_errors", {})
    # VMPFC failure is not the final synthesis stage, so the run is not errored.
    assert not final_state.get("error")
    assert "emotional_regulation" in final_state["completed_stages"]


@pytest.mark.asyncio
async def test_dlpfc_error_is_recorded_and_workflow_continues(mock_env_vars):
    """DLPFC reporting an error -> resilient delegation, run continues, no fatal error."""
    async def dlpfc_errors(self, state):
        if self.agent_name == "DLPFC":
            return {"response": {"role": "assistant", "content": "Error occurred"}, "error": True}
        return _ok_response()

    workflow = create_workflow()
    initial_state = {"task": "test task", "stage": "task_delegation", "completed_stages": []}

    patches = _patch_all_agents(dlpfc_errors)
    for p in patches:
        p.start()
    try:
        final_state = await workflow.ainvoke(initial_state, config={"recursion_limit": 50})
    finally:
        for p in patches:
            p.stop()

    assert final_state.get("agent_errors", {}).get("DLPFC") == "Error occurred"
    assert not final_state.get("error")
    assert "MPFC" in final_state.get("agent_responses", {})


@pytest.mark.asyncio
async def test_workflow_timeout_is_survivable(mock_env_vars):
    """A DLPFC timeout is recorded but the workflow still completes."""
    async def dlpfc_timeout(self, state):
        if self.agent_name == "DLPFC":
            raise TimeoutError("Operation timed out")
        return _ok_response()

    workflow = create_workflow()
    initial_state = {"task": "test task", "stage": "task_delegation", "completed_stages": []}

    patches = _patch_all_agents(dlpfc_timeout)
    for p in patches:
        p.start()
    try:
        final_state = await workflow.ainvoke(initial_state, config={"recursion_limit": 50})
    finally:
        for p in patches:
            p.stop()

    assert "timed out" in final_state.get("agent_errors", {}).get("DLPFC", "").lower()
    assert not final_state.get("error")


@pytest.mark.asyncio
async def test_cancellation_propagates(mock_env_vars, mock_state):
    """CancelledError is a BaseException and must propagate out of a node, not be
    swallowed into a normal error result (tested at the node level; LangGraph
    rewraps it as NodeCancelledError inside a full graph run)."""
    async def dlpfc_cancel(self, state):
        raise asyncio.CancelledError()

    with patch("agents.dlpfc.DLPFCAgent.process", new=dlpfc_cancel):
        with pytest.raises(asyncio.CancelledError):
            await process_task_delegation(mock_state)


# --------------------------------------------------------------------------- #
# Node-level behavior (deltas, completed_stages, no mutation)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_process_task_delegation_success(mock_env_vars, mock_state):
    with patch("agents.dlpfc.DLPFCAgent.process",
               new=AsyncMock(return_value=_ok_response(FULL_DELEGATION))):
        result = await process_task_delegation(mock_state)

    assert not result.get("error")
    assert result["delegated_agents"] == [
        "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment",
    ]
    assert "task_delegation" in result["completed_stages"]


# A DLPFC reply that follows the REQUIRED FORMAT in the agent's own prompt.
SPEC_FORMAT_DLPFC_REPLY = """**AGENT DELEGATION:**
- VMPFC Agent: YES - the decision is emotionally loaded
- OFC Agent: NO
- ACC Agent: YES - the stakeholders want incompatible things
- MPFC Agent: YES - Always needed for final integration

**Analysis:**
Two teams disagree about next quarter's roadmap.

**Subtask Breakdown:**
1. Map each stakeholder position
2. Draft a resolution memo
"""


@pytest.mark.asyncio
async def test_c1_regression_structured_delegation_reaches_the_router(mock_state):
    """C1: the router must honor DLPFC's structured AGENT DELEGATION block.

    `DLPFCAgent._format_response()` rebuilds the reply into a Subtasks /
    Agent Assignments / Integration digest, keeping a bullet only when a
    recognized section header preceded it. The delegation block is headed
    "**AGENT DELEGATION:**", which matches none of those keywords, so every
    "- VMPFC Agent: YES" line is dropped. Parsing that digest instead of the raw
    reply silently collapsed nearly every run to MPFC-only.

    The real agent runs here against a stubbed LLM, so the formatter and the
    workflow wiring are both exercised.
    """
    llm = AsyncMock()
    llm.model_name = "test-model"
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=SPEC_FORMAT_DLPFC_REPLY))

    with patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=llm):
        result = await process_task_delegation(mock_state)

    assert result["delegated_agents"] == [
        "emotional_regulation", "conflict_detection", "value_assessment",
    ]

    # Guard the guard: confirm the formatted digest really does lose the signal,
    # so this test keeps failing if the parse source regresses to the digest.
    digest = result["response"]["content"]
    assert "vmpfc" not in digest.lower()
    assert parse_agent_assignments(digest) == ["value_assessment"]


def test_mpfc_receives_peer_insights_in_full():
    """MPFC is the integration stage, but the peer-insight budget was 200 chars,
    so it saw roughly the first 10-15% of each specialist's analysis, cut
    mid-sentence."""
    vmpfc = "The emotional stakes here are substantial. " * 30
    state = {
        "task": "t",
        "agent_responses": {"VMPFC": {"role": "assistant", "content": vmpfc}},
    }

    insights = _prepare_value_assessment_state(state)["previous_agent_insights"]

    assert len(vmpfc) > 1000  # a realistic specialist response
    assert vmpfc.strip() in insights
    assert "truncated" not in insights


def test_peer_insights_are_truncated_only_when_over_budget():
    """The ellipsis used to be appended unconditionally, so short responses were
    labelled as truncated when they were complete."""
    short = {"task": "t", "agent_responses": {"VMPFC": {"content": "Brief."}}}
    assert "truncated" not in _prepare_value_assessment_state(short)["previous_agent_insights"]

    long = {"task": "t", "agent_responses": {"VMPFC": {"content": "x" * (PEER_INSIGHT_CHAR_BUDGET + 50)}}}
    insights = _prepare_value_assessment_state(long)["previous_agent_insights"]
    assert "[...truncated]" in insights
    assert len(insights) < PEER_INSIGHT_CHAR_BUDGET + 200


@pytest.mark.asyncio
async def test_delegation_delta_propagates_subtasks(mock_env_vars, mock_state):
    """C6: DLPFC parses subtasks, but the delta used to drop the key, so
    state["subtasks"] stayed [] for the whole run and the parsing was dead."""
    parsed = [{"task": "Map stakeholders", "agent": "VMPFC Agent", "category": "general"}]
    dlpfc_result = {**_ok_response(FULL_DELEGATION), "subtasks": parsed}

    with patch("agents.dlpfc.DLPFCAgent.process", new=AsyncMock(return_value=dlpfc_result)):
        result = await process_task_delegation(mock_state)

    assert result["subtasks"] == parsed


@pytest.mark.asyncio
async def test_hitl_feedback_entry_has_both_stage_and_timestamp():
    """C7: the CLI and the workflow used to emit different record shapes -- one
    with `stage` and no `timestamp`, the other the reverse. DLPFC's history
    formatter reads `stage`, so entries from this path rendered 'Stage: unknown'."""
    state = {
        "stage": "value_assessment",
        "response": {"role": "assistant", "content": "final answer"},
        "feedback_history": [],
    }

    updated = process_hitl_feedback(state, "needs more detail")
    entry = updated["feedback_history"][0]

    assert entry["stage"] == "value_assessment"
    assert entry["timestamp"]
    assert entry["feedback"] == "needs more detail"
    assert entry["response"] == "final answer"


@pytest.mark.asyncio
async def test_process_task_delegation_error(mock_env_vars, mock_state):
    with patch("agents.dlpfc.DLPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_task_delegation(mock_state)

    assert result.get("agent_errors", {}).get("DLPFC")
    assert "test error" in result["agent_errors"]["DLPFC"]
    # Resilient delegation + stage marked complete so the router keeps advancing.
    assert result["delegated_agents"] == ["emotional_regulation", "conflict_detection", "value_assessment"]
    assert "task_delegation" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_task_delegation_timeout(mock_env_vars, mock_state):
    async def timeout_process(self, state):
        raise TimeoutError("Operation timed out")

    with patch("agents.dlpfc.DLPFCAgent.process", new=timeout_process):
        result = await process_task_delegation(mock_state)

    assert "timed out" in result["agent_errors"]["DLPFC"].lower()
    assert "task_delegation" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_emotional_regulation_success(mock_env_vars, mock_state):
    with patch("agents.specialized.VMPFCAgent.process",
               new=AsyncMock(return_value=_ok_response("success"))):
        result = await process_emotional_regulation(mock_state)

    assert not result.get("error")
    assert "VMPFC" in result["agent_responses"]
    assert "emotional_regulation" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_emotional_regulation_error(mock_env_vars, mock_state):
    with patch("agents.specialized.VMPFCAgent.process", side_effect=ValueError("test error")):
        result = await process_emotional_regulation(mock_state)

    assert result["agent_errors"]["VMPFC"]
    assert "test error" in result["agent_errors"]["VMPFC"]
    # Non-final stage failure does not mark the whole run errored...
    assert not result.get("error")
    # ...but the stage IS marked complete so the router won't re-dispatch it (C1).
    assert "emotional_regulation" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_reward_processing(mock_env_vars, mock_state):
    with patch("agents.specialized.OFCAgent.process",
               new=AsyncMock(return_value=_ok_response("success"))):
        result = await process_reward_processing(mock_state)
    assert not result.get("error")
    assert "OFC" in result["agent_responses"]
    assert "reward_processing" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_conflict_detection(mock_env_vars, mock_state):
    with patch("agents.specialized.ACCAgent.process",
               new=AsyncMock(return_value=_ok_response("success"))):
        result = await process_conflict_detection(mock_state)
    assert not result.get("error")
    assert "ACC" in result["agent_responses"]
    assert "conflict_detection" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_value_assessment_success(mock_env_vars, mock_state):
    with patch("agents.specialized.MPFCAgent.process",
               new=AsyncMock(return_value=_ok_response("final"))):
        result = await process_value_assessment(mock_state)
    assert not result.get("error")
    assert "MPFC" in result["agent_responses"]
    assert "value_assessment" in result["completed_stages"]


@pytest.mark.asyncio
async def test_process_value_assessment_error_marks_run_errored(mock_env_vars, mock_state):
    """Only the final synthesis stage failing sets top-level error=True."""
    with patch("agents.specialized.MPFCAgent.process", side_effect=ValueError("boom")):
        result = await process_value_assessment(mock_state)
    assert result.get("error") is True
    assert result["agent_errors"]["MPFC"]
    assert "value_assessment" in result["completed_stages"]


@pytest.mark.asyncio
async def test_deltas_do_not_mutate_input_state(mock_env_vars):
    """Nodes must return deltas, never mutate the state dict they were handed."""
    input_state = {
        "task": "t",
        "agent_responses": {},
        "agent_errors": {},
        "completed_stages": [],
    }
    responses_ref = input_state["agent_responses"]
    completed_ref = input_state["completed_stages"]

    with patch("agents.specialized.VMPFCAgent.process",
               new=AsyncMock(return_value=_ok_response("ok"))):
        result = await process_emotional_regulation(input_state)

    # Input untouched (same objects, still empty).
    assert input_state["agent_responses"] is responses_ref and responses_ref == {}
    assert input_state["completed_stages"] is completed_ref and completed_ref == []
    # Delta carries the accumulated values.
    assert result["completed_stages"] == ["emotional_regulation"]
    assert "VMPFC" in result["agent_responses"]
