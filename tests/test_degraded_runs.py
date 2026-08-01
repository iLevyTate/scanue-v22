"""Tests for partial-failure runs.

A run where specialists fail but MPFC succeeds used to be indistinguishable from
a clean success: `completed: true`, `error: null`, every stage's `error` field
`null`, `agent_errors` absent from the log entirely -- and MPFC synthesized over
the specialists' error strings, which had been stored as their responses and fed
in as "Previous Agent Insights".
"""

from unittest.mock import MagicMock, patch

import pytest

from workflow import _prepare_value_assessment_state, create_workflow

TEST_CONFIG = {
    "agents": {n: {"models": {"primary": {"provider": "openai", "name": "m"}}}
               for n in ("DLPFC", "VMPFC", "OFC", "ACC", "MPFC")}
}
FAILURE_TEXT = "Error processing request: All connection attempts failed"


async def _specialists_fail(self, state):
    if self.agent_name == "DLPFC":
        return {"response": {"role": "assistant",
                             "content": "- VMPFC Agent: YES\n- ACC Agent: YES\n- MPFC Agent: YES"},
                "error": False}
    if self.agent_name == "MPFC":
        return {"response": {"role": "assistant", "content": "Final synthesis."}, "error": False}
    return {"response": {"role": "assistant", "content": FAILURE_TEXT}, "error": True}


@pytest.fixture
def degraded_run():
    patches = [
        patch(f"agents.{mod}.{cls}.process", new=_specialists_fail)
        for mod, cls in (("dlpfc", "DLPFCAgent"), ("specialized", "VMPFCAgent"),
                         ("specialized", "OFCAgent"), ("specialized", "ACCAgent"),
                         ("specialized", "MPFCAgent"))
    ] + [
        patch("utils.config.ConfigLoader.load_config", return_value=TEST_CONFIG),
        patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_failed_agents_are_recorded_in_the_session_log(degraded_run):
    """`agent_errors` was a state-only channel, so a run could finish with
    several failed specialists and leave no trace of them in logs/."""
    final = await create_workflow().ainvoke(
        {"task": "t", "completed_stages": [], "session_log": {"stages": []}},
        config={"recursion_limit": 50},
    )

    assert set(final["session_log"]["agent_errors"]) == {"VMPFC", "ACC"}


@pytest.mark.asyncio
async def test_failed_stages_carry_a_non_null_error(degraded_run):
    """log_stage_end was called without the error argument even when the agent
    reported failure, so every stage's `error` stayed None."""
    final = await create_workflow().ainvoke(
        {"task": "t", "completed_stages": [], "session_log": {"stages": []}},
        config={"recursion_limit": 50},
    )

    errors = {s["stage"]: s["error"] for s in final["session_log"]["stages"]}
    assert errors["emotional_regulation"] == FAILURE_TEXT
    assert errors["conflict_detection"] == FAILURE_TEXT
    assert errors["value_assessment"] is None


@pytest.mark.asyncio
async def test_failed_stage_still_records_its_output(degraded_run):
    """Output and error used to be mutually exclusive, so a failed stage lost
    its output and raw_llm_response -- where the model name lives."""
    final = await create_workflow().ainvoke(
        {"task": "t", "completed_stages": [], "session_log": {"stages": []}},
        config={"recursion_limit": 50},
    )

    failed = next(s for s in final["session_log"]["stages"] if s["stage"] == "emotional_regulation")
    assert failed["output"]["content"] == FAILURE_TEXT


# --------------------------------------------------------------------------- #
# Peer insights
# --------------------------------------------------------------------------- #

def test_failed_agents_are_excluded_from_peer_insights():
    """MPFC used to synthesize over "Error processing request: ..." and present
    the result as the answer."""
    state = {
        "agent_responses": {
            "VMPFC": {"content": FAILURE_TEXT},
            "ACC": {"content": "Real conflict analysis."},
        },
        "agent_errors": {"VMPFC": FAILURE_TEXT},
    }

    insights = _prepare_value_assessment_state(state)["previous_agent_insights"]

    assert "Real conflict analysis." in insights
    assert FAILURE_TEXT not in insights
    assert "VMPFC" not in insights


def test_unavailable_agents_are_named_for_mpfc():
    """MPFC is told what is missing so it can qualify its conclusion."""
    state = {
        "agent_responses": {"ACC": {"content": "ok"}},
        "agent_errors": {"VMPFC": FAILURE_TEXT, "OFC": FAILURE_TEXT},
    }

    assert _prepare_value_assessment_state(state)["unavailable_agents"] == ["OFC", "VMPFC"]


def test_no_peer_insights_when_every_specialist_failed():
    state = {
        "agent_responses": {"VMPFC": {"content": FAILURE_TEXT}},
        "agent_errors": {"VMPFC": FAILURE_TEXT},
    }
    enriched = _prepare_value_assessment_state(state)

    assert "previous_agent_insights" not in enriched
    assert enriched["unavailable_agents"] == ["VMPFC"]


def test_mpfc_is_not_its_own_peer():
    state = {"agent_responses": {"MPFC": {"content": "my own earlier output"}}, "agent_errors": {}}
    assert "previous_agent_insights" not in _prepare_value_assessment_state(state)
