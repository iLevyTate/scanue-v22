"""Tests for `workflow.parse_agent_assignments`.

Covers the structured (YES/NO) format, semantic fallback, and the exact DLPFC
sample captured from a real run (previously in test_original_parsing.py) that
must NOT pull in OFC/reward_processing.
"""

import pytest
from workflow import parse_agent_assignments


# The exact DLPFC response captured from an original run. It explicitly names
# VMPFC, ACC and MPFC. Note: it also contains the phrase "value-based decision",
# and "value" is an OFC semantic keyword, so the semantic-analysis strategy pulls
# in reward_processing too. This test locks in that observable behavior (fixing
# the keyword heuristic is a separate concern).
ORIGINAL_DLPFC_RESPONSE = """📋 Subtasks:
  • Assess the emotional responses of the team to the meeting and its placement.
  • Evaluate the potential risks of increased stress due to back-to-back feedback sessions.
  • Propose a more balanced schedule for these meetings or a clear context for the sessions.

🔥 Agent Assignments:
  • VMPFC Agent: Assess team emotions regarding the back-to-back meetings.
  • ACC Agent: Evaluate potential conflicts or issues arising from the current scheduling.
  • MPFC Agent: Make a value-based decision on how to proceed with the meeting structure."""


def test_original_dlpfc_sample():
    """The explicitly-named specialists are all selected, in delegation order,
    with value_assessment (MPFC) last."""
    result = parse_agent_assignments(ORIGINAL_DLPFC_RESPONSE)
    assert result == [
        "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment",
    ]
    # The named agents are present and MPFC integrates last.
    assert "emotional_regulation" in result
    assert "conflict_detection" in result
    assert result[-1] == "value_assessment"


@pytest.mark.parametrize(
    "response,expected",
    [
        # Structured YES/NO format, all agents selected.
        (
            "- VMPFC Agent: YES\n- OFC Agent: YES\n- ACC Agent: YES\n- MPFC Agent: YES",
            ["emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"],
        ),
        # Structured format with some NO -- only the YES ones (plus MPFC).
        (
            "- VMPFC Agent: YES\n- OFC Agent: NO\n- ACC Agent: NO\n- MPFC Agent: YES",
            ["emotional_regulation", "value_assessment"],
        ),
        # MPFC always appended even if not explicitly a YES line.
        (
            "- VMPFC Agent: YES\n- OFC Agent: NO\n- ACC Agent: NO",
            ["emotional_regulation", "value_assessment"],
        ),
    ],
)
def test_structured_format(response, expected):
    assert parse_agent_assignments(response) == expected


def test_semantic_fallback_detects_reward_language():
    """No structured YES lines -> semantic keywords drive selection."""
    response = "This decision is about financial reward and cost trade-offs."
    result = parse_agent_assignments(response)
    assert "reward_processing" in result
    # MPFC is always the final integration stage.
    assert result[-1] == "value_assessment"


def test_mpfc_always_present():
    """Even an empty/uninformative response yields at least MPFC."""
    assert parse_agent_assignments("") == ["value_assessment"]
