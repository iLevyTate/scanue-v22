"""Unit tests for the module-level router `workflow.get_next_stage`.

The router picks the first delegated stage that has not yet completed, else END.
Because every node appends itself to `completed_stages` on success AND failure,
this loop always advances -- these tests lock in that termination guarantee.
"""

from workflow import END, get_next_stage

DELEGATED = ["emotional_regulation", "conflict_detection", "value_assessment"]


def test_full_sequence_advances_in_order():
    """With nothing completed, the router returns the first delegated stage;
    as stages complete it walks through them in order and finally ends."""
    completed = []
    sequence = []
    # Simulate the graph completing each stage the router hands back.
    for _ in range(len(DELEGATED) + 1):
        nxt = get_next_stage({"delegated_agents": DELEGATED, "completed_stages": list(completed)})
        sequence.append(nxt)
        if nxt == END:
            break
        completed.append(nxt)

    assert sequence == [
        "emotional_regulation",
        "conflict_detection",
        "value_assessment",
        END,
    ]


def test_completed_stage_is_skipped_even_after_error():
    """A stage that failed still lands in completed_stages, so the router must
    skip it (this is the C1 infinite-loop fix)."""
    state = {
        "delegated_agents": DELEGATED,
        # emotional_regulation completed (via an error path) -> must be skipped
        "completed_stages": ["emotional_regulation"],
    }
    assert get_next_stage(state) == "conflict_detection"


def test_empty_delegation_ends():
    assert get_next_stage({"delegated_agents": [], "completed_stages": []}) == END
    # Missing keys entirely should also end cleanly.
    assert get_next_stage({}) == END


def test_all_completed_ends():
    state = {
        "delegated_agents": DELEGATED,
        "completed_stages": list(DELEGATED),
    }
    assert get_next_stage(state) == END


def test_completed_stages_order_does_not_matter():
    """The router uses set membership, so out-of-order completion still resolves
    to the first *delegated* stage that is missing."""
    state = {
        "delegated_agents": DELEGATED,
        "completed_stages": ["value_assessment", "emotional_regulation"],
    }
    assert get_next_stage(state) == "conflict_detection"
