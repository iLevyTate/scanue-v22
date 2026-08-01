"""Tests for the real-provider validation harness.

The script itself needs a live model, but its analysis is a pure function over a
session log -- so the checks that will be trusted to green-light a merge are
themselves verified here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate import FAIL, PASS, SKIP, WARN, analyze  # noqa: E402


def _log(**overrides):
    log = {
        "completed": True,
        "error": None,
        "agent_errors": {},
        "summary": {"tokens": {"total_tokens": 1234}},
        "final_response": {"content": "answer"},
        "stages": [
            {
                "stage": "task_delegation", "agent": "DLPFC", "model": "llama3.2",
                "delegation_source": "structured_output",
                "delegated_agents": ["emotional_regulation", "value_assessment"],
                "raw_llm_response": {"prompt_chars": 2000, "metadata": {"num_ctx": 8192}},
            },
            {
                "stage": "emotional_regulation", "agent": "VMPFC", "model": "llama3.2",
                "raw_llm_response": {"prompt_chars": 2000, "metadata": {"num_ctx": 8192}},
            },
            {
                "stage": "value_assessment", "agent": "MPFC", "model": "llama3.2",
                "raw_llm_response": {"prompt_chars": 4000, "metadata": {"num_ctx": 8192}},
            },
        ],
    }
    log.update(overrides)
    return log


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_a_clean_run_passes_everything():
    assert all(c.status == PASS for c in analyze(_log()))


def test_empty_log_fails_fast():
    checks = analyze({"stages": []})
    assert len(checks) == 1
    assert checks[0].status == FAIL


# --------------------------------------------------------------------------- #
# The check that stubs cannot make
# --------------------------------------------------------------------------- #

def test_total_provider_failure_reports_the_real_cause():
    """When nothing reached a model, "0 tokens" is a symptom, not the defect.
    Reporting it as a token-capture failure would send you debugging
    extract_usage() instead of starting your server."""
    log = _log(
        completed=False,
        agent_errors=dict.fromkeys(("DLPFC", "VMPFC", "MPFC"), "All connection attempts failed"),
        summary={"tokens": {"total_tokens": 0}},
    )
    checks = analyze(log)

    assert _status(checks, "provider reachable") == FAIL
    # The downstream checks are skipped, not reported as failures.
    for name in ("token capture", "delegation source", "context headroom"):
        assert _status(checks, name) == SKIP
    assert not any(c.status == WARN for c in checks)


def test_partial_failure_still_evaluates_everything():
    """One agent down is a degraded run, not an unreachable provider."""
    log = _log(agent_errors={"VMPFC": "connection refused"})
    checks = analyze(log)

    assert not any(c.name == "provider reachable" for c in checks)
    assert _status(checks, "token capture") == PASS
    assert _status(checks, "agent failures") == WARN


def test_specialists_that_all_failed_do_not_count_as_run():
    log = _log(agent_errors={"VMPFC": "boom"})
    assert _status(analyze(log), "specialists ran") == WARN


def test_zero_tokens_is_a_hard_failure():
    """Provider metadata shapes differ; a mismatch yields zero rather than an
    error, so this is the whole reason the script exists."""
    checks = analyze(_log(summary={"tokens": {"total_tokens": 0}}))
    assert _status(checks, "token capture") == FAIL


def test_missing_summary_is_a_hard_failure():
    assert _status(analyze(_log(summary={})), "token capture") == FAIL


# --------------------------------------------------------------------------- #
# Delegation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source,expected", [
    ("structured_output", PASS),
    ("structured_text", WARN),
    ("semantic", WARN),
    ("heuristic", WARN),
    ("resilient_fallback", WARN),
])
def test_delegation_source_grading(source, expected):
    log = _log()
    log["stages"][0]["delegation_source"] = source
    assert _status(analyze(log), "delegation source") == expected


def test_absent_delegation_source_fails():
    log = _log()
    del log["stages"][0]["delegation_source"]
    assert _status(analyze(log), "delegation source") == FAIL


# --------------------------------------------------------------------------- #
# Degradation and truncation
# --------------------------------------------------------------------------- #

def test_agent_failures_are_surfaced():
    checks = analyze(_log(agent_errors={"VMPFC": "connection refused"}))
    assert _status(checks, "agent failures") == WARN


def test_truncated_output_is_surfaced():
    log = _log()
    log["summary"]["truncated_stages"] = ["value_assessment"]
    assert _status(analyze(log), "output truncation") == WARN


def test_incomplete_run_fails():
    assert _status(analyze(_log(completed=False)), "run completed") == FAIL


def test_mpfc_only_run_warns():
    log = _log()
    log["stages"] = [s for s in log["stages"] if s["stage"] != "emotional_regulation"]
    assert _status(analyze(log), "specialists ran") == WARN


# --------------------------------------------------------------------------- #
# Context headroom -- Ollama drops silently past num_ctx
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prompt_chars,num_ctx,expected", [
    (2_000, 8192, PASS),    # ~500 tok of 8192
    (28_000, 8192, WARN),   # ~7000 tok of 8192 -> 85%
    (40_000, 8192, FAIL),   # over the window
])
def test_context_headroom_grading(prompt_chars, num_ctx, expected):
    log = _log()
    log["stages"][2]["raw_llm_response"] = {
        "prompt_chars": prompt_chars, "metadata": {"num_ctx": num_ctx}
    }
    assert _status(analyze(log), "context headroom") == expected


def test_headroom_is_skipped_when_num_ctx_is_unknown():
    """OpenAI has no num_ctx; the check should simply not appear."""
    log = _log()
    for stage in log["stages"]:
        stage["raw_llm_response"] = {"prompt_chars": 2000, "metadata": {}}
    assert not any(c.name == "context headroom" for c in analyze(log))


def test_missing_model_attribution_fails():
    log = _log()
    del log["stages"][1]["model"]
    assert _status(analyze(log), "model attribution") == FAIL
