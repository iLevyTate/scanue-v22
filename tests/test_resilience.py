"""Tests for retry behaviour and configurable timeouts.

`max_retries` is an OpenAI-only constructor argument -- neither ChatOllama nor
HuggingFaceEndpoint declares such a field -- so with the shipped all-Ollama
config the app had zero retry on any failure, and a one-second blip on a local
server failed every stage of the run in turn.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.base import AGENT_LLM_TIMEOUT_SECONDS, resolve_llm_timeout
from agents.factory import LLMFactory

OLLAMA = {"provider": "ollama", "name": "llama3.2"}


# --------------------------------------------------------------------------- #
# Timeout resolution
# --------------------------------------------------------------------------- #

def test_configured_timeout_is_honoured():
    """A user setting `timeout: 120` for a slow local model still got killed at
    the hardcoded 30s, with no way to change it short of editing source."""
    assert resolve_llm_timeout({"timeout": 120}) == 120.0


def test_timeout_falls_back_to_the_default():
    for config in ({}, {"timeout": None}, {"timeout": "abc"}, {"timeout": 0}, {"timeout": -5}):
        assert resolve_llm_timeout(config) == AGENT_LLM_TIMEOUT_SECONDS


def test_agent_picks_up_its_configured_timeout():
    config = {"agents": {"DLPFC": {"models": {"primary": {**OLLAMA, "timeout": 90}}}}}
    with patch("utils.config.ConfigLoader.load_config", return_value=config):
        from agents.dlpfc import DLPFCAgent
        assert DLPFCAgent().llm_timeout == 90.0


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #

def test_non_openai_providers_get_retry():
    llm = LLMFactory.create_llm(OLLAMA)
    wrapped = LLMFactory.wrap_with_retry(llm, OLLAMA)

    assert type(wrapped).__name__ == "RunnableRetry"
    assert wrapped is not llm


def test_openai_is_not_double_wrapped():
    """Its client already retries natively, with rate-limit awareness."""
    config = {"provider": "openai", "name": "gpt-4o-mini"}
    llm = MagicMock()
    assert LLMFactory.wrap_with_retry(llm, config) is llm


def test_retry_can_be_disabled():
    llm = LLMFactory.create_llm(OLLAMA)
    for config in ({**OLLAMA, "max_retries": 0}, {**OLLAMA, "max_retries": None}):
        assert LLMFactory.wrap_with_retry(llm, config) is llm


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_and_succeeds():
    """The behaviour that matters: one blip no longer fails the stage."""
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("All connection attempts failed")
        return "recovered"

    runnable = MagicMock()
    runnable.with_retry = MagicMock(side_effect=lambda **kw: _RetryStub(flaky, kw))

    wrapped = LLMFactory.wrap_with_retry(runnable, {**OLLAMA, "max_retries": 3})
    assert await wrapped.ainvoke("x") == "recovered"
    assert calls["n"] == 3


class _RetryStub:
    """Minimal stand-in for RunnableRetry so the test does not sleep on backoff."""

    def __init__(self, fn, kwargs):
        self._fn = fn
        self._attempts = kwargs["stop_after_attempt"]

    async def ainvoke(self, *args, **kwargs):
        last = None
        for _ in range(self._attempts):
            try:
                return await self._fn(*args, **kwargs)
            except Exception as e:  # noqa: PERF203 - mirrors RunnableRetry semantics
                last = e
        raise last


def test_attempts_include_the_initial_try():
    """`max_retries: 3` must mean 3 retries after the first attempt, not 3 total."""
    runnable = MagicMock()
    LLMFactory.wrap_with_retry(runnable, {**OLLAMA, "max_retries": 3})
    assert runnable.with_retry.call_args.kwargs["stop_after_attempt"] == 4


@pytest.mark.asyncio
async def test_agent_invokes_through_the_retry_wrapper():
    """Retry is applied at invocation, not in create_llm: `.with_retry()` returns
    a RunnableRetry, which is not a chat model and would break model_name
    introspection and with_structured_output()."""
    config = {"agents": {"VMPFC": {"models": {"primary": OLLAMA}}}}
    with patch("utils.config.ConfigLoader.load_config", return_value=config), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")):
        from agents.specialized import VMPFCAgent
        agent = VMPFCAgent()

    # The model itself is untouched, so introspection still works...
    assert agent.llm.model_name == "m"
    # ...while the thing actually invoked is wrapped.
    with patch.object(LLMFactory, "wrap_with_retry", return_value=MagicMock()) as wrap:
        agent.invoker()
    wrap.assert_called_once()


# --------------------------------------------------------------------------- #
# Outer node timeout follows the configured inner timeout
# --------------------------------------------------------------------------- #

def test_node_timeout_honours_a_long_configured_timeout():
    """The shipped config sets `timeout: 120`, but every node used to be killed
    by a fixed 45s outer timeout -- so the configured value never took effect."""
    from workflow import node_timeout_for

    agent = MagicMock(llm_timeout=120.0, LLM_CALLS_PER_PROCESS=1)
    assert node_timeout_for(agent) > 120.0


def test_node_timeout_covers_both_dlpfc_calls():
    """DLPFC can make a structured call and then a free-text fallback, each with
    the full inner budget; two 30s calls used to overrun the 45s node cap."""
    from agents.dlpfc import DLPFCAgent
    from workflow import node_timeout_for

    agent = MagicMock(llm_timeout=30.0, LLM_CALLS_PER_PROCESS=DLPFCAgent.LLM_CALLS_PER_PROCESS)
    assert node_timeout_for(agent) > 2 * 30.0


def test_node_timeout_keeps_its_floor_and_tolerates_odd_agents():
    from workflow import NODE_TIMEOUT_SECONDS, node_timeout_for

    assert node_timeout_for(MagicMock(llm_timeout=5.0, LLM_CALLS_PER_PROCESS=1)) == NODE_TIMEOUT_SECONDS
    # A bare mock has non-numeric attributes; fall back to the defaults.
    assert node_timeout_for(MagicMock()) > AGENT_LLM_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_outer_timeout_error_is_not_blank():
    """str(TimeoutError()) is empty, so an outer timeout used to be reported as
    'Error in task delegation: ' with no detail at all."""
    from workflow import process_task_delegation

    config = {"agents": {"DLPFC": {"models": {"primary": {"provider": "ollama", "name": "m"}}}}}

    async def bare_timeout(self, state):
        raise TimeoutError()

    with patch("utils.config.ConfigLoader.load_config", return_value=config), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")), \
         patch("agents.dlpfc.DLPFCAgent.process", new=bare_timeout):
        delta = await process_task_delegation({"task": "t", "completed_stages": []})

    assert "timed out after" in delta["agent_errors"]["DLPFC"]
