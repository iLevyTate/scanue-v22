"""Tests for the specialist agents and their prompts.

These four classes were near-identical copies -- byte-identical prompt bodies and
`process` overrides that did nothing but call super() -- and the prompts were
one-line stubs with no output format, criteria, or length guidance. MPFC in
particular never mentioned integration or its peers, despite being the
integration stage.

Assertions are structural rather than prose-matching so the wording stays
editable without breaking the suite.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.base import BaseAgent
from agents.specialized import (
    ACCAgent,
    MPFCAgent,
    OFCAgent,
    SPECIALIST_CLASSES,
    VMPFCAgent,
)

CONFIG = {
    "agents": {n: {"models": {"primary": {"provider": "ollama", "name": "m"}}}
               for n in ("VMPFC", "OFC", "ACC", "MPFC")}
}


@pytest.fixture
def agents():
    with patch("utils.config.ConfigLoader.load_config", return_value=CONFIG), \
         patch("agents.factory.LLMFactory.create_llm", return_value=MagicMock(model_name="m")):
        yield {name: cls() for name, cls in SPECIALIST_CLASSES.items()}


def _template(agent):
    return str(agent.prompt.messages[0].prompt.template)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_all_specialists_are_base_agents(agents):
    assert all(isinstance(a, BaseAgent) for a in agents.values())


def test_agent_names_are_wired_correctly(agents):
    assert {name: a.agent_name for name, a in agents.items()} == {
        "VMPFC": "VMPFC", "OFC": "OFC", "ACC": "ACC", "MPFC": "MPFC",
    }


def test_class_names_are_preserved(agents):
    """Generated from a table, but they must still be importable under the names
    the workflow and tests reference."""
    assert [c.__name__ for c in (VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent)] == [
        "VMPFCAgent", "OFCAgent", "ACCAgent", "MPFCAgent",
    ]


def test_specialists_do_not_override_process():
    """The four overrides were pure `return await super().process(state)`,
    forced by @abstractmethod decorating an already-implemented method."""
    for cls in SPECIALIST_CLASSES.values():
        assert "process" not in cls.__dict__


@pytest.mark.asyncio
async def test_inherited_process_still_works(agents):
    """Removing the overrides must not change behaviour."""
    agent = agents["VMPFC"]
    response = MagicMock(content="analysis", usage_metadata={}, response_metadata={})

    async def ainvoke(messages):
        return response

    agent.llm.ainvoke = ainvoke
    agent.llm.with_retry = MagicMock(return_value=agent.llm)

    result = await agent.process({"task": "t"})
    assert result["response"]["content"] == "analysis"
    assert not result["error"]


# --------------------------------------------------------------------------- #
# Prompt content
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["VMPFC", "OFC", "ACC", "MPFC"])
def test_every_prompt_carries_the_shared_context_slots(agents, name):
    template = _template(agents[name])
    for slot in ("{task}", "{state}", "{previous_response}", "{feedback}", "{feedback_history}"):
        assert slot in template, f"{name} prompt is missing {slot}"


@pytest.mark.parametrize("name", ["VMPFC", "OFC", "ACC", "MPFC"])
def test_prompts_are_substantive_not_one_line_stubs(agents, name):
    """The originals were a single sentence of instruction."""
    template = _template(agents[name])
    assert len(template.splitlines()) > 10
    # Numbered analysis structure, not a bare instruction.
    assert "1." in template and "2." in template


@pytest.mark.parametrize("name", ["VMPFC", "OFC", "ACC", "MPFC"])
def test_prompts_bound_their_output_length(agents, name):
    assert "words" in _template(agents[name]).lower()


def test_prompts_are_distinct(agents):
    """The four prompt bodies used to be byte-identical."""
    templates = {_template(a) for a in agents.values()}
    assert len(templates) == 4


def test_mpfc_is_told_to_integrate_its_peers(agents):
    """The core defect: the integration stage never mentioned integration,
    synthesis, or the peer analysis assembled for it."""
    template = _template(agents["MPFC"]).lower()

    assert "synthesize" in template or "integration" in template
    assert "other agents" in template
    # Must attribute, and must resolve disagreement rather than average it.
    assert "naming the" in template or "name" in template
    assert "disagree" in template


def test_mpfc_handles_missing_peers(agents):
    """It is told when a specialist failed, so it can qualify its confidence."""
    assert "unavailable" in _template(agents["MPFC"]).lower()


def test_mpfc_requires_a_committed_recommendation(agents):
    template = _template(agents["MPFC"])
    assert "RECOMMENDATION" in template
    assert "NEXT STEP" in template


@pytest.mark.parametrize("name,keyword", [
    ("VMPFC", "emotional"),
    ("OFC", "cost"),
    ("ACC", "conflict"),
])
def test_each_specialist_prompt_reflects_its_specialty(agents, name, keyword):
    assert keyword in _template(agents[name]).lower()
