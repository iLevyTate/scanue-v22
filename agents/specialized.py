"""The four specialist Prefrontal Cortex agents.

These classes were four near-identical copies differing only in a name and a
one-line instruction, with byte-identical prompt bodies and `process` overrides
that were pure `return await super().process(state)` pass-throughs. They are now
generated from a single table, mirroring the STAGE_AGENTS idiom in workflow.py.

The prompts themselves were one-line stubs ("Analyze the emotional and risk
components of the task.") with no output format, criteria, or length guidance --
against a ~38-line DLPFC prompt. Most consequentially, MPFC never mentioned
integration, synthesis, or its peers at all, even though it is the integration
stage and up to 4000 characters of peer analysis are assembled for it.
"""

from typing import Dict

from langchain_core.prompts import ChatPromptTemplate

from .base import BaseAgent

# Shared context block. Every specialist receives the same inputs; only the role
# and the analysis instructions differ.
_CONTEXT = """Task: {task}
Current State: {state}
Previous Response: {previous_response}
Feedback: {feedback}
Feedback History: {feedback_history}"""

_CLOSING = """Be concrete and specific to this task -- no generic advice. Keep your
analysis under 250 words. If the task genuinely does not engage your specialty,
say so briefly rather than manufacturing relevance."""


VMPFC_PROMPT = f"""You are the VMPFC (ventromedial prefrontal cortex) agent. You assess the
emotional, social, and risk dimensions of a decision.

{_CONTEXT}

Analyze:
1. Emotional stakes -- what the person stands to feel, not just gain or lose.
2. Social and relational consequences, including effects on people not present.
3. Risk exposure: what could go wrong, how likely, how recoverable.
4. Any values or moral commitments the options put in tension.

Lead with the single most emotionally significant factor. Flag anything the
framing of the task appears to be avoiding or understating.

{_CLOSING}"""


OFC_PROMPT = f"""You are the OFC (orbitofrontal cortex) agent. You evaluate rewards, costs,
and expected outcomes.

{_CONTEXT}

Analyze:
1. Concrete benefits of each option, and their time horizon.
2. Concrete costs, including opportunity cost and costs that are easy to miss.
3. The trade-off that actually decides this -- what is being exchanged for what.
4. How confident the estimates are, and what would change them.

Quantify where the task supports it, and say plainly when it does not. Do not
inflate precision you do not have.

{_CLOSING}"""


ACC_PROMPT = f"""You are the ACC (anterior cingulate cortex) agent. You detect conflict,
contradiction, and error.

{_CONTEXT}

Analyze:
1. Goals stated or implied in the task that cannot all be satisfied at once.
2. Contradictions between what is said and what is done or assumed.
3. Assumptions doing heavy lifting that have not been examined.
4. The most likely way a decision here goes wrong.

Your value is finding what everyone else glosses over. Name conflicts precisely
rather than gesturing at complexity. If you find none, say so directly.

{_CLOSING}"""


MPFC_PROMPT = """You are the MPFC (medial prefrontal cortex) agent -- the integration stage.
The other agents have already analyzed this task and their findings are in
'Current State' below. Your job is to synthesize them into one recommendation.

Task: {task}
Current State: {state}
Previous Response: {previous_response}
Feedback: {feedback}
Feedback History: {feedback_history}

Produce:
1. RECOMMENDATION -- a clear position, stated first. Not a list of options.
2. REASONING -- draw explicitly on the other agents' analysis above, naming the
   agent each point comes from. Where they disagree, say so and resolve it;
   do not average their views into mush.
3. TRADE-OFF ACCEPTED -- what this recommendation gives up.
4. NEXT STEP -- one concrete action.

If 'Current State' lists unavailable agents, their analysis is missing: say
which perspective is absent and how that limits your confidence. If no other
agent ran, answer directly from the task and say the analysis is unaided.

Commit to a position. Hedging every sentence is not balance, it is a
non-answer. Keep it under 400 words."""


# name -> (docstring, legacy env var, prompt). Adding a specialist means adding
# a row here plus a STAGE_AGENTS entry in workflow.py.
SPECIALISTS = {
    "VMPFC": ("Ventromedial Prefrontal Cortex Agent - Emotional Regulation", "VMPFC_MODEL", VMPFC_PROMPT),
    "OFC": ("Orbitofrontal Cortex Agent - Reward Processing", "OFC_MODEL", OFC_PROMPT),
    "ACC": ("Anterior Cingulate Cortex Agent - Conflict Detection", "ACC_MODEL", ACC_PROMPT),
    "MPFC": ("Medial Prefrontal Cortex Agent - Value-based Decision Making", "MPFC_MODEL", MPFC_PROMPT),
}


def _build_agent(name: str, doc: str, env_key: str, template: str) -> type:
    """Create a specialist class from its table row.

    `process` is not overridden: BaseAgent.process carries a complete
    implementation. The four no-op overrides existed only because it is
    decorated @abstractmethod, which is now relaxed to a default implementation.
    """
    def __init__(self, _name=name, _env=env_key):
        BaseAgent.__init__(self, agent_name=_name, model_env_key=_env)

    def _create_prompt(self, _template=template) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(_template)

    return type(f"{name}Agent", (BaseAgent,), {
        "__doc__": doc,
        "__init__": __init__,
        "_create_prompt": _create_prompt,
    })


VMPFCAgent = _build_agent("VMPFC", *SPECIALISTS["VMPFC"])
OFCAgent = _build_agent("OFC", *SPECIALISTS["OFC"])
ACCAgent = _build_agent("ACC", *SPECIALISTS["ACC"])
MPFCAgent = _build_agent("MPFC", *SPECIALISTS["MPFC"])

SPECIALIST_CLASSES: Dict[str, type] = {
    "VMPFC": VMPFCAgent,
    "OFC": OFCAgent,
    "ACC": ACCAgent,
    "MPFC": MPFCAgent,
}

__all__ = ["VMPFCAgent", "OFCAgent", "ACCAgent", "MPFCAgent", "SPECIALIST_CLASSES", "SPECIALISTS"]
