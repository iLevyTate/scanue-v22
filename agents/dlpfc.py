import asyncio
import copy
import inspect
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.factory import LLMFactory

from .base import (
    BaseAgent,
    extract_usage,
    format_feedback_history,
    state_text,
    summarize_state,
)

logger = logging.getLogger(__name__)


class AgentDelegation(BaseModel):
    """Schema-validated delegation decision.

    Asking for the decision as free text and reverse-engineering it is what
    produced the routing bugs: the reply had to survive a reformatter, then a
    ladder of regex and keyword heuristics, any of which could silently pick the
    wrong specialists. A schema removes that entire class of failure -- the
    provider constrains generation and the result is validated before use.

    MPFC is deliberately absent: it always runs as the final integration stage.
    """

    vmpfc: bool = Field(
        description="True if the task involves emotions, social dynamics, risk, or moral judgment"
    )
    ofc: bool = Field(
        description="True if the task involves rewards, costs, financial trade-offs, or benefits"
    )
    acc: bool = Field(
        description="True if the task involves conflicts, competing options, contradictions, or error monitoring"
    )
    reasoning: str = Field(
        default="",
        description="One or two sentences explaining why these specialists were chosen",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Concrete subtasks the specialists should address",
    )

    def to_stages(self) -> list[str]:
        """Map the decision onto router stage names, MPFC always last."""
        stages = []
        if self.vmpfc:
            stages.append("emotional_regulation")
        if self.ofc:
            stages.append("reward_processing")
        if self.acc:
            stages.append("conflict_detection")
        stages.append("value_assessment")
        return stages


class DLPFCAgent(BaseAgent):
    """Dorsolateral Prefrontal Cortex Agent - Central Controller"""

    def __init__(self):
        super().__init__(agent_name="DLPFC", model_env_key="DLPFC_MODEL")
        # Per-call observability. The structured attempt costs tokens even when
        # it fails validation and we fall through to the free-text call, so both
        # the prompt and the attempt count are recorded for the session log.
        self._delegation_messages_cache: list[Any] | None = None
        self._last_usage: dict[str, Any] = {}
        self._structured_attempts = 0

    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the Dorsolateral Prefrontal Cortex (DLPFC) Agent, responsible for:
        1. Analyzing task requirements and complexity
        2. Intelligently selecting only the necessary specialized agents
        3. Delegating subtasks efficiently based on cognitive demands

        Current Task: {task}
        Current State: {state}

        Previous Response (if any): {previous_response}
        User Feedback (if any): {feedback}

        Feedback History:
        {feedback_history}

        IMPORTANT: Only delegate to agents that are actually needed for this specific task.

        Available specialized brain region agents:
        - VMPFC Agent: For tasks involving emotions, social situations, risk assessment, moral decisions
        - OFC Agent: For tasks involving rewards, costs, outcomes, benefits, trade-offs
        - ACC Agent: For tasks with potential conflicts, errors, competing options, monitoring
        - MPFC Agent: Always needed for final integration and value-based decision making

        DELEGATION STRATEGY:
        - Simple factual questions: Only MPFC Agent
        - Emotional decisions: VMPFC Agent + MPFC Agent
        - Financial/reward decisions: OFC Agent + MPFC Agent
        - Complex choices with conflicts: VMPFC Agent + ACC Agent + MPFC Agent
        - Full cognitive processing: VMPFC Agent + OFC Agent + ACC Agent + MPFC Agent

        REQUIRED FORMAT - You must explicitly state which agents to use:
        **AGENT DELEGATION:**
        - VMPFC Agent: [YES/NO] - [brief reason if YES]
        - OFC Agent: [YES/NO] - [brief reason if YES]
        - ACC Agent: [YES/NO] - [brief reason if YES]
        - MPFC Agent: YES - Always needed for final integration

        Then provide your analysis and subtask breakdown.
        """
        return ChatPromptTemplate.from_template(template)

    def _create_delegation_prompt(self) -> ChatPromptTemplate:
        """Prompt for the schema-constrained delegation call.

        Deliberately shorter than the free-text prompt: the output shape is
        enforced by the schema, so none of the "REQUIRED FORMAT" scaffolding is
        needed and the model can spend its attention on the actual decision.
        """
        template = """You are the Dorsolateral Prefrontal Cortex (DLPFC) Agent, the central
        controller of a brain-inspired multi-agent system. Decide which specialized
        agents this task actually requires, and break the task into subtasks.

        Current Task: {task}
        Current State: {state}

        Previous Response (if any): {previous_response}
        User Feedback (if any): {feedback}

        Feedback History:
        {feedback_history}

        Select ONLY the specialists this specific task needs -- do not select all of
        them by default:
        - VMPFC: emotions, social situations, risk assessment, moral decisions
        - OFC: rewards, costs, outcomes, benefits, trade-offs
        - ACC: conflicts, errors, competing options, monitoring

        The MPFC agent always performs the final integration, so it is not your
        choice to make. A simple factual question may need no specialists at all.
        """
        return ChatPromptTemplate.from_template(template)

    def _delegation_messages(self, state: Mapping[str, Any]):
        return self._create_delegation_prompt().format_messages(
            task=state.get("task", ""),
            state=summarize_state(state),
            previous_response=state_text(state, "previous_response", "No previous response"),
            feedback=state_text(state, "feedback", "No feedback provided"),
            feedback_history=self._format_feedback_history(state.get("feedback_history", [])),
        )

    async def _delegate_structured(self, state: Mapping[str, Any]) -> AgentDelegation | None:
        """Ask for a schema-validated delegation decision.

        Returns None -- so the caller falls back to the free-text path -- when
        the provider cannot honor structured output or returns something that
        fails validation. A timeout is NOT swallowed: it propagates so we do not
        spend a second LLM call and blow the outer node timeout.
        """
        # Reset per-call observability state.
        self._delegation_messages_cache = None
        self._last_usage = {}

        try:
            structured_llm = self.llm.with_structured_output(AgentDelegation)
        except Exception as e:
            logger.warning("Structured output unavailable for DLPFC (%s); using text parsing", e)
            return None

        # with_structured_output is sync by contract and returns a Runnable.
        # Anything else means this model cannot be driven this way.
        if inspect.isawaitable(structured_llm):
            # Close it, or it leaks a "coroutine was never awaited" warning.
            close = getattr(structured_llm, "close", None)
            if close:
                close()
            return None
        if not hasattr(structured_llm, "ainvoke"):
            logger.warning("Structured output for DLPFC is not runnable; using text parsing")
            return None

        messages = self._delegation_messages(state)
        self._delegation_messages_cache = messages
        # This call bills tokens whether or not it validates. Count it so the
        # fallback's spend is attributable rather than invisible.
        self._structured_attempts += 1

        try:
            result = await asyncio.wait_for(
                LLMFactory.wrap_with_retry(structured_llm, self.model_config).ainvoke(messages),
                timeout=self.llm_timeout,
            )
        except TimeoutError:
            raise
        except Exception as e:
            logger.warning("Structured delegation call failed (%s); using text parsing", e)
            return None

        if not isinstance(result, AgentDelegation):
            # Notably covers mocked LLMs in tests, which return sentinel objects.
            logger.debug("Structured delegation returned %s; using text parsing", type(result).__name__)
            return None

        return result

    def _result_from_delegation(self, delegation: AgentDelegation) -> dict[str, Any]:
        """Build the standard agent result from a validated delegation."""
        selected = [
            name for name, on in
            (("VMPFC", delegation.vmpfc), ("OFC", delegation.ofc), ("ACC", delegation.acc))
            if on
        ] + ["MPFC"]

        parts = []
        if delegation.subtasks:
            parts.append("📋 Subtasks:")
            parts.extend(f"  • {s}" for s in delegation.subtasks)
        parts.append("\n👥 Agent Assignments:")
        parts.append(f"  • {', '.join(selected)}")
        if delegation.reasoning:
            parts.append(f"\n🧭 Reasoning:\n  {delegation.reasoning}")

        self.last_raw_response = {
            **self.model_descriptor(),
            "prompt": self._serialize_messages(self._delegation_messages_cache or []),
            "response": delegation.model_dump_json(indent=2),
            "usage": self._last_usage,
            "path": "structured_output",
        }

        return {
            "response": {"role": "assistant", "content": "\n".join(parts)},
            "error": False,
            "subtasks": [
                {"task": s, "category": "subtask", "agent": "MPFC Agent"}
                for s in delegation.subtasks
            ],
            "stage": "task_delegation",
            # Consumed directly by the router, so no text parsing happens at all.
            "delegated_agents": delegation.to_stages(),
            "delegation_source": "structured_output",
            "raw_llm_response": copy.deepcopy(self.last_raw_response),
        }

    async def process(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            # Log the compact summary rather than the whole state dict: the raw
            # state carries the full session log and feedback history, which is
            # both unreadable and needlessly sensitive now that logging is wired
            # up to a real handler.
            logger.debug("DLPFC Agent processing:\n%s", summarize_state(state))

            # Preferred path: let the provider constrain generation to the
            # delegation schema. Falls through to free-text parsing when the
            # model or provider cannot do that.
            delegation = await self._delegate_structured(state)
            if delegation is not None:
                logger.debug("Structured delegation: %s", delegation.to_stages())
                return self._result_from_delegation(delegation)

            # Get task breakdown from LLM. The inner timeout mirrors
            # BaseAgent._process_with_timeout -- DLPFC overrides process() and so
            # used to bypass AGENT_LLM_TIMEOUT_SECONDS entirely, leaving the
            # "inner timeout fires before the outer node timeout" invariant
            # (asserted by two tests) vacuous for the one agent that always runs.
            messages = self.prompt.format_messages(
                task=state.get("task", ""),
                state=summarize_state(state),
                previous_response=state_text(state, "previous_response", "No previous response"),
                feedback=state_text(state, "feedback", "No feedback provided"),
                feedback_history=self._format_feedback_history(state.get("feedback_history", []))
            )
            response = await asyncio.wait_for(
                self.invoker().ainvoke(messages),
                timeout=self.llm_timeout,
            )

            logger.debug("DLPFC Agent received response: %s", response)

            # Cache the raw response for logging/debugging (mirrors BaseAgent).
            self.last_raw_response = {
                **self.model_descriptor(),
                "prompt": self._serialize_messages(messages),
                "response": response.content,
                "usage": extract_usage(response),
                "path": "free_text",
                # A failed structured attempt still cost tokens; record it so
                # the fallback's spend is not invisible.
                "structured_attempts": self._structured_attempts,
            }

            # Parse response and update state
            updated_state = self._format_response(response.content)
            subtasks = self._parse_subtasks(response.content)

            logger.debug("Parsed subtasks: %s", subtasks)

            updated_state.update({
                "subtasks": subtasks,
                "stage": "task_delegation",
                "raw_llm_response": copy.deepcopy(self.last_raw_response),
            })

            logger.debug("Updated state: %s", updated_state)
            return updated_state

        except TimeoutError:
            # Same wording BaseAgent uses, so callers see one timeout message.
            error_msg = "Request timed out. Please try again."
            logger.warning("DLPFC LLM call timed out after %ss", self.llm_timeout)
            return {
                "response": {"role": "assistant", "content": error_msg},
                "error": True,
            }
        except Exception as e:
            # asyncio.CancelledError is a BaseException and intentionally
            # propagates so cooperative cancellation still works.
            error_msg = f"Error processing request: {str(e)}"
            logger.exception("DLPFC failed to process request")
            return {
                "response": {"role": "assistant", "content": error_msg},
                "error": True,
            }

    def _parse_subtasks(self, response: str) -> list[dict[str, Any]]:
        """Parse the response to extract subtasks and their assignments."""
        logger.debug("Parsing subtasks from response: %s", response)

        try:
            lines = response.split('\n')
            subtasks = []
            current_category = None
            current_subtask = None
            # The prompt asks for an "**AGENT DELEGATION:**" block of
            # "- VMPFC Agent: YES - reason" lines. Those are routing decisions,
            # not work items -- without this flag they were parsed as subtasks
            # named "YES - reason" / "NO".
            in_delegation_block = False

            # Standard brain region agent types
            brain_region_agents = ["VMPFC", "OFC", "ACC", "MPFC"]

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Skip section headers and formatting
                if line.startswith('**') or line.startswith('#'):
                    in_delegation_block = 'delegation' in line.lower()
                    if 'subtask' in line.lower():
                        current_category = 'subtask'
                    elif 'integration' in line.lower():
                        current_category = 'integration'
                    continue

                if in_delegation_block:
                    continue

                # Look for actual tasks (bullet points or numbered items)
                if line[0].isdigit() or line[0] in ['-', '*', '•']:
                    # Clean up the task text
                    task_text = line.lstrip('0123456789.-*• ').strip()
                    # Remove markdown formatting
                    task_text = task_text.replace('**', '').replace('*', '')

                    # Check for agent assignment in the same line
                    agent = None
                    if " - Assign to " in task_text:
                        task_parts = task_text.split(" - Assign to ")
                        task_text = task_parts[0].strip()
                        agent = task_parts[1].strip()
                    elif ":" in task_text and any(
                        brain_agent in task_text.split(":")[0].upper()
                        for brain_agent in brain_region_agents
                    ):
                        # Handle format like "VMPFC: task description"
                        agent_part = task_text.split(":")[0].strip().upper()
                        task_text = ":".join(task_text.split(":")[1:]).strip()

                        # Extract just the agent name
                        for brain_agent in brain_region_agents:
                            if brain_agent in agent_part:
                                agent = f"{brain_agent} Agent"
                                break

                    if task_text:
                        current_subtask = {
                            "task": task_text,
                            "category": current_category or "general",
                            "agent": agent
                        }
                        subtasks.append(current_subtask)

                # Look for agent assignments in following lines
                elif current_subtask and ('agent:' in line.lower() or 'assign to' in line.lower()):
                    if 'agent:' in line.lower():
                        agent = line.split(':')[1].strip()
                    else:
                        agent = line.split('assign to')[1].strip()

                    # Ensure agent is one of the brain region agents
                    agent_clean = agent.replace('**', '').replace('*', '')
                    for brain_agent in brain_region_agents:
                        if brain_agent in agent_clean.upper():
                            current_subtask["agent"] = f"{brain_agent} Agent"
                            break
                    else:
                        # Default to the most appropriate agent if none specified
                        current_subtask["agent"] = "MPFC Agent"

            # Filter out any empty or invalid tasks
            subtasks = [
                task for task in subtasks
                if task["task"]
                and not task["task"].lower().startswith(('list', 'agent', 'integration'))
            ]

            logger.debug("Parsed %d tasks", len(subtasks))
            for task in subtasks:
                # Assign default agent if none specified
                if not task['agent']:
                    task['agent'] = "MPFC Agent"

            return subtasks

        except Exception as e:
            logger.exception("Error parsing subtasks: %s", e)
            return [{"task": "Error parsing subtasks", "agent": "MPFC Agent", "category": "error"}]

    def _format_response(self, response: str) -> dict[str, Any]:
        """Format the response from the LLM into a structured output."""
        sections: dict[str, list[str]] = {
            "subtasks": [],
            "assignments": [],
            "integration": [],
        }

        try:
            current_section = None
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                is_header = line.startswith('**') or line.startswith('#')

                # Identify sections
                if "subtask" in line.lower():
                    current_section = "subtasks"
                elif "assignment" in line.lower():
                    current_section = "assignments"
                elif "integration" in line.lower():
                    current_section = "integration"
                elif is_header:
                    # A markdown header that names no known section ends the
                    # current one. Without this it fell through to the bullet
                    # branch below (it starts with '*') and was emitted as a
                    # content bullet -- e.g. "**Analysis:**" rendered as
                    # "• Analysis:**" under the previous section's heading.
                    current_section = None
                # Add content to appropriate section
                elif (line[0].isdigit() or line[0] in ['-', '*', '•']) and current_section:
                    sections[current_section].append(line.lstrip('0123456789.-*• ').strip())

            # Format the response in a more readable way
            formatted_response = []
            if sections["subtasks"]:
                formatted_response.append("📋 Subtasks:")
                for task in sections["subtasks"]:
                    formatted_response.append(f"  • {task}")

            if sections["assignments"]:
                formatted_response.append("\n👥 Agent Assignments:")
                for assignment in sections["assignments"]:
                    formatted_response.append(f"  • {assignment}")

            if sections["integration"]:
                formatted_response.append("\n🔄 Integration Plan:")
                for step in sections["integration"]:
                    formatted_response.append(f"  • {step}")

            # Create structured response in JSON format. If none of the expected
            # sections were found, fall back to the raw reply rather than
            # handing back an empty string -- a model that answers in prose
            # (or in an unexpected layout) used to be summarized into nothing.
            response_text = "\n".join(formatted_response) if formatted_response else response.strip()
            structured_response = {
                "role": "assistant",
                "content": response_text
            }

            return {
                "response": structured_response,
                "error": False
            }

        except Exception as e:
            logger.exception("Error formatting DLPFC response")
            structured_error = {
                "role": "assistant",
                "content": str(e)
            }
            return {
                "response": structured_error,
                "error": True
            }

    def _format_feedback_history(self, history: list[dict[str, str]]) -> str:
        """Format feedback history for HITL integration into agent prompts.

        This method processes the persistent feedback history to provide context
        about user preferences and system performance from previous sessions.
        The formatted history informs the agent's decision-making process and
        helps maintain consistency with user expectations.

        Args:
            history: List of feedback entries from previous interactions

        Returns:
            str: Formatted feedback history string for prompt integration
        """
        # Thin wrapper over the shared helper so DLPFC and the specialists render
        # history identically. Kept as a method because tests and callers use it.
        return format_feedback_history(history)
