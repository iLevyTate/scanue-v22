from abc import ABC, abstractmethod
import logging
import os
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import copy

from utils.config import ConfigLoader
from agents.factory import LLMFactory

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default inner LLM call timeout, used when a model has no `timeout:` in config.
# MUST stay strictly less than the outer per-node timeout (NODE_TIMEOUT_SECONDS
# in workflow.py) so the inner call fails first and is reported cleanly instead
# of racing the node's outer wait_for.
AGENT_LLM_TIMEOUT_SECONDS = 30.0


def resolve_llm_timeout(config: Dict[str, Any]) -> float:
    """Per-model inner timeout, from `timeout:` in config/agents.yaml.

    This constant used to be the hard cap regardless of configuration: a user
    setting `timeout: 120` for a slow local model still got killed at 30s, with
    no way to change it short of editing source. The client-side timeout the
    factory passes was therefore dead for any value above 30.
    """
    timeout = (config or {}).get("timeout")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return AGENT_LLM_TIMEOUT_SECONDS
    return timeout if timeout > 0 else AGENT_LLM_TIMEOUT_SECONDS


# Bounds on the HITL feedback history injected into every prompt.
#
# This history is persisted across sessions and grows monotonically -- and it was
# injected in full into all 5-6 LLM calls of every run. Measured at ~1,250 chars
# per entry, 25 entries filled ~95% of an 8k context window and 50 entries
# exceeded it, at which point every agent starts failing at once with a generic
# provider error. These are the only things standing between long-term use and
# that wall. Override via SCANUE_FEEDBACK_MAX_ENTRIES / SCANUE_FEEDBACK_CHAR_BUDGET.
FEEDBACK_MAX_ENTRIES = int(os.getenv("SCANUE_FEEDBACK_MAX_ENTRIES", "5"))
FEEDBACK_CHAR_BUDGET = int(os.getenv("SCANUE_FEEDBACK_CHAR_BUDGET", "4000"))
# Per-entry cap on the stored response, which is a whole LLM answer.
FEEDBACK_RESPONSE_CHAR_BUDGET = int(os.getenv("SCANUE_FEEDBACK_RESPONSE_CHARS", "500"))


def _clip(text: str, budget: int) -> str:
    text = str(text)
    return text if len(text) <= budget else text[:budget].rstrip() + " [...truncated]"


def format_feedback_history(history: Any) -> str:
    """Render HITL feedback history as readable text for prompt injection.

    Passing the raw list straight into the template rendered a Python repr
    (``[{'response': ..., 'feedback': ...}]``) into the prompt. Shared by every
    agent so DLPFC and the specialists format history identically.

    Only the most recent FEEDBACK_MAX_ENTRIES are rendered, each response is
    clipped, and the whole block is capped -- see the note on the constants
    above. Recent feedback is also the most relevant, so the window is not
    purely a cost measure.
    """
    if not history:
        return "No previous feedback"

    total = len(history)
    recent = list(history)[-FEEDBACK_MAX_ENTRIES:]

    formatted = []
    for entry in recent:
        if not isinstance(entry, dict):
            formatted.append(_clip(entry, FEEDBACK_RESPONSE_CHAR_BUDGET))
            continue
        formatted.append(
            f"Stage: {entry.get('stage', 'unknown')}\n"
            f"Response: {_clip(entry.get('response', ''), FEEDBACK_RESPONSE_CHAR_BUDGET)}\n"
            f"Feedback: {_clip(entry.get('feedback', ''), FEEDBACK_RESPONSE_CHAR_BUDGET)}\n"
        )

    if total > len(recent):
        formatted.insert(0, f"(showing the {len(recent)} most recent of {total} feedback entries)\n")

    return _clip("\n".join(formatted), FEEDBACK_CHAR_BUDGET)


def extract_usage(response: Any) -> Dict[str, Any]:
    """Pull token counts and the finish reason off a LangChain response.

    Every LLM response carries `usage_metadata` (input/output/total tokens) and
    `response_metadata` (for Ollama: eval_count, prompt_eval_count, done_reason;
    for OpenAI: model_name, finish_reason). All of it was discarded, so the app
    had no notion of spend at all.

    The finish reason matters beyond cost: "length" means the answer was cut off
    mid-generation, and a truncated answer was previously indistinguishable from
    a complete one.
    """
    usage: Dict[str, Any] = {}

    metadata = getattr(response, "usage_metadata", None) or {}
    if isinstance(metadata, dict):
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if metadata.get(key) is not None:
                usage[key] = metadata[key]

    response_metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(response_metadata, dict):
        # OpenAI calls it finish_reason; Ollama calls it done_reason.
        finish = response_metadata.get("finish_reason") or response_metadata.get("done_reason")
        if finish:
            usage["finish_reason"] = finish
            if finish == "length":
                logger.warning(
                    "Response hit the output token limit and was truncated "
                    "mid-generation (finish_reason=length)"
                )
        # Ollama reports token counts here rather than in usage_metadata.
        if "prompt_eval_count" in response_metadata and "input_tokens" not in usage:
            usage["input_tokens"] = response_metadata["prompt_eval_count"]
        if "eval_count" in response_metadata and "output_tokens" not in usage:
            usage["output_tokens"] = response_metadata["eval_count"]

    if "total_tokens" not in usage and {"input_tokens", "output_tokens"} <= usage.keys():
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    return usage


def state_text(state: Dict[str, Any], key: str, placeholder: str) -> str:
    """Read a text field, falling back to `placeholder` when it is blank.

    `state.get(key, placeholder)` was not enough: main.py seeds `feedback` and
    `previous_response` as empty strings, so the key is always PRESENT and the
    default could never fire. Prompts rendered a bare "Feedback:" line instead
    of the intended sentinel.
    """
    value = state.get(key)
    if value is None:
        return placeholder
    text = str(value).strip()
    return text or placeholder


def summarize_state(state: Dict[str, Any]) -> str:
    """Build a compact textual summary of the state for prompt injection.

    Injecting the full state dict into prompts bloats tokens and widens the
    prompt-injection surface (feedback history, session logs, etc. all get
    serialized). We pass only the fields an agent actually needs to reason.
    """
    # `task` is deliberately omitted: every prompt template already has its own
    # `{task}` slot, so including it here rendered the task twice.
    parts = [f"stage: {state.get('stage', '')}"]

    insights = state.get("previous_agent_insights")
    if insights:
        # The summary is emitted with its own heading rather than a snake_cased
        # key, which previously produced the doubled label
        # "previous_agent_insights: \n\nPrevious Agent Insights:".
        parts.append(f"\nAnalysis from other agents:\n{insights.strip()}")

    unavailable = state.get("unavailable_agents")
    if unavailable:
        parts.append(
            "\nUnavailable agents (they failed; their analysis is missing): "
            + ", ".join(unavailable)
        )
    return "\n".join(parts)


class BaseAgent(ABC):
    """Base class for all Prefrontal Cortex agents in the SCANUE-V system.

    This abstract base class provides common functionality for all specialized
    cognitive agents that mirror different regions of the prefrontal cortex.
    Each agent inherits standardized LLM integration, timeout handling, and
    error management while implementing their own specialized processing logic.

    The class supports Human-in-the-Loop (HITL) functionality through
    feedback history integration and provides structured response handling
    for consistent agent interaction patterns.
    """

    def __init__(self, agent_name: str, model_env_key: Optional[str] = None):
        """Initialize agent with models from configuration.

        Args:
            agent_name: Name of the agent (e.g., "DLPFC", "VMPFC")
            model_env_key: Optional legacy env var for backward compatibility
        """
        self.agent_name = agent_name
        self.models = {}
        # Resolved config for the primary model. Kept so the agent can derive its
        # timeout and retry policy instead of hardcoding them.
        self.model_config: Dict[str, Any] = {}

        # Load agent configuration
        agent_config = ConfigLoader.get_agent_config(agent_name)
        model_configs = agent_config.get("models", {})

        # Initialize models
        if model_configs:
            logger.debug("Initializing %s with configured models: %s", agent_name, list(model_configs.keys()))
            for model_type, config in model_configs.items():
                try:
                    self.models[model_type] = LLMFactory.create_llm(config)
                except Exception as e:
                    # A model that cannot be constructed is a real problem (bad
                    # provider name, missing credentials, unreachable base_url).
                    # Logging at debug hid it entirely. If this was the primary
                    # model the fallback path below re-raises; for secondary
                    # models a warning is the only signal the user ever gets.
                    logger.warning(
                        "Could not initialize '%s' model for agent %s: %s",
                        model_type, agent_name, e,
                    )

        if "primary" in model_configs:
            self.model_config = model_configs["primary"] or {}

        # Fallback/Legacy Initialization if no primary model found
        if "primary" not in self.models:
            logger.debug("No primary model configured for %s, falling back to legacy/default...", agent_name)
            fallback_config = ConfigLoader.get_model_config(
                agent_name,
                "primary",
                env_var_fallback=model_env_key
            )
            self.models["primary"] = LLMFactory.create_llm(fallback_config)
            self.model_config = fallback_config

        # Set primary model as default self.llm for backward compatibility
        self.llm = self.models.get("primary")
        if not self.llm:
            raise ValueError(f"Failed to initialize primary model for agent {agent_name}")

        self.llm_timeout = resolve_llm_timeout(self.model_config)
        self.prompt = self._create_prompt()     # Agent-specific prompt template
        self.last_raw_response = None           # Cache for debugging and logging

    def model_descriptor(self) -> Dict[str, Any]:
        """Which model/provider this agent resolved to.

        Recorded at stage START as well as on the response, because it used to
        live only inside raw_llm_response -- which is null on failure, so the log
        could not say which model had failed, the one thing you most want to know.
        """
        return {
            "model": getattr(self.llm, "model_name", getattr(self.llm, "model", "unknown")),
            "provider": (self.model_config or {}).get("provider", "openai"),
        }

    def invoker(self):
        """The primary model wrapped in the configured retry policy.

        Built per call rather than cached so tests that reassign `self.llm` (and
        callers that swap models) get the current one.
        """
        return LLMFactory.wrap_with_retry(self.llm, self.model_config)

    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the specialized prompt template for this agent.

        Each agent must implement this method to define their unique
        cognitive processing approach and output format. The prompt
        should include placeholders for task, feedback, and context.

        Returns:
            ChatPromptTemplate: LangChain prompt template for this agent
        """
        pass

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current workflow state through this agent's cognitive lens.

        This is a complete default implementation. It was previously decorated
        @abstractmethod *while carrying this same body*, which forced every
        specialist to define a `process` that did nothing but
        `return await super().process(state)`. Subclasses override it only when
        they genuinely need different behaviour (DLPFC does).

        This method performs the core agent processing by integrating the current
        task context, feedback history, and previous agent responses into a
        specialized cognitive analysis. Each agent provides unique insights
        based on their prefrontal cortex specialization.

        Args:
            state: Current workflow state with task, feedback, and agent responses

        Returns:
            Dict: Updated state with this agent's response and analysis
        """
        try:
            result = await self._process_with_timeout(state)
            return result
        except asyncio.TimeoutError:
            error_msg = "Request timed out. Please try again."
            logger.warning("%s LLM call timed out after %ss", self.agent_name, AGENT_LLM_TIMEOUT_SECONDS)
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }
        except Exception as e:
            # asyncio.CancelledError is a BaseException and intentionally
            # propagates so cooperative cancellation still works.
            error_msg = f"Error processing request: {str(e)}"
            logger.exception("%s failed to process request", self.agent_name)
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }

    async def _process_with_timeout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process with timeout handling."""
        try:
            # Format prompt messages. Only a compact state summary is injected to
            # limit token bloat and prompt-injection surface.
            formatted_messages = self.prompt.format_messages(
                task=state.get("task", ""),
                state=summarize_state(state),
                previous_response=state_text(state, "previous_response", "No previous response"),
                feedback=state_text(state, "feedback", "No feedback provided"),
                feedback_history=format_feedback_history(state.get("feedback_history", []))
            )

            # Log the prompt size. Ollama silently drops anything past num_ctx,
            # so without this a truncated prompt is completely invisible.
            prompt_chars = sum(len(str(m.content)) for m in formatted_messages)
            logger.debug(
                "%s prompt: %d chars (~%d tokens)",
                self.agent_name, prompt_chars, prompt_chars // 4,
            )

            # Invoke the LLM through the retry wrapper. The timeout covers all
            # attempts, so a flapping provider cannot exceed the node budget.
            response = await asyncio.wait_for(
                self.invoker().ainvoke(formatted_messages),
                timeout=self.llm_timeout
            )

            # Store the complete raw response for logging
            self.last_raw_response = {
                **self.model_descriptor(),
                "prompt": self._serialize_messages(formatted_messages),
                "prompt_chars": prompt_chars,
                "response": response.content,
                "usage": extract_usage(response),
                "metadata": {
                    "temperature": getattr(self.llm, "temperature", None),
                    # max_tokens might not exist on all model types; Ollama's
                    # equivalent is num_predict.
                    "max_tokens": getattr(self.llm, "max_tokens", None)
                    or getattr(self.llm, "num_predict", None),
                    "num_ctx": getattr(self.llm, "num_ctx", None),
                }
            }

            # Format the response
            formatted_result = self._format_response(response.content)

            # Include raw response
            formatted_result["raw_llm_response"] = copy.deepcopy(self.last_raw_response)

            return formatted_result
        except asyncio.TimeoutError:
            logger.debug("API request timed out")
            raise

    def _serialize_messages(self, messages):
        """Serialize messages to a JSON-safe format for logging."""
        try:
            return [
                {
                    "type": message.type,
                    "content": message.content
                }
                for message in messages
            ]
        except Exception:
            # Fallback if serialization fails
            return str(messages)

    def _format_response(self, response: str) -> Dict[str, Any]:
        """Format the response from the LLM."""
        # Format response in the required JSON structure
        structured_response = {
            "role": "assistant",
            "content": response
        }

        return {
            "response": structured_response,
            "error": False
        }
