from abc import ABC, abstractmethod
import logging
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

# Inner LLM call timeout. MUST stay strictly less than the outer per-node timeout
# (NODE_TIMEOUT_SECONDS in workflow.py, 45s) so the inner call fails first and is
# reported cleanly instead of racing the node's outer wait_for.
AGENT_LLM_TIMEOUT_SECONDS = 30.0


def summarize_state(state: Dict[str, Any]) -> str:
    """Build a compact textual summary of the state for prompt injection.

    Injecting the full state dict into prompts bloats tokens and widens the
    prompt-injection surface (feedback history, session logs, etc. all get
    serialized). We pass only the fields an agent actually needs to reason.
    """
    parts = [
        f"task: {state.get('task', '')}",
        f"stage: {state.get('stage', '')}",
    ]
    insights = state.get("previous_agent_insights")
    if insights:
        parts.append(f"previous_agent_insights: {insights}")
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
                    logger.debug("Error initializing %s model for %s: %s", model_type, agent_name, e)

        # Fallback/Legacy Initialization if no primary model found
        if "primary" not in self.models:
            logger.debug("No primary model configured for %s, falling back to legacy/default...", agent_name)
            fallback_config = ConfigLoader.get_model_config(
                agent_name,
                "primary",
                env_var_fallback=model_env_key
            )
            self.models["primary"] = LLMFactory.create_llm(fallback_config)

        # Set primary model as default self.llm for backward compatibility
        self.llm = self.models.get("primary")
        if not self.llm:
            raise ValueError(f"Failed to initialize primary model for agent {agent_name}")

        self.prompt = self._create_prompt()     # Agent-specific prompt template
        self.last_raw_response = None           # Cache for debugging and logging

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

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current workflow state through this agent's cognitive lens.

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
            logger.debug("Error: %s", error_msg)
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }
        except Exception as e:
            # asyncio.CancelledError is a BaseException and intentionally
            # propagates so cooperative cancellation still works.
            error_msg = f"Error processing request: {str(e)}"
            logger.debug("Error: %s", error_msg)
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
                previous_response=state.get("previous_response", "No previous response"),
                feedback=state.get("feedback", "No feedback provided"),
                feedback_history=state.get("feedback_history", [])
            )

            # Invoke the LLM
            # Note: self.llm is now an alias for self.models['primary']
            response = await asyncio.wait_for(
                self.llm.ainvoke(formatted_messages),
                timeout=AGENT_LLM_TIMEOUT_SECONDS
            )

            # Store the complete raw response for logging
            # Handle different model attributes safely
            model_name = getattr(self.llm, "model_name", getattr(self.llm, "model", "unknown"))

            self.last_raw_response = {
                "model": model_name,
                "prompt": self._serialize_messages(formatted_messages),
                "response": response.content,
                "metadata": {
                    "temperature": getattr(self.llm, "temperature", None),
                    # max_tokens might not exist on all model types
                    "max_tokens": getattr(self.llm, "max_tokens", None),
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
