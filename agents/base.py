from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import copy
import json

# Load environment variables
load_dotenv()

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
    
    def __init__(self, model_env_key: str):
        """Initialize agent with OpenAI model from environment configuration.
        
        Args:
            model_env_key: Environment variable name containing the OpenAI model identifier
        
        Raises:
            ValueError: If the specified environment variable is not found
        """
        model = os.getenv(model_env_key)
        if not model:
            raise ValueError(f"Model not found in environment variables: {model_env_key}")
        
        print(f"Initializing agent with model: {model}")  # Debug output
        # LLM CONFIGURATION: Set up OpenAI client with robust timeout and retry settings
        self.llm = ChatOpenAI(
            model=model,
            timeout=30.0,           # Total operation timeout
            max_retries=3,          # Automatic retry attempts
            request_timeout=30.0    # Individual HTTP request timeout
        )
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
            print(f"Processing state with prompt: {self.prompt}")  # Debug output
            result = await self._process_with_timeout(state)
            print(f"Received response: {result}")  # Debug output
            return result
        except asyncio.TimeoutError:
            error_msg = "Request timed out. Please try again."
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }
        except asyncio.CancelledError:
            error_msg = "Operation was cancelled."
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": {"role": "assistant", "content": error_msg},
                "raw_llm_response": None,
                "error": True
            }
    
    async def _process_with_timeout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process with timeout handling."""
        try:
            print(f"Sending request to OpenAI API...")  # Debug output
            
            # Format prompt messages
            formatted_messages = self.prompt.format_messages(
                task=state.get("task", ""),
                state=state,
                previous_response=state.get("previous_response", "No previous response"),
                feedback=state.get("feedback", "No feedback provided"),
                feedback_history=state.get("feedback_history", [])
            )
            
            # Invoke the LLM
            response = await asyncio.wait_for(
                self.llm.ainvoke(formatted_messages),
                timeout=30.0
            )
            
            # Store the complete raw response for logging
            self.last_raw_response = {
                "model": self.llm.model_name,
                "prompt": self._serialize_messages(formatted_messages),
                "response": response.content,
                "metadata": {
                    "temperature": getattr(self.llm, "temperature", None),
                    "max_tokens": getattr(self.llm, "max_tokens", None),
                    "top_p": getattr(self.llm, "top_p", None)
                }
            }
            
            print(f"Received API response: {response}")  # Debug output
            
            # Format the response
            formatted_result = self._format_response(response.content)
            
            # Include raw response
            formatted_result["raw_llm_response"] = copy.deepcopy(self.last_raw_response)
            
            return formatted_result
        except asyncio.TimeoutError:
            print("API request timed out")  # Debug output
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
