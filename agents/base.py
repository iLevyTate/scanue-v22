from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

class BaseAgent(ABC):
    """Base class for all PFC agents in the SCANUE-V system."""
    
    def __init__(self, model_env_key: str):
        """Initialize agent with specific model from environment variable."""
        model = os.getenv(model_env_key)
        if not model:
            raise ValueError(f"Model not found in environment variables: {model_env_key}")
        
        print(f"Initializing agent with model: {model}")  # Debug output
        self.llm = ChatOpenAI(
            model=model,
            timeout=30.0,  # 30 second timeout
            max_retries=3,  # Retry 3 times
            request_timeout=30.0  # HTTP request timeout
        )
        self.prompt = self._create_prompt()
    
    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for the agent."""
        pass
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        try:
            print(f"Processing state with prompt: {self.prompt}")  # Debug output
            result = await self._process_with_timeout(state)
            print(f"Received response: {result}")  # Debug output
            return result
        except asyncio.TimeoutError:
            error_msg = "Request timed out. Please try again."
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": error_msg,
                "error": True
            }
        except asyncio.CancelledError:
            error_msg = "Operation was cancelled."
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": error_msg,
                "error": True
            }
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(f"Error: {error_msg}")  # Debug output
            return {
                "response": error_msg,
                "error": True
            }
    
    async def _process_with_timeout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process with timeout handling."""
        try:
            print(f"Sending request to OpenAI API...")  # Debug output
            response = await asyncio.wait_for(
                self.llm.ainvoke(
                    self.prompt.format_messages(
                        task=state.get("task", ""),
                        state=state,
                        previous_response=state.get("previous_response", "No previous response"),
                        feedback=state.get("feedback", "No feedback provided"),
                        feedback_history=state.get("feedback_history", [])
                    )
                ),
                timeout=30.0
            )
            print(f"Received API response: {response}")  # Debug output
            return self._format_response(response.content)
        except asyncio.TimeoutError:
            print("API request timed out")  # Debug output
            raise
    
    def _format_response(self, response: str) -> Dict[str, Any]:
        """Format the response from the LLM."""
        return {
            "response": response,
            "error": False
        }
