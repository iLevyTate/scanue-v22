from abc import ABC, abstractmethod
import os
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import aiohttp
import openai
import logging
from openai import AuthenticationError, APIError
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BaseAgent(ABC):
    """Base class for all PFC agents in the SCANUE-V system."""
    
    def __init__(self, model_env_key: str):
        """Initialize agent with specific model from environment variable."""
        model = os.getenv(model_env_key)
        if not model:
            raise ValueError(f"Model not found in environment variables: {model_env_key}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        logger.info(f"Initializing agent with model: {model}")
        try:
            self.llm = ChatOpenAI(
                model=model,
                timeout=30.0,  # 30 second timeout
                max_retries=3,  # Retry 3 times
                request_timeout=30.0  # HTTP request timeout
            )
            self.prompt = self._create_prompt()
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise
    
    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for the agent."""
        pass
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        try:
            logger.info(f"Processing state with prompt template: {self.prompt}")
            result = await self._process_with_timeout(state)
            logger.info(f"Received successful response")
            return {
                "response": result.content,
                "error": False,
                "error_type": None,
                "metadata": result.additional_kwargs,
                "sections": self._parse_sections(result.content)
            }
        except Exception as e:
            return self._handle_error(e)
    
    async def _process_with_timeout(self, state: Dict[str, Any]) -> Any:
        """Process with timeout handling."""
        try:
            logger.info("Sending request to OpenAI API...")
            messages = self.prompt.format_messages(
                task=state.get("task", ""),
                state=state,
                previous_response=state.get("previous_response", "No previous response"),
                feedback=state.get("feedback", "No feedback provided"),
                feedback_history=state.get("feedback_history", [])
            )
            logger.debug(f"Formatted messages: {messages}")
            
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=30.0
            )
            logger.info("Received API response")
            logger.debug(f"Response content: {response.content}")
            return response
        except (aiohttp.ClientError, openai.APIError) as e:
            logger.error(f"API connection error: {str(e)}")
            raise ConnectionError(f"Failed to connect to OpenAI API: {str(e)}")
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}", exc_info=True)
            raise
    
    def _format_response(self, response: str) -> Dict[str, Any]:
        """Format the response from the LLM."""
        return {
            "response": response,
            "error": False,
            "error_type": None
        }
    
    def _parse_sections(self, content: str) -> Dict[str, List[str]]:
        """Parse content into sections with improved robustness"""
        sections = {
            "subtasks": [],
            "assignments": [],
            "integration": []
        }
        
        current_section = None
        section_buffer = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Enhanced section header detection
            lower_line = line.lower()
            if ':' in lower_line:
                # Flush current section buffer
                if current_section and section_buffer:
                    sections[current_section].extend(section_buffer)
                    section_buffer = []
                
                # Detect new section
                for section_type in sections.keys():
                    if section_type in lower_line:
                        current_section = section_type
                        break
                continue
            
            if current_section:
                # Preserve mathematical expressions and special characters
                cleaned_line = re.sub(r'^[\s]*[-\d.•]+[\s]+', '', line)
                if cleaned_line:
                    section_buffer.append(cleaned_line.strip())
        
        # Flush final section buffer
        if current_section and section_buffer:
            sections[current_section].extend(section_buffer)
        
        return sections
    
    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors while preserving metadata"""
        error_response = {
            "error": True,
            "response": str(error),
            "metadata": getattr(error, 'additional_kwargs', {}),
            "sections": {},
        }
        
        if isinstance(error, AuthenticationError):
            error_response["error_type"] = "connection"
        elif isinstance(error, APIError):
            error_response["error_type"] = "connection"
        elif isinstance(error, ConnectionError):
            error_response["error_type"] = "connection"
        else:
            error_response["error_type"] = "processing"
            
        return error_response
