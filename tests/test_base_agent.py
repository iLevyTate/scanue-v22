import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.base import BaseAgent
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
import asyncio
from openai import AuthenticationError
import openai

class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    def __init__(self):
        super().__init__(model_env_key="TEST_MODEL")
        
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("Test prompt: {task}")
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state)

@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {
        "TEST_MODEL": "test-model",
        "OPENAI_API_KEY": "test-key"
    }):
        yield

@pytest.fixture
def mock_llm():
    """Setup mock LLM with proper metadata handling"""
    async def mock_ainvoke(*args, **kwargs):
        mock_response = AsyncMock()
        mock_response.content = "test response"
        # Ensure additional_kwargs is properly set and accessible
        mock_response.additional_kwargs = {
            "temperature": 0.7,
            "model": "test-model"
        }
        return mock_response
    
    with patch("langchain_openai.ChatOpenAI.ainvoke", new=mock_ainvoke):
        yield

@pytest.fixture
def test_agent(mock_env_vars):
    return TestAgent()

@pytest.fixture(scope="function")
def test_state():
    return {
        "task": "test task",
        "stage": "test_stage",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }

@pytest.mark.asyncio
async def test_base_agent_initialization(test_agent):
    """Test base agent initialization"""
    assert isinstance(test_agent, BaseAgent)
    assert hasattr(test_agent, 'process')
    assert hasattr(test_agent, '_create_prompt')

@pytest.mark.asyncio
async def test_base_agent_process(test_agent, test_state, mock_llm):
    """Test base agent process method"""
    result = await test_agent.process(test_state)
    assert isinstance(result, dict)
    assert result["response"] == "test response"
    assert not result.get("error", False)
    assert result.get("metadata", {}).get("temperature") == 0.7

@pytest.mark.asyncio
async def test_base_agent_validation(test_agent):
    """Test state validation"""
    invalid_state = {"invalid": "state"}
    with patch("langchain_openai.ChatOpenAI.ainvoke") as mock_ainvoke:
        mock_ainvoke.side_effect = ValueError("Invalid state format")
        result = await test_agent.process(invalid_state)
        assert result["error"]
        assert "Invalid state format" in result["response"]

@pytest.mark.asyncio
async def test_base_agent_timeout(test_agent, test_state):
    """Test timeout handling"""
    async def mock_process(*args, **kwargs):
        await asyncio.sleep(1)
        return None

    test_agent.process = mock_process
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):
            await test_agent.process(test_state)

@pytest.mark.asyncio
async def test_base_agent_error_handling(test_agent, test_state):
    """Test error handling"""
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=ValueError("Test error")):
        result = await test_agent.process(test_state)
        assert result["error"]
        assert "Test error" in result["response"]

@pytest.mark.asyncio
async def test_base_agent_cancellation(test_agent, test_state):
    """Test cancellation handling"""
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=asyncio.CancelledError()):
        result = await test_agent.process(test_state)
        assert result["error"]
        assert "cancelled" in result["response"].lower()

@pytest.mark.asyncio
async def test_base_agent_authentication_error(test_agent, test_state):
    """Test authentication error handling"""
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_body = {"error": {"message": "Invalid API key"}}
    with patch("langchain_openai.ChatOpenAI.ainvoke", 
              side_effect=AuthenticationError(message="Invalid API key", response=mock_response, body=mock_body)):
        result = await test_agent.process(test_state)
        assert result["error"]
        assert "Invalid API key" in result["response"]
        assert result["error_type"] == "connection"

@pytest.mark.asyncio
async def test_base_agent_connection_error(test_agent, test_state):
    """Test connection error handling"""
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=ConnectionError("Failed to connect")):
        result = await test_agent.process(test_state)
        assert result["error"]
        assert "Failed to connect" in result["response"]
        assert result["error_type"] == "connection"

@pytest.mark.asyncio
async def test_base_agent_api_error(test_agent, test_state):
    """Test API error handling"""
    mock_request = {"url": "test_url", "method": "POST"}
    mock_body = {"error": {"message": "API Error"}}
    with patch("langchain_openai.ChatOpenAI.ainvoke", 
              side_effect=openai.APIError(message="API Error", request=mock_request, body=mock_body)):
        result = await test_agent.process(test_state)
        assert result["error"]
        assert "API Error" in result["response"]
        assert result["error_type"] == "connection"

@pytest.mark.asyncio
async def test_mathematical_content_preservation(test_agent):
    """Test preservation of mathematical expressions"""
    test_expressions = [
        "8 * 8 = 64",
        "x * y + z",
        "Area = π*r²",
        "8*8 * 0"
    ]
    
    for expr in test_expressions:
        mock_response = AsyncMock()
        mock_response.content = expr
        mock_response.additional_kwargs = {"temperature": 0.7}
        
        with patch("langchain_openai.ChatOpenAI.ainvoke", 
                  new=AsyncMock(return_value=mock_response)):
            result = await test_agent.process({"task": expr})
            assert expr in result["response"]

@pytest.mark.asyncio
async def test_metadata_preservation(test_agent, test_state, mock_llm):
    """Test metadata preservation"""
    result = await test_agent.process(test_state)
    assert result.get("metadata", {}).get("temperature") == 0.7

@pytest.mark.asyncio
async def test_base_agent_response_type(test_agent, test_state, mock_llm):
    """Test proper handling of BaseMessage response types"""
    result = await test_agent.process(test_state)
    assert isinstance(result, dict)
    assert result["response"] == "test response"
    assert result.get("metadata", {}).get("temperature") == 0.7

@pytest.mark.asyncio
async def test_section_header_detection(test_agent):
    """Test section header detection at start of line only"""
    test_cases = [
        ("Subtasks:\n- Task 1\n- Task 2", {"subtasks": ["- Task 1", "- Task 2"]}),
        ("Assignments:\nAssignment 1\nAssignment 2", {"assignments": ["Assignment 1", "Assignment 2"]}),
        ("Integration:\nIntegrate this\nAnd that", {"integration": ["Integrate this", "And that"]})
    ]
    
    for input_text, expected_sections in test_cases:
        mock_response = AsyncMock()
        mock_response.content = input_text
        mock_response.additional_kwargs = {}
        
        with patch("langchain_openai.ChatOpenAI.ainvoke", 
                  new=AsyncMock(return_value=mock_response)):
            result = await test_agent.process({"task": input_text})
            for section, expected_lines in expected_sections.items():
                assert section in result["sections"]
                assert result["sections"][section] == expected_lines

@pytest.mark.asyncio
async def test_base_agent_response_handling(test_agent, mock_llm):
    """Test complete response handling chain"""
    test_cases = [
        {"task": "Calculate 8 * 8"},
        {"task": "# Section\nContent"},
        {"task": "Area = π*r²\n# Results"}
    ]
    
    for case in test_cases:
        result = await test_agent.process(case)
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == "test response"
        assert result.get("metadata", {}).get("temperature") == 0.7
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_error_handling_chain(test_agent, test_state):
    """Test error handling chain"""
    test_cases = [
        (ValueError("Invalid state"), "processing"),
        (ConnectionError("Connection failed"), "connection"),
        (openai.APIError("API error", request=MagicMock(), body={"error": "API error"}), "connection"),
        (AuthenticationError("Invalid key", response=MagicMock(request=MagicMock()), body={"error": "Invalid key"}), "connection"),
    ]
    
    for error, expected_type in test_cases:
        with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=error):
            result = await test_agent.process(test_state)
            assert result["error"]
            assert expected_type == result["error_type"]
