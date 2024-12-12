import pytest
from unittest.mock import AsyncMock, patch
from typing import List, Dict, Any
import asyncio
from langchain.prompts import ChatPromptTemplate

# Import all required agents
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
from agents.dlpfc import DLPFCAgent
from tests.test_agent import TestAgent

class TestSuite:
    """Comprehensive test suite for the agent system"""
    
    @staticmethod
    @pytest.fixture(scope="class")
    def mock_env_vars():
        """Setup mock environment variables"""
        with patch.dict("os.environ", {
            "TEST_MODEL": "test-model",
            "VMPFC_MODEL": "vmpfc-model",
            "OFC_MODEL": "ofc-model",
            "ACC_MODEL": "acc-model",
            "MPFC_MODEL": "mpfc-model",
            "OPENAI_API_KEY": "test-key"
        }):
            yield

    @staticmethod
    @pytest.fixture(scope="class")
    def mock_llm():
        """Setup mock LLM"""
        with patch("langchain_openai.ChatOpenAI.invoke", 
                  new=AsyncMock(return_value=AsyncMock(content="test response"))):
            yield
    
    @staticmethod
    @pytest.mark.asyncio
    async def test_response_handling(mock_env_vars, mock_llm):
        """Test response handling across all agents"""
        agents = [TestAgent(), VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]
        test_input = "Test input with 8 * 8 calculation\n# Section Header"
        
        for agent in agents:
            mock_response = AsyncMock()
            mock_response.content = test_input
            mock_response.additional_kwargs = {"temperature": 0.7}
            
            with patch("langchain_openai.ChatOpenAI.invoke", 
                      new=AsyncMock(return_value=mock_response)):
                result = await agent.process({"task": test_input})
                assert "8 * 8" in result["response"]
                assert "# Section Header" in result["response"]
                assert result.get("metadata", {}).get("temperature") == 0.7
    
    @staticmethod
    @pytest.mark.asyncio
    async def test_concurrent_processing(mock_env_vars, mock_llm):
        """Test concurrent processing with multiple agents"""
        agents = [TestAgent(), VMPFCAgent(), OFCAgent(), ACCAgent(), MPFCAgent()]
        
        async def mock_process(*args, **kwargs):
            return {"response": "test response", "error": False}
        
        with patch("langchain_openai.ChatOpenAI.invoke", 
                  new=AsyncMock(return_value=AsyncMock(content="test"))):
            tasks = [
                agent.process({"task": "test"})
                for agent in agents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                assert isinstance(result, dict)
                assert "response" in result
                assert not result.get("error", False) 