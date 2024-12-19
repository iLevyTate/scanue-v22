from agents.base import BaseAgent
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any

class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    def __init__(self):
        super().__init__(model_env_key="TEST_MODEL")
        
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("Test prompt: {task}")
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state) 