from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from .base import BaseAgent

class VMPFCAgent(BaseAgent):
    """Ventromedial Prefrontal Cortex Agent - Emotional Regulation"""
    
    def __init__(self):
        super().__init__(model_env_key="VMPFC_MODEL")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the VMPFC Agent, responsible for emotional regulation and risk assessment.
        
        Task: {task}
        Current State: {state}
        Previous Response: {previous_response}
        Feedback: {feedback}
        Feedback History: {feedback_history}
        
        Analyze the emotional and risk components of the task.
        """
        return ChatPromptTemplate.from_template(template)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await super().process(state)
        except Exception as e:
            logger.error(f"Error in VMPFCAgent processing: {str(e)}")
            return {"error": True, "response": str(e)}

class OFCAgent(BaseAgent):
    """Orbitofrontal Cortex Agent - Reward Processing"""
    
    def __init__(self):
        super().__init__(model_env_key="OFC_MODEL")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the OFC Agent, responsible for reward-based decision making.
        
        Task: {task}
        Current State: {state}
        Previous Response: {previous_response}
        Feedback: {feedback}
        Feedback History: {feedback_history}
        
        Evaluate potential rewards and outcomes.
        """
        return ChatPromptTemplate.from_template(template)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state)

class ACCAgent(BaseAgent):
    """Anterior Cingulate Cortex Agent - Conflict Detection"""
    
    def __init__(self):
        super().__init__(model_env_key="ACC_MODEL")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the ACC Agent, responsible for detecting and resolving conflicts.
        
        Task: {task}
        Current State: {state}
        Previous Response: {previous_response}
        Feedback: {feedback}
        Feedback History: {feedback_history}
        
        Identify potential conflicts and propose resolutions.
        """
        return ChatPromptTemplate.from_template(template)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state)

class MPFCAgent(BaseAgent):
    """Medial Prefrontal Cortex Agent - Value-based Decision Making"""
    
    def __init__(self):
        super().__init__(model_env_key="MPFC_MODEL")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the MPFC Agent, responsible for value-based decision making.
        
        Task: {task}
        Current State: {state}
        Previous Response: {previous_response}
        Feedback: {feedback}
        Feedback History: {feedback_history}
        
        Assess alignment with goals and values, and make final recommendations.
        """
        return ChatPromptTemplate.from_template(template)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await super().process(state)
