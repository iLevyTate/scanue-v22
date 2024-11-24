from typing import Dict, Any, List
import asyncio
from langchain.prompts import ChatPromptTemplate
from .base import BaseAgent

class DLPFCAgent(BaseAgent):
    """Dorsolateral Prefrontal Cortex Agent - Central Controller"""
    
    def __init__(self):
        super().__init__(model_env_key="DLPFC_MODEL")

    def _create_prompt(self) -> ChatPromptTemplate:
        template = """You are the Dorsolateral Prefrontal Cortex (DLPFC) Agent, responsible for:
        1. Breaking down complex tasks into subtasks
        2. Delegating subtasks to specialized agents
        3. Integrating responses from all agents
        
        Current Task: {task}
        Current State: {state}
        
        Previous Response (if any): {previous_response}
        User Feedback (if any): {feedback}
        
        Feedback History:
        {feedback_history}
        
        Consider any feedback provided and adjust your approach accordingly.
        
        Provide:
        1. List of subtasks (incorporating feedback if relevant)
        2. Agent assignments
        3. Integration plan
        """
        return ChatPromptTemplate.from_template(template)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(f"DLPFC Agent processing state: {state}")  # Debug output
            
            # Get task breakdown from LLM
            response = await self.llm.ainvoke(
                self.prompt.format_messages(
                    task=state.get("task", ""),
                    state=state,
                    previous_response=state.get("previous_response", "No previous response"),
                    feedback=state.get("feedback", "No feedback provided"),
                    feedback_history=self._format_feedback_history(state.get("feedback_history", []))
                )
            )
            
            print(f"DLPFC Agent received response: {response}")  # Debug output
            
            # Parse response and update state
            updated_state = self._format_response(response.content)
            subtasks = self._parse_subtasks(response.content)
            
            print(f"Parsed subtasks: {subtasks}")  # Debug output
            
            updated_state.update({
                "subtasks": subtasks,
                "stage": "task_delegation"
            })
            
            print(f"Updated state: {updated_state}")  # Debug output
            return updated_state
            
        except asyncio.CancelledError:
            error_msg = "Operation was cancelled"
            print(f"DLPFC Error: {error_msg}")  # Debug output
            return {
                "response": error_msg,
                "error": True,
                "stage": "error"
            }
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(f"DLPFC Error: {error_msg}")  # Debug output
            return {
                "response": error_msg,
                "error": True,
                "stage": "error"
            }
    
    def _parse_subtasks(self, response: str) -> List[Dict[str, Any]]:
        """Parse the response to extract subtasks and their assignments.
        
        Returns a list of dictionaries containing:
        - task: The task description
        - agent: The assigned agent (if specified)
        - category: The category of the task (subtask/integration)
        """
        print(f"Parsing subtasks from response: {response}")
        
        try:
            lines = response.split('\n')
            subtasks = []
            current_category = None
            current_subtask = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip section headers and formatting
                if line.startswith('**') or line.startswith('#'):
                    if 'subtask' in line.lower():
                        current_category = 'subtask'
                    elif 'integration' in line.lower():
                        current_category = 'integration'
                    continue
                
                # Look for actual tasks (bullet points or numbered items)
                if line[0].isdigit() or line[0] in ['-', '*', '•']:
                    # Clean up the task text
                    task_text = line.lstrip('0123456789.-*• ').strip()
                    # Remove markdown formatting
                    task_text = task_text.replace('**', '').replace('*', '')
                    
                    if task_text:
                        current_subtask = {
                            "task": task_text,
                            "category": current_category or "general",
                            "agent": None
                        }
                        subtasks.append(current_subtask)
                
                # Look for agent assignments
                elif current_subtask and 'agent:' in line.lower():
                    agent = line.split(':')[1].strip()
                    current_subtask["agent"] = agent.replace('**', '').replace('*', '')
            
            # Filter out any empty or invalid tasks
            subtasks = [task for task in subtasks if task["task"] and not task["task"].lower().startswith(('list', 'agent', 'integration'))]
            
            print(f"\nParsed {len(subtasks)} tasks:")
            for task in subtasks:
                print(f"- {task['category'].upper()}: {task['task']}")
                if task['agent']:
                    print(f"  Assigned to: {task['agent']}")
            
            return subtasks
            
        except Exception as e:
            print(f"Error parsing subtasks: {str(e)}")
            return [{"task": "Error parsing subtasks", "agent": "error", "category": "error"}]

    def _format_response(self, response: str) -> Dict[str, Any]:
        """Format the response from the LLM into a structured output."""
        sections = {
            "subtasks": [],
            "assignments": [],
            "integration": [],
            "error": False
        }
        
        try:
            current_section = None
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify sections
                if "subtask" in line.lower():
                    current_section = "subtasks"
                elif "assignment" in line.lower():
                    current_section = "assignments"
                elif "integration" in line.lower():
                    current_section = "integration"
                # Add content to appropriate section
                elif line[0].isdigit() or line[0] in ['-', '*', '•']:
                    if current_section:
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
            
            return {
                "response": "\n".join(formatted_response),
                "error": False
            }
            
        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            return {
                "response": str(e),
                "error": True
            }

    def _format_feedback_history(self, history: List[Dict[str, str]]) -> str:
        """Format feedback history for prompt."""
        if not history:
            return "No previous feedback"
        
        formatted = []
        for entry in history:
            formatted.append(
                f"Stage: {entry.get('stage', 'unknown')}\n"
                f"Response: {entry.get('response', '')}\n"
                f"Feedback: {entry.get('feedback', '')}\n"
            )
        return "\n".join(formatted)
