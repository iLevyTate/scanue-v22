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
        
        You must only delegate subtasks to these specific brain region agents:
        - VMPFC Agent: Ventromedial Prefrontal Cortex - Handles emotional regulation and risk assessment
        - OFC Agent: Orbitofrontal Cortex - Processes rewards and evaluates outcomes
        - ACC Agent: Anterior Cingulate Cortex - Detects conflicts and monitors for errors
        - MPFC Agent: Medial Prefrontal Cortex - Makes value-based decisions
        
        DO NOT assign tasks to any other agent types like Agent A, Agent B, etc.
        
        Provide:
        1. List of subtasks (incorporating feedback if relevant)
        2. Agent assignments (only to VMPFC, OFC, ACC, or MPFC agents)
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
            updated_state = await self._format_response(response.content)
            subtasks = await self._parse_subtasks(response.content)
            
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
    
    async def _parse_subtasks(self, response: str) -> List[Dict[str, Any]]:
        """Parse the response to extract subtasks and their assignments."""
        print(f"Parsing subtasks from response: {response}")
        
        try:
            lines = response.split('\n')
            subtasks = []
            current_category = None
            current_subtask = None
            
            # Standard brain region agent types
            brain_region_agents = ["VMPFC", "OFC", "ACC", "MPFC"]
            
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
                    
                    # Check for agent assignment in the same line
                    agent = None
                    if " - Assign to " in task_text:
                        task_parts = task_text.split(" - Assign to ")
                        task_text = task_parts[0].strip()
                        agent = task_parts[1].strip()
                    elif ":" in task_text and any(brain_agent in task_text.split(":")[0].upper() for brain_agent in brain_region_agents):
                        # Handle format like "VMPFC: task description"
                        agent_part = task_text.split(":")[0].strip().upper()
                        task_text = ":".join(task_text.split(":")[1:]).strip()
                        
                        # Extract just the agent name
                        for brain_agent in brain_region_agents:
                            if brain_agent in agent_part:
                                agent = f"{brain_agent} Agent"
                                break
                    
                    if task_text:
                        current_subtask = {
                            "task": task_text,
                            "category": current_category or "general",
                            "agent": agent
                        }
                        subtasks.append(current_subtask)
                
                # Look for agent assignments in following lines
                elif current_subtask and ('agent:' in line.lower() or 'assign to' in line.lower()):
                    if 'agent:' in line.lower():
                        agent = line.split(':')[1].strip()
                    else:
                        agent = line.split('assign to')[1].strip()
                    
                    # Ensure agent is one of the brain region agents
                    agent_clean = agent.replace('**', '').replace('*', '')
                    for brain_agent in brain_region_agents:
                        if brain_agent in agent_clean.upper():
                            current_subtask["agent"] = f"{brain_agent} Agent"
                            break
                    else:
                        # Default to the most appropriate agent if none specified
                        current_subtask["agent"] = "MPFC Agent"
            
            # Filter out any empty or invalid tasks
            subtasks = [task for task in subtasks if task["task"] and not task["task"].lower().startswith(('list', 'agent', 'integration'))]
            
            print(f"\nParsed {len(subtasks)} tasks:")
            for task in subtasks:
                print(f"- {task['category'].upper()}: {task['task']}")
                if task['agent']:
                    print(f"  Assigned to: {task['agent']}")
                else:
                    # Assign default agent if none specified
                    task['agent'] = "MPFC Agent"
                    print(f"  Defaulted to: {task['agent']}")
            
            return subtasks
            
        except Exception as e:
            print(f"Error parsing subtasks: {str(e)}")
            return [{"task": "Error parsing subtasks", "agent": "MPFC Agent", "category": "error"}]

    async def _format_response(self, response: str) -> Dict[str, Any]:
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
