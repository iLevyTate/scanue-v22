from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from agents.dlpfc import DLPFCAgent
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
import asyncio
from contextlib import asynccontextmanager

class AgentState(TypedDict, total=False):
    task: str
    stage: str
    response: str
    subtasks: list
    feedback: str
    previous_response: str
    feedback_history: list
    error: bool

@asynccontextmanager
async def timeout_context(timeout_seconds: float = 30.0):
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError("Operation timed out")
    except asyncio.CancelledError:
        raise KeyboardInterrupt("Operation was cancelled")

async def process_task_delegation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process task delegation through DLPFC agent."""
    print("\n DLPFC Agent: Breaking down task and delegating...")
    dlpfc = DLPFCAgent()
    try:
        async with timeout_context():
            result = await asyncio.wait_for(dlpfc.process(state), timeout=30.0)
            print(f" Task delegation complete: {result.get('response', '')}")
        return {**state, **result, "stage": "emotional_regulation"}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f" Task delegation error: {str(e)}")
        return {**state, "error": True, "response": str(e), "stage": END}
    except Exception as e:
        print(f" Task delegation error: {str(e)}")
        return {**state, "error": True, "response": f"Error in task delegation: {str(e)}", "stage": END}

async def process_emotional_regulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process emotional regulation through VMPFC agent."""
    print("\n VMPFC Agent: Analyzing emotional aspects...")
    vmpfc = VMPFCAgent()
    try:
        async with timeout_context():
            result = await asyncio.wait_for(vmpfc.process(state), timeout=30.0)
            print(f" Emotional analysis complete: {result.get('response', '')}")
        return {**state, **result, "stage": "reward_processing"}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f" Emotional regulation error: {str(e)}")
        return {**state, "error": True, "response": str(e), "stage": END}
    except Exception as e:
        print(f" Emotional regulation error: {str(e)}")
        return {**state, "error": True, "response": f"Error in emotional regulation: {str(e)}", "stage": END}

async def process_reward_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process reward processing through OFC agent."""
    print("\n OFC Agent: Evaluating rewards and outcomes...")
    ofc = OFCAgent()
    try:
        async with timeout_context():
            result = await asyncio.wait_for(ofc.process(state), timeout=30.0)
            print(f" Reward analysis complete: {result.get('response', '')}")
        return {**state, **result, "stage": "conflict_detection"}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f" Reward processing error: {str(e)}")
        return {**state, "error": True, "response": str(e), "stage": END}
    except Exception as e:
        print(f" Reward processing error: {str(e)}")
        return {**state, "error": True, "response": f"Error in reward processing: {str(e)}", "stage": END}

async def process_conflict_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process conflict detection through ACC agent."""
    print("\n ACC Agent: Detecting potential conflicts...")
    acc = ACCAgent()
    try:
        async with timeout_context():
            result = await asyncio.wait_for(acc.process(state), timeout=30.0)
            print(f" Conflict detection complete: {result.get('response', '')}")
        return {**state, **result, "stage": "value_assessment"}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f" Conflict detection error: {str(e)}")
        return {**state, "error": True, "response": str(e), "stage": END}
    except Exception as e:
        print(f" Conflict detection error: {str(e)}")
        return {**state, "error": True, "response": f"Error in conflict detection: {str(e)}", "stage": END}

async def process_value_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process value assessment through MPFC agent."""
    print("\n MPFC Agent: Assessing values and goals...")
    mpfc = MPFCAgent()
    try:
        async with timeout_context():
            result = await asyncio.wait_for(mpfc.process(state), timeout=30.0)
            print(f" Value assessment complete: {result.get('response', '')}")
        return {**state, **result, "stage": END}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f" Value assessment error: {str(e)}")
        return {**state, "error": True, "response": str(e), "stage": END}
    except Exception as e:
        print(f" Value assessment error: {str(e)}")
        return {**state, "error": True, "response": f"Error in value assessment: {str(e)}", "stage": END}

async def process_agent_chain(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process state through agent chain with improved concurrency"""
    agents = {
        "emotional_regulation": VMPFCAgent(),
        "reward_processing": OFCAgent(),
        "conflict_detection": ACCAgent(),
        "value_assessment": MPFCAgent()
    }
    
    try:
        # Process DLPFC first
        dlpfc_result = await process_task_delegation(state)
        if dlpfc_result.get("error"):
            return dlpfc_result
        
        # Concurrent processing of specialized agents
        tasks = [
            agent.process({**dlpfc_result, "agent_type": agent_type})
            for agent_type, agent in agents.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and preserve metadata
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                return {
                    "error": True,
                    "error_type": "processing",
                    "response": str(result),
                    "metadata": getattr(result, "metadata", {})
                }
            processed_results.append(result)
        
        return integrate_results(processed_results)
        
    except Exception as e:
        return {
            "error": True,
            "error_type": "processing",
            "response": f"Chain processing error: {str(e)}",
            "metadata": getattr(e, "metadata", {})
        }

def create_workflow() -> StateGraph:
    """Create the workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("task_delegation", process_task_delegation)
    workflow.add_node("emotional_regulation", process_emotional_regulation)
    workflow.add_node("reward_processing", process_reward_processing)
    workflow.add_node("conflict_detection", process_conflict_detection)
    workflow.add_node("value_assessment", process_value_assessment)
    
    # Define conditional edges
    def get_next_stage(state: Dict[str, Any]) -> str:
        if state.get("error"):
            return END
            
        current_stage = state.get("stage", "")
        if current_stage == END:
            return END
            
        stage_map = {
            "task_delegation": "emotional_regulation",
            "emotional_regulation": "reward_processing",
            "reward_processing": "conflict_detection",
            "conflict_detection": "value_assessment",
            "value_assessment": END
        }
        return stage_map.get(current_stage, END)
    
    # Add edges for all stages
    for stage in ["task_delegation", "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"]:
        workflow.add_conditional_edges(
            stage,
            get_next_stage,
            {
                "emotional_regulation": "emotional_regulation",
                "reward_processing": "reward_processing",
                "conflict_detection": "conflict_detection",
                "value_assessment": "value_assessment",
                END: END
            }
        )
    
    # Set entry point
    workflow.set_entry_point("task_delegation")
    
    return workflow.compile()

def process_hitl_feedback(state: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """Process human-in-the-loop feedback."""
    if not state.get("feedback_history"):
        state["feedback_history"] = []
    
    state["feedback_history"].append({
        "response": state.get("response", ""),
        "feedback": feedback
    })
    state["feedback"] = feedback
    state["previous_response"] = state.get("response", "")
    
    return state
