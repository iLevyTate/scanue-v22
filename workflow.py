from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from agents.dlpfc import DLPFCAgent
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
import asyncio
from contextlib import asynccontextmanager
import time
import logging
import json
import signal

class AgentState(TypedDict, total=False):
    task: str
    stage: str
    response: str
    subtasks: list
    feedback: str
    previous_response: str
    feedback_history: list
    error: bool

def validate_state(state: Dict[str, Any]) -> AgentState:
    """Validate state with corruption prevention"""
    try:
        # Deep copy to prevent mutation
        state_copy = json.loads(json.dumps(state))
        
        required_keys = {'task', 'stage', 'response'}
        missing_keys = required_keys - set(state_copy.keys())
        if missing_keys:
            raise ValueError(f"Missing required state keys: {missing_keys}")
        
        # Type validation
        if not isinstance(state_copy.get('task', ''), str):
            raise TypeError("Task must be a string")
        if not isinstance(state_copy.get('stage', ''), str):
            raise TypeError("Stage must be a string")
        if not isinstance(state_copy.get('feedback_history', []), list):
            raise TypeError("Feedback history must be a list")
            
        # Initialize optional fields with defaults
        defaults = {
            'subtasks': [],
            'feedback': '',
            'previous_response': '',
            'feedback_history': [],
            'error': False,
            'error_type': None,
            'retry_count': 0
        }
        
        validated_state = {**defaults, **state_copy}
        
        # Ensure feedback history integrity
        for entry in validated_state['feedback_history']:
            if not isinstance(entry, dict):
                raise TypeError("Feedback history entries must be dictionaries")
            if 'timestamp' not in entry:
                entry['timestamp'] = time.time()
                
        return validated_state
    except json.JSONDecodeError:
        raise ValueError("State contains non-serializable data")

@asynccontextmanager
async def timeout_context(timeout_seconds: float = 30.0):
    """Timeout context with proper cleanup"""
    try:
        async with asyncio.timeout(timeout_seconds):
            yield
    except asyncio.TimeoutError:
        print(f"Operation timed out after {timeout_seconds} seconds")
        raise
    except asyncio.CancelledError:
        print("Operation was cancelled")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

async def process_task_delegation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process task delegation with proper error handling"""
    print("\n DLPFC Agent: Delegating tasks...")
    try:
        dlpfc = DLPFCAgent()
        async with timeout_context():
            result = await dlpfc.process(state)
            return {**state, **result, "stage": "emotional_regulation", "error": state.get("error", False)}
    except asyncio.TimeoutError as e:
        return {
            **state,
            "error": True,
            "response": f"Task delegation timed out: {str(e)}",
            "stage": END
        }
    except Exception as e:
        return {
            **state,
            "error": True,
            "response": f"Error in task delegation: {str(e)}",
            "stage": END
        }

async def process_emotional_regulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process emotional regulation through VMPFC agent."""
    print("\n VMPFC Agent: Analyzing emotional aspects...")
    try:
        validated_state = validate_state(state)
        vmpfc = None
        result = None
        
        try:
            vmpfc = VMPFCAgent()
            async with timeout_context():
                result = await vmpfc.process(validated_state)
                print(f" Emotional analysis complete: {result.get('response', '')}")
                return {
                    **validated_state,
                    **result,
                    "stage": "reward_processing"
                }
        finally:
            if vmpfc and hasattr(vmpfc, 'cleanup'):
                await vmpfc.cleanup()
            result = None
            vmpfc = None
            
    except asyncio.TimeoutError as e:
        logging.warning(f"Emotional regulation timed out: {str(e)}")
        return {
            **validated_state,
            "error": True,
            "error_type": "emotional_regulation_timeout",
            "response": f"Emotional regulation timed out: {str(e)}",
            "stage": END
        }
    except Exception as e:
        logging.error(f"Emotional regulation error: {str(e)}", exc_info=True)
        return {
            **validated_state,
            "error": True,
            "error_type": "emotional_regulation_error",
            "response": f"Error in emotional regulation: {str(e)}",
            "stage": END
        }

async def process_reward_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process reward processing through OFC agent with retry."""
    print("\n OFC Agent: Evaluating rewards and outcomes...")
    
    validated_state = validate_state(state)
    ofc = None
    result = None
    retry_count = 0
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    while retry_count < max_retries:
        try:
            ofc = OFCAgent()
            async with timeout_context():
                result = await ofc.process(validated_state)
                print(f" Reward analysis complete: {result.get('response', '')}")
                return {
                    **validated_state,
                    **result,
                    "stage": "conflict_detection",
                    "retry_count": retry_count
                }
        except asyncio.TimeoutError as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.warning(f"Reward processing failed after {max_retries} retries: {str(e)}")
                return {
                    **validated_state,
                    "error": True,
                    "error_type": "reward_processing_timeout",
                    "response": f"Reward processing timed out after {max_retries} retries: {str(e)}",
                    "stage": END,
                    "retry_count": retry_count
                }
            logging.warning(f"Reward processing timed out, retrying ({retry_count}/{max_retries})...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Reward processing error: {str(e)}", exc_info=True)
            return {
                **validated_state,
                "error": True,
                "error_type": "reward_processing_error", 
                "response": f"Error in reward processing: {str(e)}",
                "stage": END,
                "retry_count": retry_count
            }
        finally:
            if ofc and hasattr(ofc, 'cleanup'):
                await ofc.cleanup()
            result = None
            ofc = None

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

async def process_hitl_feedback(state: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """Process HITL feedback with proper validation"""
    try:
        new_state = state.copy()
        
        if not isinstance(feedback, str):
            raise ValueError("Feedback must be a string")
            
        if "feedback_history" not in new_state:
            new_state["feedback_history"] = []
            
        feedback_entry = {
            "stage": new_state.get("stage", "unknown"),
            "response": new_state.get("response", ""),
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        new_state["feedback_history"].append(feedback_entry)
        new_state["feedback"] = feedback
        new_state["previous_response"] = new_state.get("response", "")
        
        return new_state
    except Exception as e:
        return {
            **state,
            "error": True,
            "response": f"Error processing feedback: {str(e)}"
        }

def create_workflow() -> StateGraph:
    """Create workflow with proper edge conditions"""
    workflow = StateGraph(AgentState)
    
    # Add nodes with proper edge transitions
    workflow.add_node("task_delegation", process_task_delegation)
    workflow.add_edge("task_delegation", "emotional_regulation")
    workflow.add_edge("task_delegation", END)
    
    workflow.add_node("emotional_regulation", process_emotional_regulation)
    workflow.add_edge("emotional_regulation", "reward_processing")
    workflow.add_edge("emotional_regulation", END)
    
    workflow.add_node("reward_processing", process_reward_processing)
    workflow.add_edge("reward_processing", "conflict_detection")
    workflow.add_edge("reward_processing", END)
    
    workflow.add_node("conflict_detection", process_conflict_detection)
    workflow.add_edge("conflict_detection", "value_assessment")
    workflow.add_edge("conflict_detection", END)
    
    workflow.add_node("value_assessment", process_value_assessment)
    workflow.add_edge("value_assessment", END)
    
    workflow.set_entry_point("task_delegation")
    return workflow.compile()

async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]

    logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info(f"Shutdown complete.")
    loop.stop()

async def run_workflow():
    """Run the cognitive workflow."""
    initial_state = {
        "task": "Develop a plan to solve world hunger",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    workflow = create_workflow()
    final_state = await workflow.ainvoke(initial_state)
    
    print(f"\nFinal State: {final_state}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S"
    )
    
    loop = asyncio.get_event_loop()
    
    # May want to catch other signals too
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop)))
    
    try:
        loop.run_until_complete(run_workflow())
    finally:
        loop.close()
        logging.info("Successfully shutdown the workflow.")

if __name__ == "__main__":
    main()
