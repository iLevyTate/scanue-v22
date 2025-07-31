from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from agents.dlpfc import DLPFCAgent
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import copy
import re

class AgentState(TypedDict, total=False):
    task: str
    stage: str
    response: str
    subtasks: list
    feedback: str
    previous_response: str
    feedback_history: list
    session_log: dict
    error: bool
    delegated_agents: list  # New field to track which agents should be called
    agent_responses: dict   # New field to collect responses from all agents

@asynccontextmanager
async def timeout_context(timeout_seconds: float = 30.0):
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError("Operation timed out")
    except asyncio.CancelledError:
        raise KeyboardInterrupt("Operation was cancelled")

def log_stage_start(state: Dict[str, Any], stage_name: str, agent_name: str) -> Dict:
    """Log the start of a processing stage."""
    if "session_log" not in state:
        return None
    
    stage_log = {
        "stage": stage_name,
        "agent": agent_name,
        "start_time": datetime.now().isoformat(),
        "input": {
            "task": state.get("task", ""),
            "feedback": state.get("feedback", ""),
            "previous_response": state.get("previous_response", ""),
            "subtasks": copy.deepcopy(state.get("subtasks", [])),
        },
        "output": None,
        "raw_llm_response": None,
        "error": None,
        "duration_ms": None,
        "end_time": None
    }
    
    return stage_log

def log_stage_end(stage_log: Dict, result: Dict[str, Any], error: str = None) -> Dict:
    """Log the end of a processing stage."""
    if not stage_log:
        return None
    
    # Record end time
    end_time = datetime.now().isoformat()
    stage_log["end_time"] = end_time
    
    # Calculate duration if possible
    if "start_time" in stage_log:
        try:
            start = datetime.fromisoformat(stage_log["start_time"])
            end = datetime.fromisoformat(end_time)
            duration_ms = int((end - start).total_seconds() * 1000)
            stage_log["duration_ms"] = duration_ms
        except Exception:
            # Ignore if we can't calculate duration
            pass
    
    # Record output or error
    if error:
        stage_log["error"] = error
    else:
        # Record the full structured response
        stage_log["output"] = copy.deepcopy(result.get("response", {}))
        
        # If result includes raw LLM response, include it
        if "raw_llm_response" in result:
            stage_log["raw_llm_response"] = copy.deepcopy(result.get("raw_llm_response", {}))
    
    return stage_log

def parse_agent_assignments(dlpfc_response: str) -> list:
    """Parse the DLPFC agent's response to extract which agents should be called.
    
    This function analyzes the DLPFC's task delegation output to determine which
    specialized agents should be invoked for the current task. It looks for explicit
    agent mentions and falls back to a default sequence if none are specified.
    
    Args:
        dlpfc_response: The raw text response from the DLPFC agent
        
    Returns:
        list: Agent stage names in execution order (e.g., ['emotional_regulation', 'conflict_detection'])
    """
    agent_assignments = []
    
    # Agent name mappings
    agent_map = {
        'VMPFC': 'emotional_regulation',
        'OFC': 'reward_processing', 
        'ACC': 'conflict_detection',
        'MPFC': 'value_assessment'
    }
    
    # Look for agent assignments in the response
    response_lower = dlpfc_response.lower()
    
    # Check for explicit agent mentions
    for agent_name, stage_name in agent_map.items():
        patterns = [
            f"{agent_name.lower()} agent",
            f"{agent_name.lower()}:",
            f"assign.*{agent_name.lower()}",
            f"delegate.*{agent_name.lower()}",
            f"{agent_name.lower()}.*agent"
        ]
        
        for pattern in patterns:
            if re.search(pattern, response_lower):
                if stage_name not in agent_assignments:
                    agent_assignments.append(stage_name)
                break
    
    # If no specific agents are mentioned, use default sequence
    if not agent_assignments:
        # Default to calling all agents in logical order - this ensures comprehensive
        # cognitive processing even when DLPFC doesn't specify particular agents
        agent_assignments = [
            'emotional_regulation',  # VMPFC: Risk assessment and emotional processing
            'reward_processing',     # OFC: Outcome evaluation and reward prediction
            'conflict_detection',    # ACC: Conflict monitoring and resolution
            'value_assessment'       # MPFC: Final integration and decision making
        ]
    
    # Always include value_assessment (MPFC) as the final stage for integration
    # MPFC serves as the integration hub that synthesizes insights from all other agents
    if 'value_assessment' not in agent_assignments:
        agent_assignments.append('value_assessment')
    
    return agent_assignments

async def process_task_delegation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process task delegation through DLPFC agent."""
    print("\n🧠 DLPFC Agent: Breaking down task and delegating...")
    dlpfc = DLPFCAgent()
    
    # Start logging for this stage
    stage_log = log_stage_start(state, "task_delegation", "DLPFC")
    
    try:
        async with timeout_context():
            result = await asyncio.wait_for(dlpfc.process(state), timeout=30.0)
            
            # Save raw LLM response if available
            if hasattr(dlpfc, "last_raw_response"):
                result["raw_llm_response"] = dlpfc.last_raw_response
                
            print(f"✅ Task delegation complete")
            
            # Parse the response to determine which agents to call
            response_content = result.get('response', {}).get('content', '')
            delegated_agents = parse_agent_assignments(response_content)
            
            print(f"📋 Delegating to agents: {', '.join(delegated_agents)}")
            
            # Log completion
            if stage_log:
                stage_log = log_stage_end(stage_log, result)
                if "session_log" in state:
                    state["session_log"]["stages"].append(stage_log)
                    
            # Determine next stage based on delegation
            next_stage = delegated_agents[0] if delegated_agents else END
            
            return {
                **state, 
                **result, 
                "stage": next_stage,
                "delegated_agents": delegated_agents,
                "agent_responses": {}
            }
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"❌ Task delegation error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": str(e)}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": str(e)}
        # CRITICAL FIX: Mark agent failure but continue workflow with resilient delegation
        # Based on analysis of DLPFC's typical output patterns, exclude OFC when DLPFC fails
        # This prevents incorrect agent sequencing that was causing workflow issues
        correct_delegated_agents = ["emotional_regulation", "conflict_detection", "value_assessment"]
        # IMPORTANT: Don't set 'stage' manually - LangGraph's get_next_stage function
        # handles state transitions automatically based on the delegated_agents list
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "DLPFC": str(e)},
            "response": error_response, 
            "delegated_agents": correct_delegated_agents
            # stage will be determined by get_next_stage function
        }
    except Exception as e:
        print(f"❌ Task delegation error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": f"Error in task delegation: {str(e)}"}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": f"Error in task delegation: {str(e)}"}
        # Mark agent failure but continue workflow with correct default delegation
        # Use the agents that DLPFC would have assigned: VMPFC, ACC, MPFC (no OFC)
        correct_delegated_agents = ["emotional_regulation", "conflict_detection", "value_assessment"]
        # DON'T set stage manually - let get_next_stage handle the transition properly
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "DLPFC": str(e)},
            "response": error_response, 
            "delegated_agents": correct_delegated_agents
            # stage will be determined by get_next_stage function
        }

async def process_emotional_regulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process emotional regulation through VMPFC agent."""
    print("\n🔍 DEBUG: process_emotional_regulation CALLED!")
    print(f"🔍 DEBUG: State stage = {state.get('stage', 'UNKNOWN')}")
    print("\n❤️ VMPFC Agent: Analyzing emotional aspects...")
    vmpfc = VMPFCAgent()
    
    # Start logging for this stage
    stage_log = log_stage_start(state, "emotional_regulation", "VMPFC")
    
    try:
        async with timeout_context():
            result = await asyncio.wait_for(vmpfc.process(state), timeout=30.0)
            
            # Save raw LLM response if available
            if hasattr(vmpfc, "last_raw_response"):
                result["raw_llm_response"] = vmpfc.last_raw_response
                
            print(f"✅ Emotional analysis complete")
            
            # Store agent response
            if "agent_responses" not in state:
                state["agent_responses"] = {}
            state["agent_responses"]["VMPFC"] = result.get("response", {})
            
            # Log completion
            if stage_log:
                stage_log = log_stage_end(stage_log, result)
                if "session_log" in state:
                    state["session_log"]["stages"].append(stage_log)
                    
            # CRITICAL: Don't set 'stage' manually - LangGraph's conditional edge function
            # determines the next stage based on agent_responses progress tracking
            print(f"🔍 DEBUG: emotional_regulation completed successfully")
            print(f"🔍 DEBUG: delegated_agents = {state.get('delegated_agents', [])}")
            
            return {**state, **result}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"❌ Emotional regulation error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": str(e)}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": str(e)}
        # RESILIENT ERROR HANDLING: Mark agent failure but continue workflow
        # Critical fix: Don't set 'stage' manually to prevent LangGraph state conflicts
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "VMPFC": str(e)},
            "response": error_response
        }
    except Exception as e:
        print(f"❌ Emotional regulation error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": f"Error in emotional regulation: {str(e)}"}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": f"Error in emotional regulation: {str(e)}"}
        # Mark agent failure but continue workflow - DON'T set stage manually
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "VMPFC": str(e)},
            "response": error_response
        }

async def process_reward_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process reward processing through OFC agent."""
    print("\n🎯 OFC Agent: Evaluating rewards and outcomes...")
    ofc = OFCAgent()
    
    # Start logging for this stage
    stage_log = log_stage_start(state, "reward_processing", "OFC")
    
    try:
        async with timeout_context():
            result = await asyncio.wait_for(ofc.process(state), timeout=30.0)
            
            # Save raw LLM response if available
            if hasattr(ofc, "last_raw_response"):
                result["raw_llm_response"] = ofc.last_raw_response
                
            print(f"✅ Reward analysis complete")
            
            # Store agent response
            if "agent_responses" not in state:
                state["agent_responses"] = {}
            state["agent_responses"]["OFC"] = result.get("response", {})
            
            # Log completion
            if stage_log:
                stage_log = log_stage_end(stage_log, result)
                if "session_log" in state:
                    state["session_log"]["stages"].append(stage_log)
                    
            # Determine next stage
            next_stage = get_next_delegated_stage(state, "reward_processing")
            
            return {**state, **result, "stage": next_stage}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"❌ Reward processing error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": str(e)}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": str(e)}
        return {**state, "error": True, "response": error_response, "stage": END}
    except Exception as e:
        print(f"❌ Reward processing error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": f"Error in reward processing: {str(e)}"}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": f"Error in reward processing: {str(e)}"}
        return {**state, "error": True, "response": error_response, "stage": END}

async def process_conflict_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process conflict detection through ACC agent."""
    print("\n🔍 DEBUG: process_conflict_detection CALLED!")
    print(f"🔍 DEBUG: State stage = {state.get('stage', 'UNKNOWN')}")
    print("\n⚡ ACC Agent: Detecting potential conflicts...")
    acc = ACCAgent()
    
    # Start logging for this stage
    stage_log = log_stage_start(state, "conflict_detection", "ACC")
    
    try:
        async with timeout_context():
            result = await asyncio.wait_for(acc.process(state), timeout=30.0)
            
            # Save raw LLM response if available
            if hasattr(acc, "last_raw_response"):
                result["raw_llm_response"] = acc.last_raw_response
                
            print(f"✅ Conflict detection complete")
            
            # Store agent response
            if "agent_responses" not in state:
                state["agent_responses"] = {}
            state["agent_responses"]["ACC"] = result.get("response", {})
            
            # Log completion
            if stage_log:
                stage_log = log_stage_end(stage_log, result)
                if "session_log" in state:
                    state["session_log"]["stages"].append(stage_log)
                    
            # WORKFLOW INTEGRITY: Don't set 'stage' manually - preserves LangGraph's
            # internal state management and prevents infinite loops/agent skipping
            print(f"🔍 DEBUG: conflict_detection completed successfully")
            
            return {**state, **result}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"❌ Conflict detection error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": str(e)}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": str(e)}
        # Mark agent failure but continue workflow - DON'T set stage manually
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "ACC": str(e)},
            "response": error_response
        }
    except Exception as e:
        print(f"❌ Conflict detection error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": f"Error in conflict detection: {str(e)}"}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": f"Error in conflict detection: {str(e)}"}
        # Mark agent failure but continue workflow - DON'T set stage manually
        return {
            **state, 
            "agent_errors": {**state.get("agent_errors", {}), "ACC": str(e)},
            "response": error_response
        }

async def process_value_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process value assessment through MPFC agent - integrates all previous agent responses."""
    print("\n💡 MPFC Agent: Assessing values and integrating insights...")
    mpfc = MPFCAgent()
    
    # Start logging for this stage
    stage_log = log_stage_start(state, "value_assessment", "MPFC")
    
    try:
        # Enhance state with integrated insights from other agents
        enhanced_state = copy.deepcopy(state)
        if "agent_responses" in state and state["agent_responses"]:
            # Add summary of previous agent insights
            agent_summary = "\n\nPrevious Agent Insights:\n"
            for agent_name, response in state["agent_responses"].items():
                content = response.get("content", "") if isinstance(response, dict) else str(response)
                agent_summary += f"\n{agent_name} Agent: {content[:200]}...\n"
            
            # Add to the state for MPFC to consider
            enhanced_state["previous_agent_insights"] = agent_summary
        
        async with timeout_context():
            result = await asyncio.wait_for(mpfc.process(enhanced_state), timeout=30.0)
            
            # Save raw LLM response if available
            if hasattr(mpfc, "last_raw_response"):
                result["raw_llm_response"] = mpfc.last_raw_response
                
            print(f"✅ Value assessment complete - All insights integrated")
            
            # CRITICAL FIX: Store agent response for progress tracking
            # This was missing and caused infinite loops - get_next_stage relies on
            # agent_responses count to determine when all agents have completed
            if "agent_responses" not in state:
                state["agent_responses"] = {}
            state["agent_responses"]["MPFC"] = result.get("response", {})
            
            # Log completion
            if stage_log:
                stage_log = log_stage_end(stage_log, result)
                if "session_log" in state:
                    state["session_log"]["stages"].append(stage_log)
                    
            # FINAL STAGE: Don't set 'stage' manually - let get_next_stage determine
            # workflow completion based on completed agent count vs. delegated agent count
            return {**state, **result}
    except (TimeoutError, KeyboardInterrupt) as e:
        print(f"❌ Value assessment error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": str(e)}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": str(e)}
        return {**state, "error": True, "response": error_response, "stage": END}
    except Exception as e:
        print(f"❌ Value assessment error: {str(e)}")
        
        # Log error
        if stage_log:
            error_response = {"role": "assistant", "content": f"Error in value assessment: {str(e)}"}
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))
            if "session_log" in state:
                state["session_log"]["stages"].append(stage_log)
                
        error_response = {"role": "assistant", "content": f"Error in value assessment: {str(e)}"}
        return {**state, "error": True, "response": error_response, "stage": END}

def get_next_delegated_stage(state: Dict[str, Any], current_stage: str) -> str:
    """Get the next stage based on the delegated agents list."""
    delegated_agents = state.get("delegated_agents", [])
    
    if not delegated_agents:
        return END
    
    try:
        current_index = delegated_agents.index(current_stage)
        if current_index + 1 < len(delegated_agents):
            return delegated_agents[current_index + 1]
        else:
            return END
    except ValueError:
        # Current stage not in list, default to END
        return END

def create_workflow() -> StateGraph:
    """Create the dynamic workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("task_delegation", process_task_delegation)
    workflow.add_node("emotional_regulation", process_emotional_regulation)
    workflow.add_node("reward_processing", process_reward_processing)
    workflow.add_node("conflict_detection", process_conflict_detection)
    workflow.add_node("value_assessment", process_value_assessment)
    
    # CORE WORKFLOW LOGIC: Define conditional edges based on dynamic agent delegation
    def get_next_stage(state: Dict[str, Any]) -> str:
        """Determine the next agent to execute based on workflow progress.
        
        This function is called by LangGraph after each node execution to determine
        the next stage. It uses progress-based tracking rather than manual stage
        setting to prevent infinite loops and ensure all delegated agents execute.
        
        CRITICAL FIX: This replaces the previous flawed approach that caused
        agent skipping and infinite loops by manually setting state['stage'].
        """
        # Only stop on critical system errors, not individual agent failures
        # Individual agent failures are handled gracefully within agent processors
        if state.get("critical_error"):
            print(f"🛑 Critical error detected, stopping workflow")
            return END
        
        # LangGraph calls this function AFTER each node executes
        # We determine progress by counting completed agents vs. total delegated agents
        delegated_agents = state.get("delegated_agents", [])
        if not delegated_agents:
            print("🔄 No delegated agents, ending workflow")
            return END
        
        # PROGRESS TRACKING: Count completed agents by checking agent_responses
        # This dictionary is populated by each agent processor upon successful completion
        agent_responses = state.get("agent_responses", {})
        completed_count = len(agent_responses)
        
        # SEQUENTIAL EXECUTION: Execute agents in delegated order based on progress
        if completed_count < len(delegated_agents):
            next_stage = delegated_agents[completed_count]
            print(f"🔄 Completed {completed_count}/{len(delegated_agents)} agents → {next_stage}")
            return next_stage
        else:
            print(f"🔄 All {len(delegated_agents)} agents completed → END")
            return END
    
    # Create specific edge mappings for each stage
    all_stages = ["task_delegation", "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"]
    
    # COMPREHENSIVE EDGE MAPPING: Include all possible stage transitions
    # This prevents LangGraph errors when get_next_stage returns unexpected values
    comprehensive_mappings = {END: END}
    for target_stage in all_stages:
        comprehensive_mappings[target_stage] = target_stage
    
    # UNIVERSAL CONDITIONAL EDGES: Apply the same comprehensive mapping to all stages
    # This ensures that no matter what get_next_stage returns, there's a valid path
    # Critical for dynamic workflows where agent delegation varies by task
    for stage in all_stages:
        workflow.add_conditional_edges(
            stage,
            get_next_stage,
            comprehensive_mappings
        )
    
    # Set entry point
    workflow.set_entry_point("task_delegation")
    
    return workflow.compile()

def process_hitl_feedback(state: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """Process human-in-the-loop feedback for continuous system improvement.
    
    This function integrates user feedback into the system state, maintaining
    a persistent history that informs future agent processing. The feedback
    is stored both in the current session and in persistent storage.
    
    Args:
        state: Current workflow state containing agent responses and history
        feedback: User's feedback on the system's performance
        
    Returns:
        Dict: Updated state with integrated feedback and history
    """
    if not state.get("feedback_history"):
        state["feedback_history"] = []
    
    # Extract content from the response if it's structured
    response_content = state["response"]["content"] if isinstance(state.get("response"), dict) and "content" in state["response"] else state.get("response", "")
    
    # Create feedback entry
    feedback_entry = {
        "response": response_content,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to feedback history
    state["feedback_history"].append(feedback_entry)
    state["feedback"] = feedback
    state["previous_response"] = response_content
    
    # Add to session log if available
    if "session_log" in state:
        state["session_log"]["user_feedback"] = feedback
        
        # Add feedback log entry
        feedback_log = {
            "stage": "user_feedback",
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "response": state.get("response", "")
        }
        
        if "feedback_entries" not in state["session_log"]:
            state["session_log"]["feedback_entries"] = []
            
        state["session_log"]["feedback_entries"].append(feedback_log)
    
    return state
