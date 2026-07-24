import sys
import logging
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from agents.dlpfc import DLPFCAgent
from agents.specialized import VMPFCAgent, OFCAgent, ACCAgent, MPFCAgent
import asyncio
from datetime import datetime
import copy
import re

logger = logging.getLogger(__name__)

# Ensure Unicode output works on Windows consoles where stdout may default to cp1252.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Never fail the application due to console encoding.
    pass

# Outer per-node timeout. MUST stay strictly greater than the agent's inner LLM
# timeout (AGENT_LLM_TIMEOUT_SECONDS in agents/base.py, 30s) so the inner call
# fails and is reported cleanly instead of racing the outer wait_for.
NODE_TIMEOUT_SECONDS = 45.0

# Per-agent character budget for the peer insights handed to MPFC.
#
# MPFC is the integration stage -- synthesizing the specialists is its entire
# job -- but the budget used to be 200 characters, so it saw roughly the first
# 10-15% of each specialist's analysis, cut mid-sentence. The budget exists only
# to bound prompt growth; it should be generous enough that normal responses
# pass through whole.
PEER_INSIGHT_CHAR_BUDGET = 4000


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
    delegated_agents: list   # Which specialist stages DLPFC selected, in order
    agent_responses: dict    # Collected responses keyed by agent name
    agent_errors: dict       # Per-agent failures (workflow continues past these)
    # Stages that have finished executing (success OR failure). This is a plain
    # LastValue channel with NO reducer: every node COPIES the incoming list,
    # appends its own stage, and returns the whole list as a delta. Never add an
    # `Annotated[list, operator.add]` reducer here while nodes echo the list --
    # that would double-count entries and cause the router to skip stages.
    completed_stages: list


# Maps a router stage name to its (agent name, agent class). One helper drives
# all four specialist stages from this table.
STAGE_AGENTS = {
    "emotional_regulation": ("VMPFC", VMPFCAgent),
    "reward_processing": ("OFC", OFCAgent),
    "conflict_detection": ("ACC", ACCAgent),
    "value_assessment": ("MPFC", MPFCAgent),
}


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


def _session_log_delta(state: Dict[str, Any], stage_log: Dict) -> Dict[str, Any]:
    """Build a delta that appends stage_log to session_log["stages"].

    Returns a fresh session_log dict (never mutates the incoming one) so nodes
    return deltas instead of editing shared state in place. Empty dict if there
    is no session log to update.
    """
    session_log = state.get("session_log")
    if not session_log or not stage_log:
        return {}
    return {
        "session_log": {
            **session_log,
            "stages": list(session_log.get("stages", [])) + [stage_log],
        }
    }


# Semantic keywords used only when DLPFC does not emit the structured YES/NO block.
#
# 'value' and 'worth' are deliberately NOT OFC keywords: MPFC *is* the value
# assessment agent, so any DLPFC text describing MPFC's own role ("make a
# value-based decision on how to proceed") used to drag reward_processing into
# runs that never needed it. 'outcome' is omitted for the same reason -- it is
# generic enough to appear in nearly any delegation summary.
SEMANTIC_PATTERNS = {
    'VMPFC': ['emotional', 'feeling', 'social', 'moral', 'risk', 'anxiety', 'fear', 'empathy', 'interpersonal'],
    'OFC': ['reward', 'benefit', 'cost', 'trade', 'tradeoff', 'financial', 'profit', 'loss'],
    'ACC': ['conflict', 'error', 'mistake', 'competing', 'contradiction', 'monitor'],
    'MPFC': []  # Always included
}

# Inflections allowed after a keyword stem, so "rewards"/"conflicts"/"emotionally"
# match while a bare substring scan can no longer fire on an unrelated word
# (e.g. "gloss" matching "loss", or "morale" matching "moral").
_KEYWORD_SUFFIXES = r"(?:s|es|ed|ing|ly|al|ally)?"


def _keyword_present(keyword: str, text_lower: str) -> bool:
    """Word-boundary-anchored keyword match, tolerant of common inflections."""
    return re.search(rf"\b{re.escape(keyword)}{_KEYWORD_SUFFIXES}\b", text_lower) is not None


def parse_agent_assignments(dlpfc_response: str) -> list:
    """Parse the DLPFC agent's response to extract which agents should be called.

    This function intelligently analyzes the DLPFC's task delegation output using
    multiple parsing strategies to determine which specialized agents are needed.
    It prioritizes structured format parsing and falls back to semantic analysis.

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

    response_lower = dlpfc_response.lower()

    logger.debug("DLPFC Response Preview: %s...", response_lower[:200])

    # STRATEGY 1: Parse structured format (YES/NO responses)
    structured_found = False
    for agent_name, stage_name in agent_map.items():
        # Look for "- VMPFC Agent: YES" pattern
        yes_patterns = [
            rf"- {agent_name.lower()} agent:\s*yes",
            rf"{agent_name.lower()} agent:\s*yes",
            rf"- {agent_name.lower()}:\s*yes"
        ]

        for pattern in yes_patterns:
            if re.search(pattern, response_lower):
                if stage_name not in agent_assignments:
                    agent_assignments.append(stage_name)
                    structured_found = True
                    logger.debug("Structured format: %s -> %s", agent_name, stage_name)
                break

    # STRATEGY 2: Semantic keyword analysis (if structured format not found)
    if not structured_found:
        logger.debug("Using semantic analysis fallback...")

        for agent_name, keywords in SEMANTIC_PATTERNS.items():
            if agent_name == 'MPFC':  # Always include MPFC
                continue

            # Check if any semantic keywords are present
            for keyword in keywords:
                if _keyword_present(keyword, response_lower):
                    stage_name = agent_map[agent_name]
                    if stage_name not in agent_assignments:
                        agent_assignments.append(stage_name)
                        logger.debug("Semantic match: '%s' -> %s -> %s", keyword, agent_name, stage_name)
                    break

    # STRATEGY 3: Original pattern matching (final fallback)
    if not agent_assignments and not structured_found:
        logger.debug("Using original pattern matching...")
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
                        logger.debug("Pattern match: '%s' -> %s -> %s", pattern, agent_name, stage_name)
                    break

    # INTELLIGENT FALLBACK: Use minimal viable agents instead of all agents
    if not agent_assignments:
        logger.debug("No specific agents detected, using intelligent minimal fallback...")

        # Analyze task complexity for intelligent fallback
        complexity_indicators = ['complex', 'difficult', 'multiple', 'various', 'several', 'many', 'challenging']
        emotional_indicators = ['feel', 'emotion', 'relationship', 'social', 'personal', 'family', 'friend']
        decision_indicators = ['decide', 'choice', 'option', 'should', 'better', 'prefer', 'recommend']

        is_complex = any(_keyword_present(word, response_lower) for word in complexity_indicators)
        has_emotional = any(_keyword_present(word, response_lower) for word in emotional_indicators)
        is_decision = any(_keyword_present(word, response_lower) for word in decision_indicators)

        if is_complex:
            # Complex tasks get full processing
            agent_assignments = ['emotional_regulation', 'reward_processing', 'conflict_detection', 'value_assessment']
            logger.debug("Complex task detected -> Full cognitive processing")
        elif has_emotional:
            # Emotional tasks get VMPFC + MPFC
            agent_assignments = ['emotional_regulation', 'value_assessment']
            logger.debug("Emotional task detected -> VMPFC + MPFC")
        elif is_decision:
            # Decision tasks get OFC + MPFC
            agent_assignments = ['reward_processing', 'value_assessment']
            logger.debug("Decision task detected -> OFC + MPFC")
        else:
            # Simple tasks get only MPFC
            agent_assignments = ['value_assessment']
            logger.debug("Simple task detected -> MPFC only")

    # Always ensure MPFC is included as the final integration stage
    if 'value_assessment' not in agent_assignments:
        agent_assignments.append('value_assessment')
        logger.debug("Added MPFC for final integration")

    logger.debug("Final agent delegation: %s", agent_assignments)
    return agent_assignments


async def process_task_delegation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process task delegation through DLPFC agent.

    Returns a delta dict (declared AgentState keys only). On any failure the
    stage is still marked complete and a resilient delegation is used so the
    workflow always makes progress toward END.
    """
    print("\n🧠 DLPFC Agent: Breaking down task and delegating...")
    dlpfc = DLPFCAgent()

    # Start logging for this stage
    stage_log = log_stage_start(state, "task_delegation", "DLPFC")

    agent_errors = dict(state.get("agent_errors") or {})
    completed_stages = list(state.get("completed_stages") or [])
    # If DLPFC cannot tell us which agents to run, fall back to a sensible default
    # (VMPFC + ACC + MPFC). OFC is excluded because DLPFC most often omits it.
    resilient_delegation = ["emotional_regulation", "conflict_detection", "value_assessment"]

    try:
        result = await asyncio.wait_for(dlpfc.process(state), timeout=NODE_TIMEOUT_SECONDS)

        # Check for agent-reported errors
        if result.get("error"):
            error_msg = result.get("response", {}).get("content", "Unknown error")
            logger.warning("Task delegation reported error: %s", error_msg)
            agent_errors["DLPFC"] = error_msg
            if stage_log:
                stage_log = log_stage_end(stage_log, result, error_msg)
            delta = {
                "response": result.get("response", {}),
                "agent_errors": agent_errors,
                "delegated_agents": resilient_delegation,
                "agent_responses": {},
                "completed_stages": completed_stages + ["task_delegation"],
            }
            delta.update(_session_log_delta(state, stage_log))
            return delta

        print("✅ Task delegation complete")

        # Parse the RAW LLM reply to determine which agents to call.
        #
        # DLPFCAgent._format_response() rebuilds the reply into a Subtasks /
        # Agent Assignments / Integration Plan digest and keeps a bullet line only
        # if a recognized section header preceded it. The prompt's delegation block
        # is headed "**AGENT DELEGATION:**", which matches none of those keywords,
        # so every "- VMPFC Agent: YES" line is dropped from the digest. Parsing the
        # digest therefore threw away the decision and silently collapsed almost
        # every run to MPFC-only. The raw text is cached by the agent for exactly
        # this reason; the formatted content is only a last-resort fallback.
        raw_reply = (result.get("raw_llm_response") or {}).get("response")
        response_content = result.get("response", {}).get("content", "")
        delegated_agents = parse_agent_assignments(raw_reply or response_content)

        print(f"📋 Delegating to agents: {', '.join(delegated_agents)}")

        if stage_log:
            stage_log = log_stage_end(stage_log, result)

        delta = {
            "response": result.get("response", {}),
            "delegated_agents": delegated_agents,
            "agent_responses": {},
            "agent_errors": agent_errors,
            # `subtasks` is a declared AgentState key and DLPFC parses it out of
            # the reply, but the delta used to omit it -- so the parsed subtasks
            # were computed and then discarded on every run.
            "subtasks": result.get("subtasks", []),
            "completed_stages": completed_stages + ["task_delegation"],
        }
        delta.update(_session_log_delta(state, stage_log))
        return delta

    except Exception as e:
        # asyncio.TimeoutError and all provider errors are subclasses of Exception;
        # asyncio.CancelledError is a BaseException and intentionally propagates.
        error_msg = f"Error in task delegation: {str(e)}"
        print(f"❌ {error_msg}")

        error_response = {"role": "assistant", "content": error_msg}
        agent_errors["DLPFC"] = error_msg

        if stage_log:
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))

        # Mark agent failure but continue workflow with resilient delegation.
        # The router (get_next_stage) picks the next stage from delegated_agents.
        delta = {
            "response": error_response,
            "agent_errors": agent_errors,
            "delegated_agents": resilient_delegation,
            "agent_responses": {},
            "completed_stages": completed_stages + ["task_delegation"],
        }
        delta.update(_session_log_delta(state, stage_log))
        return delta


def _prepare_value_assessment_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich MPFC's input with a summary of the other agents' insights."""
    enhanced_state = copy.deepcopy(state)
    if state.get("agent_responses"):
        agent_summary = "\n\nPrevious Agent Insights:\n"
        for agent_name, response in state["agent_responses"].items():
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            if len(content) > PEER_INSIGHT_CHAR_BUDGET:
                # Only claim truncation when it actually happened; the ellipsis
                # used to be appended unconditionally.
                content = content[:PEER_INSIGHT_CHAR_BUDGET].rstrip() + " [...truncated]"
            agent_summary += f"\n{agent_name} Agent: {content}\n"
        enhanced_state["previous_agent_insights"] = agent_summary
    return enhanced_state


async def _run_specialist_stage(state: Dict[str, Any], stage_name: str, *, prepare_state=None) -> Dict[str, Any]:
    """Run a single specialist stage and return a delta dict.

    Drives all four specialist stages. Copies the incoming accumulator channels
    (agent_responses / agent_errors / completed_stages / session_log), appends
    this stage's contribution, and returns only declared AgentState keys -- never
    mutates the input state in place, never echoes the whole state back.

    The stage is appended to completed_stages on BOTH success and failure, which
    is what makes non-termination structurally impossible: the router will never
    re-dispatch a stage that already ran. `error: True` is set only when the
    final synthesis stage (value_assessment) fails.
    """
    agent_name, agent_class = STAGE_AGENTS[stage_name]
    agent = agent_class()

    stage_log = log_stage_start(state, stage_name, agent_name)

    agent_responses = dict(state.get("agent_responses") or {})
    agent_errors = dict(state.get("agent_errors") or {})
    completed_stages = list(state.get("completed_stages") or [])

    process_input = prepare_state(state) if prepare_state else state

    try:
        result = await asyncio.wait_for(agent.process(process_input), timeout=NODE_TIMEOUT_SECONDS)

        # Per-agent failures are recorded but do not stop the workflow.
        if result.get("error"):
            error_msg = result.get("response", {}).get("content", "Unknown error")
            agent_errors[agent_name] = error_msg

        response = result.get("response", {})
        agent_responses[agent_name] = response

        print(f"✅ {stage_name.replace('_', ' ').title()} complete")

        if stage_log:
            stage_log = log_stage_end(stage_log, result)

        delta = {
            "response": response,
            "agent_responses": agent_responses,
            "agent_errors": agent_errors,
            "completed_stages": completed_stages + [stage_name],
        }
        delta.update(_session_log_delta(state, stage_log))

        # Only a failure of the final synthesis stage marks the whole run errored.
        if stage_name == "value_assessment" and result.get("error"):
            delta["error"] = True

        return delta

    except Exception as e:
        # asyncio.CancelledError (BaseException) intentionally propagates.
        error_msg = f"Error in {stage_name.replace('_', ' ')}: {str(e)}"
        print(f"❌ {error_msg}")

        error_response = {"role": "assistant", "content": error_msg}
        agent_errors[agent_name] = error_msg

        if stage_log:
            stage_log = log_stage_end(stage_log, {"response": error_response}, str(e))

        delta = {
            "response": error_response,
            "agent_errors": agent_errors,
            "completed_stages": completed_stages + [stage_name],
        }
        delta.update(_session_log_delta(state, stage_log))

        if stage_name == "value_assessment":
            delta["error"] = True

        return delta


# The four named node wrappers below stay thin so the graph, tests, and session
# logs can keep referencing them by name. Each just prints its banner and defers
# to the shared specialist runner.

async def process_emotional_regulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process emotional regulation through VMPFC agent."""
    print("\n❤️ VMPFC Agent: Analyzing emotional aspects...")
    return await _run_specialist_stage(state, "emotional_regulation")


async def process_reward_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process reward processing through OFC agent."""
    print("\n🎯 OFC Agent: Evaluating rewards and outcomes...")
    return await _run_specialist_stage(state, "reward_processing")


async def process_conflict_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process conflict detection through ACC agent."""
    print("\n⚡ ACC Agent: Detecting potential conflicts...")
    return await _run_specialist_stage(state, "conflict_detection")


async def process_value_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process value assessment through MPFC agent - integrates all prior responses."""
    print("\n💡 MPFC Agent: Assessing values and integrating insights...")
    return await _run_specialist_stage(
        state, "value_assessment", prepare_state=_prepare_value_assessment_state
    )


def get_next_stage(state: Dict[str, Any]) -> str:
    """Router: pick the first delegated stage that has not completed yet.

    Module-level and pure so it is unit-testable in isolation. Because every node
    appends itself to completed_stages on success AND failure, this loop always
    advances and reaches END -- an infinite loop is structurally impossible.
    """
    delegated = state.get("delegated_agents") or []
    completed = set(state.get("completed_stages") or [])
    for stage in delegated:
        if stage not in completed:
            return stage
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

    # Every stage routes through the same conditional edge function. The mapping
    # below enumerates every possible target so LangGraph always has a valid path.
    all_stages = ["task_delegation", "emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"]

    comprehensive_mappings = {END: END}
    for target_stage in all_stages:
        comprehensive_mappings[target_stage] = target_stage

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

    # Create feedback entry.
    #
    # `stage` and `timestamp` must BOTH be present. main.py used to build its own
    # entry inline with `stage` but no `timestamp`, while this function wrote
    # `timestamp` but no `stage` -- so the two producers emitted different record
    # shapes and DLPFC's history formatter (which reads `stage`) rendered
    # "Stage: unknown" for anything this path wrote.
    feedback_entry = {
        "response": response_content,
        "feedback": feedback,
        "stage": state.get("stage", "unknown"),
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
