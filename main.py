import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from utils.config import ConfigLoader
from workflow import _response_content, create_workflow, process_hitl_feedback

# Ensure Unicode output works on Windows consoles where stdout may default to cp1252.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Never fail the application due to console encoding.
    pass

# Load environment variables
load_dotenv()

# Persistent state locations.
#
# Anchored to the project root rather than the process CWD. As bare relative
# paths these silently fragmented: launching the CLI from another directory
# started an empty feedback history and a second logs/ tree, with no warning.
# utils/config.py already resolves its config path this way (ConfigLoader
# ._config_path); this matches it. SCANUE_STATE_DIR overrides the root.
PROJECT_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("SCANUE_STATE_DIR", PROJECT_ROOT))
FEEDBACK_HISTORY_FILE = str(STATE_DIR / "feedback_history.json")
LOGS_DIRECTORY = str(STATE_DIR / "logs")

# Keep the most recent N session logs; older ones are pruned after each run.
# 0 disables pruning. Per-file size grows with feedback history, so an unbounded
# logs/ grows superlinearly.
LOG_RETENTION_COUNT = int(os.getenv("SCANUE_LOG_RETENTION", "50"))

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Attach a stderr handler so the library's diagnostics are actually visible.

    Every module logs through `logging`, but nothing ever configured the root
    logger, so the default WARNING threshold silently discarded every message --
    including genuine failures such as a model that could not be constructed.
    Level is taken from SCANUE_LOG_LEVEL (default WARNING); an unrecognized value
    falls back to WARNING rather than crashing at startup.
    """
    level = logging.getLevelName(os.getenv("SCANUE_LOG_LEVEL", "WARNING").upper())
    if not isinstance(level, int):
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

def load_feedback_history():
    """Load persistent feedback history from JSON file for HITL integration.

    This function enables Human-in-the-Loop functionality by loading previously
    collected user feedback that informs agent processing in future sessions.
    The feedback history provides context about user preferences and system performance.

    Returns:
        list: Historical feedback entries with response, feedback, and stage information
    """
    try:
        if os.path.exists(FEEDBACK_HISTORY_FILE):
            with open(FEEDBACK_HISTORY_FILE) as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Warning: Could not load feedback history: {str(e)}")
        return []

def save_feedback_history(feedback_history):
    """Persist feedback history to JSON file for cross-session HITL continuity.

    This function ensures that user feedback is maintained across application
    sessions, enabling the system to learn from previous interactions and
    continuously improve its responses based on accumulated user preferences.

    Args:
        feedback_history: List of feedback entries to persist
    """
    try:
        # Serialize first so a failure cannot truncate an existing history file.
        payload = json.dumps(feedback_history, default=str)
        with open(FEEDBACK_HISTORY_FILE, 'w', encoding='utf-8') as f:
            f.write(payload)
    except Exception as e:
        logger.warning("Could not save feedback history: %s", e)
        print(f"Warning: Could not save feedback history: {str(e)}")

def summarize_run(session_log: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-stage timing and token usage into run totals.

    Only per-stage `duration_ms` existed, and nothing at all recorded tokens --
    so there was no way to see what a run cost or where its time went.
    """
    stages = session_log.get("stages") or []

    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    stage_ms = 0
    truncated = []

    for stage in stages:
        stage_ms += stage.get("duration_ms") or 0
        usage = ((stage.get("raw_llm_response") or {}).get("usage")) or {}
        for key in totals:
            totals[key] += usage.get(key) or 0
        if usage.get("finish_reason") == "length":
            truncated.append(stage.get("stage"))

    summary = {
        "stages_run": len(stages),
        "stage_duration_ms": stage_ms,
        "tokens": totals,
    }
    if truncated:
        # A response cut off mid-generation reads exactly like a complete one.
        summary["truncated_stages"] = truncated
    return summary


def _prune_old_logs() -> None:
    """Keep only the most recent LOG_RETENTION_COUNT session logs.

    Never raises: losing a run because cleanup failed would be worse than
    leaving a stale file behind.
    """
    if LOG_RETENTION_COUNT <= 0:
        return
    try:
        logs = sorted(
            Path(LOGS_DIRECTORY).glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in logs[LOG_RETENTION_COUNT:]:
            stale.unlink()
    except Exception as e:
        logger.debug("Could not prune old session logs: %s", e)


def create_session_log(task: str) -> dict[str, Any]:
    """Create comprehensive session log for workflow execution tracking.

    This function initializes a structured log that captures the complete
    cognitive processing pipeline, including all agent interactions, timing,
    responses, and user feedback for analysis and debugging purposes.

    Args:
        task: The user's input task or query

    Returns:
        Dict: Structured session log with metadata and stage tracking
    """
    timestamp = datetime.now().isoformat()
    return {
        "task": task,
        "timestamp": timestamp,
        "session_id": str(uuid.uuid4()),
        "stages": [],                    # Detailed log of each agent's processing
        "final_response": None,          # Integrated final response from all agents
        "user_feedback": None,           # User's feedback on system performance
        "error": None,                   # Any system errors encountered
        "completed": False               # Whether workflow completed successfully
    }

def save_session_log(session_log: dict[str, Any]) -> str | None:
    """Save the session log to a JSON file and return the filename."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(LOGS_DIRECTORY, exist_ok=True)

        # Generate timestamp string for filename
        timestamp_str = session_log["timestamp"].replace(':', '-').replace('.', '-')
        session_id = session_log["session_id"][:8]

        # Create filename with timestamp and session ID
        filename = f"{LOGS_DIRECTORY}/session_{timestamp_str}_{session_id}.json"

        # Serialize BEFORE opening the file. json.dump() writes incrementally, so
        # anything it cannot encode (a provider object that leaked into
        # raw_llm_response, say) raised partway through and left a truncated,
        # unparseable file behind while this function reported failure.
        # `default=str` keeps an unexpected value from destroying the whole log.
        payload = json.dumps(session_log, indent=2, default=str)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(payload)

        _prune_old_logs()

        return filename
    except Exception as e:
        print(f"Warning: Could not save session log: {str(e)}")
        return None

async def main(args=None):
    """Main entry point for the application."""
    configure_logging()
    try:
        # Validate provider credentials based on configured agents/models.
        # This allows fully-local setups (e.g., Ollama) to run without OPENAI_API_KEY.
        config = ConfigLoader.load_config()
        # Every level here can legitimately be null in hand-edited YAML (an
        # agent with its whole `models:` block commented out, for example), so
        # coerce rather than assume a mapping.
        agents_cfg = (config or {}).get("agents") or {}

        openai_models_need_key = []
        hf_models_need_token = []

        for agent_name, agent_cfg in agents_cfg.items():
            for model_type, model_cfg in ((agent_cfg or {}).get("models") or {}).items():
                provider = (model_cfg or {}).get("provider", "openai").lower()
                has_key = bool((model_cfg or {}).get("api_key"))
                if provider == "openai" and not has_key and not os.getenv("OPENAI_API_KEY"):
                    openai_models_need_key.append(f"{agent_name}.{model_type}")
                elif provider == "huggingface" and not has_key and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                    hf_models_need_token.append(f"{agent_name}.{model_type}")

        if openai_models_need_key:
            print(
                "Error: OPENAI_API_KEY is required for these configured OpenAI models: "
                + ", ".join(openai_models_need_key)
            )
            sys.exit(1)

        if hf_models_need_token:
            print(
                "Error: HUGGINGFACEHUB_API_TOKEN is required for these configured HuggingFace models: "
                + ", ".join(hf_models_need_token)
            )
            sys.exit(1)

        print("=" * 50)
        print("Welcome to SCANUE-V: Brain-Inspired Decision Making System")
        print("=" * 50)
        print("\n")

        # Initialize workflow
        workflow = create_workflow()

        # HITL INITIALIZATION: Load persistent feedback history from file
        # This provides context from previous sessions to inform agent processing
        feedback_history = load_feedback_history()

        # USER AWARENESS: Display feedback history status for transparency
        if feedback_history:
            print(f"📚 Loaded {len(feedback_history)} previous feedback items")

        interactive = not (args and len(args) > 0)

        while True:
            # Get task from command line args or user input
            if not interactive:
                task = args[0].strip()
            else:
                print("Please describe your task or issue:")
                print(">")
                task = input().strip()

            if not task:
                print("❌ Task cannot be empty. Please try again.")
                # One-shot mode re-reads the same argv every iteration, so
                # `continue` here would spin forever on `python main.py ""`.
                if not interactive:
                    sys.exit(1)
                continue

            if task.lower() == "exit":
                print("👋 Thank you for using SCANUE-V. Goodbye!")
                break

            print("\n🧠 Starting cognitive processing pipeline...\n")

            # Create session log. perf_counter, not wall clock: immune to
            # clock adjustment, and it covers agent construction too, which the
            # per-stage durations exclude.
            run_started = time.perf_counter()
            session_log = create_session_log(task)

            # WORKFLOW STATE INITIALIZATION: Include HITL context and session tracking
            state = {
                "task": task,
                "stage": "task_delegation",        # Entry point for workflow
                "response": "",
                "subtasks": [],
                "feedback": "",
                "previous_response": "",
                "feedback_history": feedback_history.copy(),  # HITL: Historical user feedback
                "session_log": session_log,          # Comprehensive execution tracking
                "completed_stages": [],              # Stages that have finished (router state)
                "error": False
            }

            # Process task
            try:
                # recursion_limit guards against pathological routing; the router
                # is designed to always terminate, but this is a safety net.
                result = await workflow.ainvoke(state, config={"recursion_limit": 50})

                # Update session log with final results. A run that reported an
                # error is not "completed" -- recording it as such made failed
                # runs indistinguishable from successful ones in logs/.
                session_log = result.get("session_log", session_log)
                session_log["completed"] = not result.get("error")

                # A run can finish with the final synthesis intact while one or
                # more specialists failed. That is not a clean success, and it
                # used to be recorded and presented as one.
                agent_errors = result.get("agent_errors") or {}
                session_log["agent_errors"] = agent_errors
                session_log["degraded"] = bool(agent_errors)

                session_log["summary"] = summarize_run(session_log)
                session_log["wall_clock_ms"] = int((time.perf_counter() - run_started) * 1000)

                if result.get("error"):
                    error_content = _response_content(result.get("response"))
                    session_log["error"] = error_content
                    print(f"\n❌ {error_content}")

                    # Save session log even on error
                    log_file = save_session_log(session_log)
                    if log_file:
                        print(f"\n📝 Session log saved to: {log_file}")

                    # One-shot runs must not loop back onto the same failing task.
                    if not interactive:
                        break
                    continue

                # Extract content from structured response
                response_content = _response_content(result.get("response"))

                # Store final response in session log
                session_log["final_response"] = result.get("response")

                # Always present the response and offer feedback option
                print(f"\n✅ Result: {response_content}")

                # Say so when the answer was produced without some specialists.
                # Silently returning a partial analysis as if it were complete is
                # the most misleading thing this CLI can do.
                summary = session_log["summary"]
                tokens = summary["tokens"]["total_tokens"]
                print(
                    f"\n⏱️  {session_log['wall_clock_ms'] / 1000:.1f}s"
                    f" · {summary['stages_run']} stages"
                    + (f" · {tokens:,} tokens" if tokens else "")
                )
                if summary.get("truncated_stages"):
                    print(
                        "⚠️  Output token limit reached in: "
                        + ", ".join(summary["truncated_stages"])
                        + " (response cut off mid-generation)"
                    )

                if agent_errors:
                    failed = ", ".join(sorted(agent_errors))
                    print(
                        f"\n⚠️  Partial result: {len(agent_errors)} agent(s) failed "
                        f"({failed}) and were excluded from the final integration."
                    )
                    for name, message in sorted(agent_errors.items()):
                        print(f"     • {name}: {message}")

                # HUMAN-IN-THE-LOOP: Offer feedback collection only in interactive mode.
                # Non-interactive runs (args provided) should never block on stdin.
                if interactive:
                    print("\n📝 Would you like to provide feedback? (y/n)")
                    feedback_choice = input().strip().lower()

                    if feedback_choice == "y":
                        print("Please provide your feedback:")
                        feedback = input().strip()
                        if feedback:
                            print("\n🔄 Processing your feedback...")
                            # PERSISTENT LEARNING: Add feedback to cross-session history
                            # via the single shared implementation, so entries from
                            # the CLI and the workflow always have the same shape.
                            # `stage` in the returned state is still the entry
                            # stage -- no node updates it -- so every entry would
                            # be tagged "task_delegation". The response the user
                            # is reacting to came from the last stage that ran.
                            completed = result.get("completed_stages") or []
                            feedback_state = process_hitl_feedback(
                                {
                                    **result,
                                    "stage": completed[-1] if completed else result.get("stage", "unknown"),
                                    "feedback_history": feedback_history,
                                    "session_log": session_log,
                                },
                                feedback,
                            )
                            feedback_history = feedback_state["feedback_history"]
                            session_log = feedback_state["session_log"]

                            # PERSISTENCE: Save updated feedback history to file for future sessions
                            save_feedback_history(feedback_history)

                            print("\n✅ Feedback stored for future queries.")

                # Save the complete session log
                log_file = save_session_log(session_log)
                if log_file:
                    print(f"\n📝 Session log saved to: {log_file}")

            except GraphRecursionError as e:
                # The workflow failed to converge. Record it and keep the CLI alive
                # instead of crashing the whole session.
                error_msg = f"Workflow did not converge (recursion limit reached): {str(e)}"
                session_log["error"] = error_msg
                session_log["completed"] = False

                log_file = save_session_log(session_log)
                if log_file:
                    print(f"\n📝 Session log saved to: {log_file}")

                print(f"\n❌ {error_msg}")

                if not interactive:
                    break
                continue

            except Exception as e:
                # Record exception in session log
                session_log["error"] = str(e)
                session_log["completed"] = False

                # Save session log on exception
                log_file = save_session_log(session_log)
                if log_file:
                    print(f"\n📝 Session log saved to: {log_file}")

                logger.exception("Workflow raised an exception")
                print(f"\n❌ An error occurred: {str(e)}")

                # A transient provider error should not tear down an interactive
                # session -- the user can just try another task. One-shot runs
                # still exit non-zero so scripts and CI can detect the failure.
                if not interactive:
                    sys.exit(1)
                print("\n↩️  You can try again with a different task.\n")
                continue


            print("\n✨ Processing complete! Type 'exit' to quit or enter a new task.\n")

            # If using command line args, exit after processing
            if not interactive:
                break

    except EOFError:
        # stdin closed with no input: piped input that ran out, a cron job, or
        # `docker run` without -i. KeyboardInterrupt was already handled here;
        # EOF was not, so those runs died with a raw traceback.
        print("\n\n👋 Input stream closed. Goodbye!")
    except KeyboardInterrupt:
        print("\n\n👋 SCANUE-V processing interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        raise

def cli() -> None:
    """Synchronous entry point for the `scanue` console script."""
    # Pass CLI args through so one-shot mode (`scanue "task"`) works.
    asyncio.run(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
