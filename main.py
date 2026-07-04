import asyncio
import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from workflow import create_workflow
from utils.config import ConfigLoader

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

# File to store persistent feedback history
FEEDBACK_HISTORY_FILE = "feedback_history.json"
LOGS_DIRECTORY = "logs"

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
            with open(FEEDBACK_HISTORY_FILE, 'r') as f:
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
        with open(FEEDBACK_HISTORY_FILE, 'w') as f:
            json.dump(feedback_history, f)
    except Exception as e:
        print(f"Warning: Could not save feedback history: {str(e)}")

def create_session_log(task: str) -> Dict[str, Any]:
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

def save_session_log(session_log: Dict[str, Any]) -> str:
    """Save the session log to a JSON file and return the filename."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(LOGS_DIRECTORY, exist_ok=True)
        
        # Generate timestamp string for filename
        timestamp_str = session_log["timestamp"].replace(':', '-').replace('.', '-')
        session_id = session_log["session_id"][:8]
        
        # Create filename with timestamp and session ID
        filename = f"{LOGS_DIRECTORY}/session_{timestamp_str}_{session_id}.json"
        
        # Save log file
        with open(filename, 'w') as f:
            json.dump(session_log, f, indent=2)
        
        return filename
    except Exception as e:
        print(f"Warning: Could not save session log: {str(e)}")
        return None

async def main(args=None):
    """Main entry point for the application."""
    try:
        # Validate provider credentials based on configured agents/models.
        # This allows fully-local setups (e.g., Ollama) to run without OPENAI_API_KEY.
        config = ConfigLoader.load_config()
        agents_cfg = (config or {}).get("agents", {})

        openai_models_need_key = []
        hf_models_need_token = []

        for agent_name, agent_cfg in agents_cfg.items():
            for model_type, model_cfg in (agent_cfg or {}).get("models", {}).items():
                provider = (model_cfg or {}).get("provider", "openai").lower()
                if provider == "openai":
                    if not (model_cfg or {}).get("api_key") and not os.getenv("OPENAI_API_KEY"):
                        openai_models_need_key.append(f"{agent_name}.{model_type}")
                elif provider == "huggingface":
                    if not (model_cfg or {}).get("api_key") and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
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
                task = args[0]
            else:
                print("Please describe your task or issue:")
                print(">")
                task = input().strip()
                
            if not task:
                print("❌ Task cannot be empty. Please try again.")
                continue
                
            if task.lower() == "exit":
                print("👋 Thank you for using SCANUE-V. Goodbye!")
                break
                
            print("\n🧠 Starting cognitive processing pipeline...\n")
            
            # Create session log
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
                
                # Update session log with final results
                session_log = result.get("session_log", session_log)
                session_log["completed"] = True
                
                if result.get("error"):
                    error_content = result['response']['content'] if isinstance(result['response'], dict) and 'content' in result['response'] else result['response']
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
                response_content = result['response']['content'] if isinstance(result['response'], dict) and 'content' in result['response'] else result['response']
                
                # Store final response in session log
                session_log["final_response"] = result["response"]
                    
                # Always present the response and offer feedback option
                print(f"\n✅ Result: {response_content}")
                    
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
                            # This enables the system to learn from previous interactions
                            new_feedback = {
                                "response": response_content,  # Store response for context
                                "feedback": feedback,          # User's qualitative assessment
                                "stage": result.get("stage", "unknown")  # Processing stage context
                            }
                            feedback_history.append(new_feedback)
                            # PERSISTENCE: Save updated feedback history to file for future sessions
                            save_feedback_history(feedback_history)
                            
                            # SESSION TRACKING: Add feedback to current session log
                            session_log["user_feedback"] = feedback
                            
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

                print(f"\n❌ An error occurred: {str(e)}")
                raise
            
            print("\n✨ Processing complete! Type 'exit' to quit or enter a new task.\n")
            
            # If using command line args, exit after processing
            if not interactive:
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 SCANUE-V processing interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Pass CLI args through so one-shot mode (`python main.py "task"`) works.
    asyncio.run(main(sys.argv[1:]))
