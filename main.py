import asyncio
import os
import time
import sys
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from workflow import create_workflow, process_hitl_feedback

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

def print_thinking_animation(message: str, duration: int = 2):
    """Display a thinking animation with dots."""
    for _ in range(duration):
        for dots in range(4):
            print(f"\r{message}{'.' * dots}   ", end="", flush=True)
            time.sleep(0.3)
    print("\r" + " " * (len(message) + 4), end="\r")

def print_agent_transition(from_stage: str, to_stage: str):
    """Display a visual transition between agents."""
    print(f"\n{'-' * 20}")
    print(f"🔄 {from_stage.upper()} → {to_stage.upper()}")
    print(f"{'-' * 20}\n")

def format_stage_name(stage: str) -> str:
    """Convert stage name to a readable format with emoji."""
    stage_emojis = {
        "task_delegation": "📋",
        "emotional_regulation": "❤️",
        "reward_processing": "🎯",
        "conflict_detection": "⚡",
        "value_assessment": "💡",
        "complete": "✅"
    }
    formatted_name = stage.replace("_", " ").title()
    return f"{stage_emojis.get(stage, '🔹')} {formatted_name}"

async def main(args=None):
    """Main entry point for the application."""
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
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
        
        while True:
            # Get task from command line args or user input
            if args and len(args) > 0:
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
                "error": False
            }
            
            # Process task
            try:
                result = await workflow.ainvoke(state)
                
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
                    
                    continue
                
                # Extract content from structured response
                response_content = result['response']['content'] if isinstance(result['response'], dict) and 'content' in result['response'] else result['response']
                
                # Store final response in session log
                session_log["final_response"] = result["response"]
                    
                # Always present the response and offer feedback option
                print(f"\n✅ Result: {response_content}")
                    
                # HUMAN-IN-THE-LOOP: Always offer feedback collection for continuous improvement
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
            if args:
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 SCANUE-V processing interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
