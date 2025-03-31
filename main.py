import asyncio
import os
import time
import sys
import json
from typing import List
from dotenv import load_dotenv
from workflow import create_workflow, process_hitl_feedback

# Load environment variables
load_dotenv()

# File to store persistent feedback history
FEEDBACK_HISTORY_FILE = "feedback_history.json"

def load_feedback_history():
    """Load feedback history from a JSON file."""
    try:
        if os.path.exists(FEEDBACK_HISTORY_FILE):
            with open(FEEDBACK_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Warning: Could not load feedback history: {str(e)}")
        return []

def save_feedback_history(feedback_history):
    """Save feedback history to a JSON file."""
    try:
        with open(FEEDBACK_HISTORY_FILE, 'w') as f:
            json.dump(feedback_history, f)
    except Exception as e:
        print(f"Warning: Could not save feedback history: {str(e)}")

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
        
        # Load persistent feedback history from file
        feedback_history = load_feedback_history()
        
        # Print feedback history count if any exists
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
            
            # Initial state - include existing feedback history
            state = {
                "task": task,
                "stage": "task_delegation",
                "response": "",
                "subtasks": [],
                "feedback": "",
                "previous_response": "",
                "feedback_history": feedback_history.copy(),  # Use the persistent feedback history
                "error": False
            }
            
            # Process task
            try:
                result = await workflow.ainvoke(state)
                if result.get("error"):
                    print(f"\n❌ {result['response']}")
                    continue
                    
                # Always present the response and offer feedback option
                print(f"\n✅ Result: {result['response']}")
                    
                # Always offer feedback option
                print("\n📝 Would you like to provide feedback? (y/n)")
                feedback_choice = input().strip().lower()
                
                if feedback_choice == "y":
                    print("Please provide your feedback:")
                    feedback = input().strip()
                    if feedback:
                        print("\n🔄 Processing your feedback...")
                        # Add feedback to history
                        new_feedback = {
                            "response": result["response"],
                            "feedback": feedback,
                            "stage": result.get("stage", "unknown")
                        }
                        feedback_history.append(new_feedback)
                        # Save updated feedback history to file
                        save_feedback_history(feedback_history)
                        print("\n✅ Feedback stored for future queries.")
                        
            except Exception as e:
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
