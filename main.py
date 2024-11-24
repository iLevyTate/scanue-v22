import asyncio
import os
import time
from typing import List
from dotenv import load_dotenv
from workflow import create_workflow, process_hitl_feedback

# Load environment variables
load_dotenv()

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

async def main():
    print("==================================================")
    print("Welcome to SCANUE-V: Brain-Inspired Decision Making System")
    print("==================================================\n")
    
    # Initialize workflow
    workflow = create_workflow()
    
    while True:
        try:
            print("\nPlease describe your task or issue:")
            print("> ", end="")
            task = input().strip()
            
            if not task:
                print("\n❌ Task cannot be empty. Please try again.")
                continue
                
            if task.lower() in ['exit', 'quit']:
                print("\n👋 Thank you for using SCANUE-V. Goodbye!")
                break
            
            # Initial state
            state = {
                "task": task,
                "stage": "task_delegation",
                "response": "",
                "subtasks": [],
                "feedback": "",
                "previous_response": "",
                "feedback_history": [],
                "error": False
            }
            
            print("\n🧠 Starting cognitive processing pipeline...\n")
            previous_stage = None
            
            try:
                state = await workflow.ainvoke(state)
                if state.get("error"):
                    print(f"\n❌ {state['response']}")
                    continue
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                continue
                
            # Get feedback
            print("\n📝 Would you like to provide feedback? (y/n)")
            if input().lower().startswith('y'):
                print("Please enter your feedback:")
                feedback = input().strip()
                if feedback:
                    state["feedback"] = feedback
                    state["feedback_history"].append({
                        "response": state["response"],
                        "feedback": feedback
                    })
            
            print("\n✨ Processing complete! Type 'exit' to quit or enter a new task.")
            
        except KeyboardInterrupt:
            print("\n\n👋 SCANUE-V processing interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        exit(1)
        
    asyncio.run(main())
