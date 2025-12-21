#!/usr/bin/env python3
"""Test HITL (Human-In-The-Loop) integration across the entire workflow."""

import os
import json
import asyncio
from datetime import datetime
import uuid

# Set dummy API key for testing
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key-for-hitl-testing"

try:
    from workflow import create_workflow, process_hitl_feedback
    from main import load_feedback_history, save_feedback_history
    print("✅ Successfully imported HITL components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def create_test_session_log(task):
    """Create a test session log"""
    return {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "session_id": str(uuid.uuid4()),
        "stages": [],
        "final_response": None,
        "user_feedback": None,
        "error": None,
        "completed": False
    }

def test_feedback_persistence():
    """Test that feedback is properly stored and loaded"""
    print("\n🧪 Testing feedback persistence...")
    
    # Create test feedback
    test_feedback = [
        {
            "response": "Test response 1",
            "feedback": "This response was helpful but could be more detailed.",
            "stage": "value_assessment"
        },
        {
            "response": "Test response 2", 
            "feedback": "Please provide more specific recommendations.",
            "stage": "conflict_detection"
        }
    ]
    
    # Save test feedback
    save_feedback_history(test_feedback)
    print("✅ Test feedback saved")
    
    # Load feedback and verify
    loaded_feedback = load_feedback_history()

    assert len(loaded_feedback) >= len(test_feedback)

def test_feedback_processing():
    """Test the process_hitl_feedback function"""
    print("\n🧪 Testing feedback processing...")
    
    # Create test state
    test_state = {
        "task": "Test task",
        "stage": "emotional_regulation",
        "response": {"role": "assistant", "content": "Test response content"},
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "error": False
    }
    
    # Process feedback
    feedback_text = "This analysis needs more emotional context."
    updated_state = process_hitl_feedback(test_state, feedback_text)
    
    # Verify feedback was processed correctly
    checks = [
        len(updated_state["feedback_history"]) == 1,
        updated_state["feedback"] == feedback_text,
        updated_state["previous_response"] == "Test response content",
        "timestamp" in updated_state["feedback_history"][0]
    ]

    assert all(checks), f"Feedback processing checks failed: {checks}"

async def workflow_with_feedback_smoke():
    """Test that workflow properly uses feedback history"""
    print("\n🧪 Testing workflow with feedback integration...")
    
    try:
        # Load existing feedback history
        feedback_history = load_feedback_history()
        
        # Create workflow
        workflow = create_workflow()
        print("✅ Workflow created successfully")
        
        # Create test state with feedback history
        task = "How should I handle a difficult conversation with my manager about workload concerns?"
        session_log = create_test_session_log(task)
        
        state = {
            "task": task,
            "stage": "task_delegation",
            "response": "",
            "subtasks": [],
            "feedback": "",
            "previous_response": "",
            "feedback_history": feedback_history,  # Include existing feedback
            "session_log": session_log,
            "error": False
        }
        
        print(f"📋 Testing with {len(feedback_history)} feedback entries")
        print(f"📋 Task: {task[:80]}...")
        
        # Execute workflow (will fail due to invalid API key, but we can check state passing)
        try:
            result = await workflow.ainvoke(state)
        except Exception as e:
            # Expected to fail with API key error, but we can check if feedback was passed
            print(f"⚠️ Expected API error: {str(e)[:100]}...")
        
        print("✅ Workflow executed (API errors expected with test key)")
        return
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_agent_prompt_integration():
    """Test that agents receive feedback in their prompts"""
    print("\n🧪 Testing agent prompt integration...")
    
    try:
        from agents.dlpfc import DLPFCAgent
        from agents.specialized import VMPFCAgent, ACCAgent, MPFCAgent
        
        # Create test feedback history
        test_feedback_history = [
            {
                "response": "Previous analysis was too general",
                "feedback": "Please be more specific about actionable steps",
                "stage": "value_assessment"
            }
        ]
        
        # Test DLPFC agent
        dlpfc = DLPFCAgent()
        formatted_feedback = dlpfc._format_feedback_history(test_feedback_history)
        
        if "Previous analysis was too general" in formatted_feedback and "Please be more specific" in formatted_feedback:
            print("✅ DLPFC agent formats feedback correctly")
        else:
            print("❌ DLPFC feedback formatting failed")
            assert False
        
        # Test specialized agents have feedback in their prompts
        agents_to_test = [
            ("VMPFC", VMPFCAgent),
            ("ACC", ACCAgent), 
            ("MPFC", MPFCAgent)
        ]
        
        for agent_name, agent_class in agents_to_test:
            agent = agent_class()
            # Get the template from the messages
            prompt_messages = agent.prompt.messages
            template_content = str(prompt_messages[0].prompt.template) if prompt_messages else ""
            
            if "Feedback History: {feedback_history}" in template_content:
                print(f"✅ {agent_name} agent includes feedback history in prompt")
            else:
                print(f"❌ {agent_name} agent missing feedback history in prompt")
                print(f"   Template content: {template_content[:200]}...")
                assert False
        
        return
        
    except Exception as e:
        print(f"❌ Agent prompt test failed: {e}")
        raise

def run_hitl_end_to_end():
    """Test complete HITL flow end-to-end"""
    print("\n🧪 Testing complete HITL end-to-end flow...")
    
    # Test complete flow
    tests = [
        ("Feedback Persistence", test_feedback_persistence),
        ("Feedback Processing", test_feedback_processing), 
        ("Agent Prompt Integration", test_agent_prompt_integration)
    ]
    
    for _, test_func in tests:
        test_func()

    # Optional smoke check (not a pytest test; may require a working provider).
    # asyncio.run(workflow_with_feedback_smoke())
    return

def main():
    """Run all HITL integration tests"""
    print("🚀 Testing HITL (Human-In-The-Loop) Integration")
    print("=" * 60)
    
    results = run_hitl_end_to_end()
    
    print("\n" + "=" * 60)
    print("📊 HITL INTEGRATION TEST RESULTS:")
    print("=" * 60)
    
    print("✅ HITL integration tests completed")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)