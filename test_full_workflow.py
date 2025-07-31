#!/usr/bin/env python3
"""Comprehensive test to debug actual workflow execution"""

import sys
import os
import asyncio
from datetime import datetime
import uuid

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from workflow import create_workflow
    print("✅ Successfully imported workflow functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

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

async def test_workflow_execution():
    """Test actual workflow execution"""
    
    print("🔍 Testing actual workflow execution...")
    
    # Create workflow
    try:
        workflow = create_workflow()
        print("✅ Workflow created successfully")
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return
    
    # Create test state
    task = "My office manager put a meeting on the calendar for a succession of one on one staff work feedback sessions which is not typical being back to back without context and has caused stress within the office."
    
    session_log = create_test_session_log(task)
    
    state = {
        "task": task,
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "session_log": session_log,
        "error": False
    }
    
    print(f"📋 Initial state: {state['stage']}")
    print(f"📋 Task: {task[:100]}...")
    
    # Execute workflow and trace the stages
    try:
        print("\n🔄 Starting workflow execution...")
        result = await workflow.ainvoke(state)
        
        print("\n📊 WORKFLOW EXECUTION RESULTS:")
        print(f"Final stage: {result.get('stage', 'UNKNOWN')}")
        print(f"Error: {result.get('error', False)}")
        print(f"Delegated agents: {result.get('delegated_agents', [])}")
        
        # Show all stages that were executed
        session_log = result.get('session_log', {})
        stages = session_log.get('stages', [])
        
        print(f"\n📝 STAGES EXECUTED ({len(stages)} total):")
        for i, stage in enumerate(stages):
            agent = stage.get('agent', 'UNKNOWN')
            stage_name = stage.get('stage', 'UNKNOWN')
            duration = stage.get('duration_ms', 0)
            print(f"  {i+1}. {stage_name} ({agent}) - {duration}ms")
        
        # Show expected vs actual sequence
        delegated_agents = result.get('delegated_agents', [])
        expected_sequence = ['task_delegation'] + delegated_agents
        actual_sequence = [s.get('stage') for s in stages]
        
        print(f"\n🔍 SEQUENCE ANALYSIS:")
        print(f"Expected: {expected_sequence}")
        print(f"Actual:   {actual_sequence}")
        print(f"Match: {expected_sequence == actual_sequence}")
        
        if expected_sequence != actual_sequence:
            print("\n❌ WORKFLOW EXECUTION BUG CONFIRMED!")
            print("The workflow is not executing all delegated agents.")
            
            # Find missing stages
            missing = set(expected_sequence) - set(actual_sequence)
            extra = set(actual_sequence) - set(expected_sequence)
            
            if missing:
                print(f"Missing stages: {list(missing)}")
            if extra:
                print(f"Extra stages: {list(extra)}")
        else:
            print("\n✅ Workflow execution is correct.")
            
        return result
        
    except Exception as e:
        print(f"❌ Workflow execution error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧠 Starting comprehensive workflow execution test...")
    
    # Set a dummy API key for testing
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test-key-for-debugging"
        print("⚠️  Using dummy API key for testing")
    
    try:
        result = asyncio.run(test_workflow_execution())
        if result:
            print("\n✅ Test completed successfully")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc() 