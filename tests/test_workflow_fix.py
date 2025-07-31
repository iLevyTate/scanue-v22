#!/usr/bin/env python3
"""Test if the workflow fix properly engages all agents"""

import os
import sys
import asyncio
from datetime import datetime
import uuid

# Set a dummy API key for testing
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key-for-debugging"

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

async def test_workflow_fix():
    """Test if the workflow fix works"""
    
    print("🧠 Testing workflow fix...")
    
    # Create workflow
    try:
        workflow = create_workflow()
        print("✅ Workflow created successfully")
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return False
    
    # Use the exact same task from the failing log
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
    
    print(f"📋 Testing with task: {task[:100]}...")
    print(f"📋 Initial state stage: {state['stage']}")
    
    # Execute workflow
    try:
        print("\n🔄 Starting workflow execution...")
        result = await workflow.ainvoke(state)
        
        print("\n📊 WORKFLOW RESULTS:")
        print(f"Final stage: {result.get('stage', 'UNKNOWN')}")
        print(f"Error: {result.get('error', False)}")
        print(f"Delegated agents: {result.get('delegated_agents', [])}")
        
        # Analyze stages executed
        session_log = result.get('session_log', {})
        stages = session_log.get('stages', [])
        
        print(f"\n📝 STAGES EXECUTED ({len(stages)} total):")
        executed_agents = []
        for i, stage in enumerate(stages):
            agent = stage.get('agent', 'UNKNOWN')
            stage_name = stage.get('stage', 'UNKNOWN')
            duration = stage.get('duration_ms', 0)
            print(f"  {i+1}. {stage_name} ({agent}) - {duration}ms")
            executed_agents.append(agent)
        
        # Check if all expected agents were executed
        expected_agents = ['DLPFC', 'VMPFC', 'ACC', 'MPFC']  # Expected order
        
        print(f"\n🔍 AGENT ANALYSIS:")
        print(f"Expected agents: {expected_agents}")
        print(f"Executed agents: {executed_agents}")
        
        missing_agents = set(expected_agents) - set(executed_agents)
        if missing_agents:
            print(f"❌ MISSING AGENTS: {list(missing_agents)}")
            print("🔧 The fix did NOT work - agents are still being skipped!")
            return False
        else:
            print("✅ ALL EXPECTED AGENTS EXECUTED!")
            print("🎉 The fix WORKED - all agents are properly engaged!")
            return True
            
    except Exception as e:
        print(f"❌ Workflow execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing if workflow fix worked...")
    
    try:
        success = asyncio.run(test_workflow_fix())
        if success:
            print("\n🎯 RESULT: ✅ WORKFLOW FIX SUCCESSFUL")
            print("All agents are now properly engaged in the workflow!")
        else:
            print("\n🎯 RESULT: ❌ WORKFLOW FIX FAILED")
            print("Agents are still being skipped - more work needed.")
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()