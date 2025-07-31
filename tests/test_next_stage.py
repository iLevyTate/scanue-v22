#!/usr/bin/env python3
"""Test the get_next_stage function behavior"""

import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow import get_next_delegated_stage

def test_get_next_stage_function():
    """Test the get_next_stage conditional function logic"""
    
    print("🔍 Testing get_next_stage function...")
    
    # Simulate the get_next_stage function from workflow.py
    def get_next_stage(state):
        if state.get("error"):
            return "__end__"
            
        current_stage = state.get("stage", "")
        if current_stage == "__end__":
            return "__end__"
        
        # For task_delegation, use the first delegated agent
        if current_stage == "task_delegation":
            delegated_agents = state.get("delegated_agents", [])
            result = delegated_agents[0] if delegated_agents else "__end__"
            print(f"  task_delegation -> {result}")
            return result
        
        # For other stages, get next from delegation list
        result = get_next_delegated_stage(state, current_stage)
        print(f"  {current_stage} -> {result}")
        return result
    
    # Test the exact scenario from the log
    delegated_agents = ['emotional_regulation', 'conflict_detection', 'value_assessment']
    
    print(f"Testing with delegated_agents: {delegated_agents}")
    print()
    
    # Test each stage transition
    test_states = [
        {"stage": "task_delegation", "delegated_agents": delegated_agents, "error": False},
        {"stage": "emotional_regulation", "delegated_agents": delegated_agents, "error": False},
        {"stage": "conflict_detection", "delegated_agents": delegated_agents, "error": False},
        {"stage": "value_assessment", "delegated_agents": delegated_agents, "error": False}
    ]
    
    sequence = []
    for state in test_states:
        next_stage = get_next_stage(state)
        sequence.append(f"{state['stage']} -> {next_stage}")
        
        if next_stage == "__end__":
            break
    
    print("Full transition sequence:")
    for transition in sequence:
        print(f"  {transition}")
    
    expected_sequence = [
        "task_delegation -> emotional_regulation",
        "emotional_regulation -> conflict_detection", 
        "conflict_detection -> value_assessment",
        "value_assessment -> __end__"
    ]
    
    print(f"\nExpected: {expected_sequence}")
    print(f"Actual:   {sequence}")
    print(f"Match: {sequence == expected_sequence}")
    
    if sequence != expected_sequence:
        print("\n❌ get_next_stage function has issues!")
    else:
        print("\n✅ get_next_stage function works correctly")
    
    return sequence == expected_sequence

if __name__ == "__main__":
    print("🧠 Testing get_next_stage function behavior...")
    
    try:
        success = test_get_next_stage_function()
        if success:
            print("\n✅ Test passed - get_next_stage function is correct")
            print("⚠️  The issue must be elsewhere in the workflow execution")
        else:
            print("\n❌ Test failed - get_next_stage function has bugs")
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc() 