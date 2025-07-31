#!/usr/bin/env python3
"""Debug stage transitions to see why agents are being skipped"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow import get_next_delegated_stage

def debug_stage_transitions():
    """Debug the exact stage transition logic"""
    
    print("🔍 Debugging stage transitions...")
    
    # Test the exact scenario from the failing execution
    delegated_agents = ['emotional_regulation', 'reward_processing', 'conflict_detection', 'value_assessment']
    
    # Test each transition step by step
    test_cases = [
        {
            "description": "After task_delegation fails", 
            "state": {
                "delegated_agents": delegated_agents,
                "stage": "task_delegation",
                "agent_errors": {"DLPFC": "API error"}
            },
            "current_stage": "task_delegation"
        },
        {
            "description": "After emotional_regulation", 
            "state": {
                "delegated_agents": delegated_agents,
                "stage": "emotional_regulation",
                "agent_errors": {"DLPFC": "API error"}
            },
            "current_stage": "emotional_regulation"
        },
        {
            "description": "After reward_processing", 
            "state": {
                "delegated_agents": delegated_agents,
                "stage": "reward_processing",
                "agent_errors": {"DLPFC": "API error"}
            },
            "current_stage": "reward_processing"
        },
        {
            "description": "After conflict_detection", 
            "state": {
                "delegated_agents": delegated_agents,
                "stage": "conflict_detection",
                "agent_errors": {"DLPFC": "API error"}
            },
            "current_stage": "conflict_detection"
        }
    ]
    
    print(f"Delegated agents: {delegated_agents}")
    print()
    
    for test_case in test_cases:
        description = test_case["description"]
        state = test_case["state"]
        current_stage = test_case["current_stage"]
        
        print(f"--- {description} ---")
        print(f"Current stage: {current_stage}")
        print(f"State: {state}")
        
        try:
            next_stage = get_next_delegated_stage(state, current_stage)
            print(f"Next stage: {next_stage}")
            
            # Show the index calculation
            if current_stage in delegated_agents:
                current_index = delegated_agents.index(current_stage)
                print(f"Current index in delegated_agents: {current_index}")
                if current_index + 1 < len(delegated_agents):
                    expected_next = delegated_agents[current_index + 1]
                    print(f"Expected next agent: {expected_next}")
                else:
                    print("Should go to END (reached end of list)")
            else:
                print(f"'{current_stage}' not found in delegated_agents!")
                
        except Exception as e:
            print(f"Error: {e}")
            
        print()
    
    # Test the specific issue: why does task_delegation go to reward_processing?
    print("🚨 SPECIFIC ISSUE TEST:")
    print("Why does task_delegation → reward_processing instead of emotional_regulation?")
    
    # The issue might be that task_delegation is not in delegated_agents
    if "task_delegation" not in delegated_agents:
        print("✅ task_delegation is NOT in delegated_agents (this is correct)")
        print("When get_next_delegated_stage is called with 'task_delegation', it should return END")
        
        result = get_next_delegated_stage({
            "delegated_agents": delegated_agents
        }, "task_delegation")
        
        print(f"get_next_delegated_stage('task_delegation') = {result}")
        
        print("\n🤔 But in the workflow, task_delegation should go to the FIRST delegated agent")
        print(f"First delegated agent should be: {delegated_agents[0]}")

if __name__ == "__main__":
    try:
        debug_stage_transitions()
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()