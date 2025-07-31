#!/usr/bin/env python3
"""Debug script to test workflow agent sequencing"""

import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from workflow import parse_agent_assignments, get_next_delegated_stage
    print("✅ Successfully imported workflow functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_agent_parsing():
    """Test if agent parsing works correctly"""
    
    # This is the actual DLPFC response from the log
    dlpfc_response = """📋 Subtasks:
  • Assess the emotional responses of the team to the meeting and its placement.
  • Evaluate the potential risks of increased stress due to back-to-back feedback sessions.
  • Propose a more balanced schedule for these meetings or a clear context for the sessions.

🔥 Agent Assignments:
  • VMPFC Agent: Assess team emotions regarding the back-to-back meetings.
  • ACC Agent: Evaluate potential conflicts or issues arising from the current scheduling.
  • MPFC Agent: Make a value-based decision on how to proceed with the meeting structure.

🔄 Integration Plan:
  • Integrate findings from the VMPFC and ACC on team emotional responses and potential conflicts. Afterward, the MPFC will finalize the meeting schedule based on these evaluations."""

    print("Testing agent parsing...")
    print("DLPFC Response:", dlpfc_response[:200] + "...")
    
    try:
        delegated_agents = parse_agent_assignments(dlpfc_response)
        print(f"Parsed delegated_agents: {delegated_agents}")
        return delegated_agents
    except Exception as e:
        print(f"❌ Error parsing agents: {e}")
        return []

def test_stage_transitions(delegated_agents):
    """Test stage transition logic"""
    
    print("\nTesting stage transitions...")
    
    # Simulate state
    state = {
        "delegated_agents": delegated_agents,
        "stage": "task_delegation"
    }
    
    print(f"Initial delegated_agents: {delegated_agents}")
    
    # Test transitions
    current = "task_delegation"
    sequence = [current]
    
    # After task_delegation, should go to first agent
    if delegated_agents:
        next_stage = delegated_agents[0]
        print(f"After {current} -> should go to: {next_stage}")
        sequence.append(next_stage)
        current = next_stage
    
    # Test subsequent transitions
    for i in range(len(delegated_agents)):
        try:
            next_stage = get_next_delegated_stage(state, current)
            print(f"After {current} -> get_next_delegated_stage returns: {next_stage}")
            if next_stage == "END":
                break
            sequence.append(next_stage)
            current = next_stage
        except Exception as e:
            print(f"❌ Error getting next stage: {e}")
            break
    
    print(f"Full sequence: {' -> '.join(sequence)}")
    print(f"Expected sequence: task_delegation -> emotional_regulation -> conflict_detection -> value_assessment")
    
    return sequence

if __name__ == "__main__":
    print("🔍 Starting workflow debug analysis...")
    
    try:
        delegated_agents = test_agent_parsing()
        sequence = test_stage_transitions(delegated_agents)
        
        # Check if the sequence is correct
        expected = ["task_delegation", "emotional_regulation", "conflict_detection", "value_assessment"]
        
        print(f"\n--- ANALYSIS ---")
        print(f"Expected: {expected}")
        print(f"Actual:   {sequence}")
        print(f"Match: {sequence == expected}")
        
        if sequence != expected:
            print("\n❌ WORKFLOW BUG IDENTIFIED!")
            print("The workflow is not following the correct agent sequence.")
        else:
            print("\n✅ Workflow sequence is correct.")
            
    except Exception as e:
        print(f"❌ Script execution error: {e}")
        import traceback
        traceback.print_exc() 