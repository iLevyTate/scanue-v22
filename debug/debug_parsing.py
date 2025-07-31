#!/usr/bin/env python3
"""Debug the exact parsing behavior and agent ordering"""

import sys
import os
import re

# Add the current directory to the path to ensure imports work  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_parse_agent_assignments(dlpfc_response: str) -> list:
    """Debug version of parse_agent_assignments with detailed logging."""
    try:
        agent_assignments = []
        
        # Agent name mappings
        agent_map = {
            'VMPFC': 'emotional_regulation',
            'OFC': 'reward_processing', 
            'ACC': 'conflict_detection',
            'MPFC': 'value_assessment'
        }
        
        print("🔍 DEBUGGING AGENT PARSING:")
        print(f"Input response length: {len(dlpfc_response)} characters")
        print(f"Agent map: {agent_map}")
        
        # Look for agent assignments in the response
        response_lower = dlpfc_response.lower()
        print(f"\nSearching in lowercase response...")
        
        # Check for explicit agent mentions
        for agent_name, stage_name in agent_map.items():
            print(f"\n--- Checking for {agent_name} ---")
            patterns = [
                f"{agent_name.lower()} agent",
                f"{agent_name.lower()}:",
                f"assign.*{agent_name.lower()}",
                f"delegate.*{agent_name.lower()}",
                f"{agent_name.lower()}.*agent"
            ]
            
            found = False
            for pattern in patterns:
                print(f"  Pattern: '{pattern}'")
                if re.search(pattern, response_lower):
                    print(f"  ✅ MATCH found!")
                    if stage_name not in agent_assignments:
                        agent_assignments.append(stage_name)
                        print(f"  Added '{stage_name}' to assignments")
                        found = True
                        break
                    else:
                        print(f"  Stage '{stage_name}' already in assignments")
                else:
                    print(f"  ❌ No match")
            
            if not found:
                print(f"  No patterns matched for {agent_name}")
        
        print(f"\nInitial agent_assignments: {agent_assignments}")
        
        # If no specific agents are mentioned, use default sequence
        if not agent_assignments:
            print("No agents found, using default sequence...")
            # Default to calling all agents in logical order
            agent_assignments = [
                'emotional_regulation',
                'reward_processing', 
                'conflict_detection',
                'value_assessment'
            ]
        
        # Always include value_assessment (MPFC) as the final stage for integration
        if 'value_assessment' not in agent_assignments:
            print("Adding value_assessment as final stage...")
            agent_assignments.append('value_assessment')
        
        print(f"Final agent_assignments: {agent_assignments}")
        return agent_assignments
    
    except Exception as e:
        print(f"❌ Error in debug_parse_agent_assignments: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        print("🧠 Debugging agent parsing...")
        
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

        print("Original DLPFC response:")
        print("-" * 50)
        print(dlpfc_response)
        print("-" * 50)
        
        delegated_agents = debug_parse_agent_assignments(dlpfc_response)
        
        print(f"\n🎯 FINAL RESULT: {delegated_agents}")
        print(f"Expected order: ['emotional_regulation', 'conflict_detection', 'value_assessment']")
        
        expected = ['emotional_regulation', 'conflict_detection', 'value_assessment']
        if delegated_agents == expected:
            print("✅ Parsing is correct!")
        else:
            print("❌ Parsing order is wrong!")
            print(f"Expected: {expected}")
            print(f"Got:      {delegated_agents}")
            
    except Exception as e:
        print(f"❌ Script execution error: {e}")
        import traceback
        traceback.print_exc() 