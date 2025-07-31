#!/usr/bin/env python3
"""Test parsing of the original DLPFC response"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow import parse_agent_assignments

# The exact DLPFC response from the original log
original_dlpfc_response = """📋 Subtasks:
  • Assess the emotional responses of the team to the meeting and its placement.
  • Evaluate the potential risks of increased stress due to back-to-back feedback sessions.
  • Propose a more balanced schedule for these meetings or a clear context for the sessions.

🔥 Agent Assignments:
  • VMPFC Agent: Assess team emotions regarding the back-to-back meetings.
  • ACC Agent: Evaluate potential conflicts or issues arising from the current scheduling.
  • MPFC Agent: Make a value-based decision on how to proceed with the meeting structure.

🔄 Integration Plan:
  • Integrate findings from the VMPFC and ACC on team emotional responses and potential conflicts. Afterward, the MPFC will finalize the meeting schedule based on these evaluations."""

print("🔍 Testing original DLPFC response parsing...")
print("Original DLPFC response:")
print("-" * 50)
print(original_dlpfc_response)
print("-" * 50)

parsed_agents = parse_agent_assignments(original_dlpfc_response)

print(f"\n🎯 PARSED RESULT: {parsed_agents}")
print(f"Expected: ['emotional_regulation', 'conflict_detection', 'value_assessment']")
print(f"Contains OFC: {'reward_processing' in parsed_agents}")

if 'reward_processing' in parsed_agents:
    print("\n❌ BUG FOUND: OFC should NOT be in the parsed agents!")
    print("The DLPFC response doesn't mention OFC at all!")
else:
    print("\n✅ Parsing is correct - no OFC agent")

# Let's also test what the workflow should do
print(f"\n📋 WORKFLOW SEQUENCE TEST:")
print(f"1. task_delegation → {parsed_agents[0] if parsed_agents else 'END'}")

for i, agent in enumerate(parsed_agents[:-1]):
    next_agent = parsed_agents[i + 1]
    print(f"{i + 2}. {agent} → {next_agent}")

if parsed_agents:
    print(f"{len(parsed_agents) + 1}. {parsed_agents[-1]} → END")