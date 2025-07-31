#!/usr/bin/env python3
"""Test to examine workflow graph structure and conditional edges"""

import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from workflow import create_workflow
    print("✅ Successfully imported workflow functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_workflow_structure():
    """Test the workflow graph structure"""
    
    print("🔍 Testing workflow graph structure...")
    
    # Create workflow
    try:
        workflow = create_workflow()
        print("✅ Workflow created successfully")
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return
    
    # Examine the workflow graph structure
    print("\n📊 WORKFLOW GRAPH ANALYSIS:")
    
    # Get the compiled graph
    try:
        # Access the underlying graph structure
        graph = workflow.graph
        print(f"Graph nodes: {list(graph.nodes.keys())}")
        
        # Examine edges
        print("\n🔗 GRAPH EDGES:")
        for node_name, node in graph.nodes.items():
            print(f"\nNode: {node_name}")
            # Check if node has edges
            if hasattr(node, 'edges') or node_name in graph.edges:
                edges = graph.edges.get(node_name, [])
                if edges:
                    for edge in edges:
                        print(f"  -> {edge}")
                else:
                    print("  -> No edges found")
            else:
                print("  -> No edge information available")
        
        # Test the conditional edge function directly  
        print("\n🧪 TESTING CONDITIONAL EDGE FUNCTION:")
        
        # Test state after task_delegation
        test_state = {
            "stage": "task_delegation",
            "delegated_agents": ['emotional_regulation', 'conflict_detection', 'value_assessment'],
            "error": False
        }
        
        # We need to get the conditional function that determines the next stage
        # This is tricky because it's embedded in the graph, but let's try to access it
        print(f"Test state: {test_state}")
        
        # Let's examine what happens with different stage transitions
        test_cases = [
            ("task_delegation", ['emotional_regulation', 'conflict_detection', 'value_assessment']),
            ("emotional_regulation", ['emotional_regulation', 'conflict_detection', 'value_assessment']),
            ("conflict_detection", ['emotional_regulation', 'conflict_detection', 'value_assessment']),
            ("value_assessment", ['emotional_regulation', 'conflict_detection', 'value_assessment'])
        ]
        
        for stage, delegated_agents in test_cases:
            state = {
                "stage": stage, 
                "delegated_agents": delegated_agents,
                "error": False
            }
            print(f"\nTesting stage transition from '{stage}':")
            print(f"  State: {state}")
            
            # Try to determine what the next stage should be
            if stage == "task_delegation":
                expected_next = delegated_agents[0] if delegated_agents else "__end__"
            else:
                # Use the same logic as get_next_delegated_stage
                try:
                    current_index = delegated_agents.index(stage)
                    if current_index + 1 < len(delegated_agents):
                        expected_next = delegated_agents[current_index + 1]
                    else:
                        expected_next = "__end__"
                except ValueError:
                    expected_next = "__end__"
            
            print(f"  Expected next stage: {expected_next}")
        
    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧠 Starting workflow structure test...")
    
    try:
        test_workflow_structure()
        print("\n✅ Structure test completed")
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc() 