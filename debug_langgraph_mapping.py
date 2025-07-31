#!/usr/bin/env python3
"""Debug LangGraph edge mappings to see why nodes are being skipped"""

import os
import sys

# Set dummy API key
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-key"

try:
    from workflow import create_workflow
    print("✅ Successfully imported workflow")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_workflow_graph_structure():
    """Test the LangGraph structure to understand why nodes are skipped"""
    
    print("🔍 Testing LangGraph structure...")
    
    try:
        workflow = create_workflow()
        print("✅ Workflow created")
        
        # Try to access the compiled graph structure
        print("\n📊 ANALYZING WORKFLOW GRAPH:")
        
        # Check if we can get information about the nodes and edges
        if hasattr(workflow, 'graph'):
            graph = workflow.graph
            print(f"Graph object: {type(graph)}")
            
            if hasattr(graph, 'nodes'):
                print(f"Nodes: {list(graph.nodes.keys())}")
            
            if hasattr(graph, 'edges'):
                print("Edges:")
                for node, edges in graph.edges.items():
                    print(f"  {node} → {edges}")
                    
        # Test a minimal state transition
        print("\n🧪 TESTING MINIMAL WORKFLOW:")
        
        # Create a minimal test state
        test_state = {
            "task": "test task",
            "stage": "emotional_regulation",  # Start at emotional_regulation
            "response": "",
            "subtasks": [],
            "feedback": "",
            "previous_response": "",
            "feedback_history": [],
            "session_log": {"stages": []},
            "error": False,
            "delegated_agents": ["emotional_regulation", "reward_processing", "conflict_detection", "value_assessment"]
        }
        
        print(f"Test state: stage = {test_state['stage']}")
        print(f"Delegated agents: {test_state['delegated_agents']}")
        
        # Try to invoke just the emotional_regulation stage
        print("\n🔄 Attempting to run emotional_regulation directly...")
        
        # We can't easily test individual nodes in LangGraph, so let's create a debug version
        print("Creating debug workflow to trace execution...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_workflow_graph_structure()