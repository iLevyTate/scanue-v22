---
permalink: /workflow/
title: "Workflow Engine"
excerpt: "Understanding the LangGraph-powered workflow orchestration in SCANUE v22."
last_modified_at: 2025-01-31T10:24:00-05:00
toc: true
---

## Overview

SCANUE v22’s workflow engine is built on **LangGraph** (`workflow.py`). It compiles a `StateGraph` where:
- the **DLPFC** node decides which specialist stages to run (dynamic delegation)
- specialist stages run in the delegated order
- **MPFC** integrates prior agent outputs into the final answer

## Core Architecture

### State Management
The workflow passes a single shared `state` dict between stages. Key fields include:
- `task`: user input
- `stage`: current stage name
- `delegated_agents`: list of stage names selected by DLPFC (e.g. `["emotional_regulation", "conflict_detection", "value_assessment"]`)
- `agent_responses`: responses collected from specialist agents
- `feedback_history`: persisted HITL feedback loaded from `feedback_history.json`
- `session_log`: per-run timing/trace data saved under `logs/`

### Conditional Routing
Routing is dynamic:
- DLPFC output is parsed by `parse_agent_assignments()` to produce `delegated_agents`
- after each node completes, `get_next_stage()` picks the next stage based on progress (completed agents vs delegated)

### Error Handling
Specialist agent failures are handled **gracefully**: the workflow records per-agent errors and continues when possible, so a single specialist failure does not automatically abort the run.

## Workflow Components

### Stages
The compiled workflow contains these stage nodes:
- `task_delegation` (DLPFC)
- `emotional_regulation` (VMPFC)
- `reward_processing` (OFC)
- `conflict_detection` (ACC)
- `value_assessment` (MPFC)

### Transitions
Transitions are chosen at runtime based on `delegated_agents` and `agent_responses`.

### Example Workflow Structure
```python
import asyncio
from workflow import create_workflow

async def run():
    workflow = create_workflow()
    state = {
        "task": "Help me evaluate whether I should take a new role.",
        "stage": "task_delegation",
        "response": "",
        "subtasks": [],
        "feedback": "",
        "previous_response": "",
        "feedback_history": [],
        "session_log": {"stages": []},
        "error": False,
    }
    result = await workflow.ainvoke(state)
    print(result["response"]["content"])

asyncio.run(run())
```

## Human-in-the-Loop Integration

### Interactive Decision Points
The CLI (`main.py`) always offers to collect feedback after presenting the result. If you provide feedback, it is appended to `feedback_history.json` and loaded on future runs.

### Feedback Mechanisms
- **Persistent feedback history**: stored in `feedback_history.json`
- **Per-run logs**: stored under `logs/` to debug behavior and timing

### Implementation Example
```python
from main import load_feedback_history

feedback_history = load_feedback_history()
print(f"Loaded {len(feedback_history)} feedback items")
```

## Monitoring and Debugging

### Real-time Monitoring
Track workflow execution in real-time:
- **Stage Progress**: Current stage and completion percentage
- **Agent Status**: Individual agent states and activities
- **Performance Metrics**: Execution time, resource usage, success rates

### Debug Utilities
Comprehensive debugging tools:
- **Workflow Visualization**: Graphical representation of workflow structure
- **State Inspection**: Detailed view of workflow and agent states
- **Execution Traces**: Step-by-step execution history

### Debug Scripts
The system includes several debug utilities:
- `debug_workflow.py`: General workflow debugging
- `debug_stage_transitions.py`: Stage transition analysis
- `debug_langgraph_mapping.py`: LangGraph integration debugging
- `demonstrate_hitl.py`: Human-in-the-loop demonstration

## Performance Optimization

### Execution Strategies
- **Eager Execution**: Immediate processing for time-critical tasks
- **Lazy Evaluation**: Deferred processing for resource optimization
- **Batch Processing**: Grouping similar tasks for efficiency

### Resource Management
- **Memory Pooling**: Efficient memory allocation and reuse
- **Connection Pooling**: Optimized external service connections
- **Load Balancing**: Distributing work across available resources

### Caching Strategies
- **Result Caching**: Storing computation results for reuse
- **State Caching**: Persisting workflow states for quick recovery
- **Agent Model Caching**: Caching trained models for faster initialization

## Configuration
Agent/model configuration is defined in `config/agents.yaml`. See:
- [Configuration]({{ "/docs/configuration/" | relative_url }})
- [Local & Multi-Model Setup]({{ "/docs/local-models/" | relative_url }})

## Testing Workflows

### Unit Testing
Test individual workflow components:
```python
def test_stage_transition():
    # See tests/ for end-to-end and workflow integrity tests.
    assert True
```

### Integration Testing
Test complete workflow execution:
```python
def test_full_workflow():
    # See tests/test_full_workflow.py
    assert True
```

### Performance Testing
Measure workflow performance:
```python
def test_workflow_performance():
    # Use session logs (logs/) and profiling as needed.
    assert True
```

---

*For more detailed information about workflow implementation, see our [technical documentation]({{ "/docs/workflow-engine/" | relative_url }}).*