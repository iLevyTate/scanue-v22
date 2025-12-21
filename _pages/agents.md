---
permalink: /agents/
title: "Agent Architecture"
excerpt: "Explore the multi-agent architecture of SCANUE v22 and its specialized cognitive agents."
last_modified_at: 2025-01-31T10:24:00-05:00
toc: true
---

## Overview

SCANUE v22 uses a small set of specialized agents that mirror “PFC region” roles. The workflow always starts with DLPFC (delegation) and ends with MPFC (integration), with VMPFC/OFC/ACC invoked only when DLPFC delegates them.

## Base Agent

All agents inherit from `agents/base.py::BaseAgent`, which provides:

### Features
- **Common Interface**: Standardized methods for agent communication
- **State Management**: Persistent state handling across interactions
- **Error Handling**: Robust error management and recovery
- **Logging**: Comprehensive logging for debugging and monitoring

### Notes
- `BaseAgent` is **abstract** (you don’t instantiate it directly).
- Models are loaded from `config/agents.yaml` (with environment variable fallback for legacy setups).

### Example (specialized agent)
```python
import asyncio
from agents.specialized import VMPFCAgent

async def run():
    agent = VMPFCAgent()
    state = {"task": "I need to decide whether to switch jobs.", "feedback_history": []}
    result = await agent.process(state)
    print(result["response"]["content"])

asyncio.run(run())
```

## DLPFC Agent

The Dorsolateral Prefrontal Cortex (DLPFC) agent specializes in executive functions and cognitive control, mimicking the cognitive processes of the human prefrontal cortex.

### Capabilities
- **Executive Control**: High-level decision making and task coordination
- **Working Memory**: Temporary information storage and manipulation
- **Cognitive Flexibility**: Adapting strategies based on context
- **Attention Management**: Focusing on relevant information

### Specialized Functions
- Strategic planning and goal management
- Conflict monitoring and resolution
- Abstract reasoning and problem-solving
- Multi-tasking coordination

### Usage
```python
import asyncio
from agents.dlpfc import DLPFCAgent

async def run():
    dlpfc = DLPFCAgent()
    state = {"task": "Help me plan a hard conversation with my manager.", "feedback_history": []}
    result = await dlpfc.process(state)
    print(result["response"]["content"])

asyncio.run(run())
```

## Specialized Agents

SCANUE v22 ships four specialist agents (in `agents/specialized.py`):
- **VMPFC**: emotional/social framing and risk nuance
- **OFC**: reward/cost tradeoffs and outcomes
- **ACC**: conflict/error/contradiction checking
- **MPFC**: final integration and recommendation

### Design Principles
- **Domain Expertise**: Deep specialization in specific areas
- **Efficient Processing**: Optimized algorithms for target tasks
- **Interoperability**: Seamless integration with other agents
- **Scalability**: Ability to handle varying workloads

### Implementation Example (new agent)
```python
from agents.specialized import SpecializedAgent

class AnalysisAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(name="analysis_agent")
        self.specialty = "data_analysis"
    
    def analyze_data(self, dataset):
        # Specialized analysis logic
        return self.perform_analysis(dataset)
```

*Note: SCANUE’s default workflow graph only includes the built-in agents. To wire in a new agent you’ll also update `workflow.py` (add a node + include it in delegation/edge mappings).*

## Agent Coordination

### Communication Patterns
- **Direct Messaging**: Point-to-point communication between agents
- **Broadcast Messages**: One-to-many communication for coordination
- **Event-Driven Updates**: Reactive communication based on system events

### Workflow Integration
Agents are seamlessly integrated into the LangGraph workflow engine, allowing for:
- Dynamic agent selection based on task requirements
- Parallel agent execution for improved performance
- Sequential agent chaining for complex processing pipelines

### Example Coordination
```python
from workflow import create_workflow

# The workflow is defined in workflow.py and runs a LangGraph StateGraph.
workflow = create_workflow()
```

## Human-Agent Interaction

### Interactive Decision Points
Agents can request human input at critical decision points, ensuring that human expertise is leveraged when needed.

### Feedback Learning
Agents adapt their behavior based on human feedback, improving performance over time.

### Collaboration Modes
- **Advisory Mode**: Agents provide recommendations for human decision-making
- **Supervisory Mode**: Humans oversee agent actions with intervention capabilities
- **Collaborative Mode**: Real-time collaboration between humans and agents

## Performance Monitoring

### Metrics Collection
- **Response Time**: Agent processing duration
- **Accuracy**: Task completion success rates
- **Resource Usage**: Memory and CPU utilization
- **Interaction Quality**: Human feedback scores

### Optimization Strategies
- **Load Balancing**: Distributing tasks across available agents
- **Caching**: Storing frequently accessed data for faster retrieval
- **Model Fine-tuning**: Adjusting agent parameters based on performance data

## Testing and Validation

Each agent type includes comprehensive test coverage:
- **Unit Tests**: Individual agent functionality
- **Integration Tests**: Agent interaction and coordination
- **Performance Tests**: Load and stress testing
- **Behavioral Tests**: Validation of cognitive functions

---

*Learn more about implementing custom agents in our [development documentation]({{ "/docs/agent-design/" | relative_url }}).*