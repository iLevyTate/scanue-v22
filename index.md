---
layout: home
author_profile: true
classes: wide
title: "SCANUE v22: Advanced Multi-Agent Workflow System"
excerpt: "A sophisticated multi-agent system leveraging LangGraph for complex workflow orchestration and human-in-the-loop interactions."
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  actions:
    - label: "View on GitHub"
      url: "https://github.com/iLevyTate/scanue-v22"
    - label: "Documentation"
      url: "/docs/"
feature_row:
  - title: "Intelligent Agents"
    excerpt: "Specialized agents for different cognitive functions including DLPFC and task-specific processing."
    url: "/agents/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - title: "LangGraph Workflows"
    excerpt: "Advanced workflow orchestration with state management and conditional routing."
    url: "/workflow/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - title: "Human-in-the-Loop"
    excerpt: "Seamless integration of human feedback and decision-making in automated workflows."
    url: "/workflow/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
---

{% include feature_row %}

## Welcome to SCANUE v22

SCANUE v22 is a sophisticated multi-agent system that leverages the power of LangGraph for complex workflow orchestration. The system is designed to handle intricate cognitive tasks through specialized agents, each optimized for specific functions.

### Key Features

- **Multi-Agent Architecture**: Specialized agents for different cognitive functions
- **LangGraph Integration**: Advanced workflow orchestration with state management
- **Human-in-the-Loop**: Seamless integration of human feedback and decision-making
- **Flexible Workflow Design**: Conditional routing and dynamic task allocation
- **Comprehensive Testing**: Full test coverage with pytest framework

### Getting Started

```python
# Clone the repository
git clone https://github.com/iLevyTate/scanue-v22.git
cd scanue-v22

# Install dependencies
pip install -r requirements.txt

# Run the main workflow
python main.py
```

### Project Structure

The project is organized into several key components:

- **`agents/`** - Core agent implementations including base, DLPFC, and specialized agents
- **`tests/`** - Comprehensive test suite covering all system components
- **`debug/`** - Debugging utilities and workflow demonstration scripts
- **`workflow.py`** - Main workflow orchestration logic
- **`main.py`** - Entry point for the application

