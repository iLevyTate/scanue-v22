---
permalink: /docs/agent-design/
title: "Agent Design"
excerpt: "How agents are structured and how to extend them."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## BaseAgent contract
All agents inherit from `BaseAgent` (`agents/base.py`) and must implement:
- `_create_prompt()`: returns a `ChatPromptTemplate`
- `process(state)`: asynchronous; returns a dict including `response` and `error`

## Model selection
Agents load models via:
- `config/agents.yaml` (preferred)
- legacy env-var fallback if config is missing/incomplete

Providers are created through `agents/factory.py::LLMFactory`.

## Adding a new agent
1. Create a new agent class (subclass `BaseAgent`) with a prompt and `process`.
2. Update `workflow.py` to:
   - add a new node
   - include the stage in the conditional edge mappings
   - update delegation logic if you want DLPFC to select it


