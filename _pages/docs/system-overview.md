---
permalink: /docs/system-overview/
title: "System Overview"
excerpt: "High-level architecture and key modules."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Key modules
- `main.py`: CLI loop, feedback persistence (`feedback_history.json`), session logging (`logs/`)
- `workflow.py`: LangGraph `StateGraph` definition and dynamic delegation logic
- `agents/`: agent implementations + model provider factory
- `config/agents.yaml`: per-agent model/provider configuration

## Execution flow
1. User enters a task
2. `workflow.create_workflow()` runs `task_delegation` (DLPFC)
3. DLPFC output is parsed into `delegated_agents` (stage names)
4. Specialist stages run in sequence and store results in `agent_responses`
5. `value_assessment` (MPFC) integrates the final response
6. CLI optionally collects feedback and persists it for future runs


