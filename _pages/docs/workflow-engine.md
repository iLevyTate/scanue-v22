---
permalink: /docs/workflow-engine/
title: "Workflow Engine"
excerpt: "How the LangGraph workflow is built and how delegation works."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Where it lives
The workflow is implemented in `workflow.py`.

## Stages (nodes)
- `task_delegation` (DLPFC)
- `emotional_regulation` (VMPFC)
- `reward_processing` (OFC)
- `conflict_detection` (ACC)
- `value_assessment` (MPFC)

## Delegation
DLPFC produces a response that includes an **AGENT DELEGATION** section. `parse_agent_assignments()` extracts which specialist stages to run and stores them as `delegated_agents`.

## Progress tracking
After each stage, `get_next_stage()` selects the next stage by comparing:
- total delegated agents (`delegated_agents`)
- completed agents (`agent_responses`)

This avoids manual `stage` mutation bugs and keeps routing deterministic.


