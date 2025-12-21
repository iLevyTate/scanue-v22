---
permalink: /docs/debugging/
title: "Debugging"
excerpt: "Debug scripts and where to find execution traces."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Session logs
Each run writes a session log under `logs/` containing timing, stage inputs/outputs, and errors.

## Debug scripts
The `debug/` directory contains scripts for investigating workflow behavior:
- `debug_workflow.py`
- `debug_stage_transitions.py`
- `debug_langgraph_mapping.py`
- `demonstrate_hitl.py`

## Run a debug script (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
python .\debug\debug_workflow.py
```


