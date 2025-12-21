---
permalink: /docs/quick-start-guide/
title: "Quick Start Guide"
excerpt: "Run SCANUE v22 locally in minutes."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Prerequisites
- Python 3.8+
- One model provider configured:
  - **Ollama (local)** (default in `config/agents.yaml`)
  - **OpenAI**
  - **HuggingFace** (endpoint/TGI)

## Quick start (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

When prompted, enter a task. Type `exit` to quit.

## What gets written to disk
- `feedback_history.json`: persistent feedback you optionally provide after a run
- `logs/`: per-run session logs (timing + stage traces)


