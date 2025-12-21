---
permalink: /docs/installation/
title: "Installation"
excerpt: "Install dependencies and run SCANUE v22."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Install (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
python main.py
```

## Provider prerequisites
- **Ollama**: install Ollama and ensure it’s running (default `base_url` is `http://localhost:11434`)
- **OpenAI**: set `OPENAI_API_KEY` (or set `api_key:` in `config/agents.yaml`)
- **HuggingFace**: set `HUGGINGFACEHUB_API_TOKEN` (or set `api_key:` in `config/agents.yaml`)


