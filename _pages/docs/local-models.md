---
permalink: /docs/local-models/
title: "Local & Multi-Model Setup"
excerpt: "Configure per-agent models across providers (Ollama/OpenAI/HuggingFace)."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Why multi-model
SCANUE v22 can assign different models to different agents (“brain regions”) via `config/agents.yaml`.

## Example: Ollama (fully local)
```yaml
agents:
  DLPFC:
    models:
      primary:
        provider: "ollama"
        name: "llama3.2:latest"
        temperature: 0.1
        base_url: "http://localhost:11434"
```

## Example: OpenAI
```yaml
agents:
  MPFC:
    models:
      primary:
        provider: "openai"
        name: "gpt-4o-mini"
        temperature: 0.4
```

Then set `OPENAI_API_KEY` (or set `api_key:` on the model entry).

## Example: HuggingFace
```yaml
agents:
  ACC:
    models:
      primary:
        provider: "huggingface"
        name: "mistralai/Mistral-7B-Instruct-v0.2"
        temperature: 0.0
```

Then set `HUGGINGFACEHUB_API_TOKEN` (or set `api_key:` on the model entry).

## Legacy fallback
If an agent does not have a configured `primary` model, SCANUE falls back to the legacy env vars (e.g. `ACC_MODEL`) and uses the OpenAI provider by default.


