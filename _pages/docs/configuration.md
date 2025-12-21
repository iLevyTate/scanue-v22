---
permalink: /docs/configuration/
title: "Configuration"
excerpt: "Configure models per agent using config/agents.yaml."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Primary configuration: `config/agents.yaml`
Each agent can have one or more model entries (typically `primary`, optionally `fast`, etc.). Example:

```yaml
agents:
  DLPFC:
    description: "Executive controller"
    models:
      primary:
        provider: "ollama"
        name: "llama3.2:latest"
        temperature: 0.1
        base_url: "http://localhost:11434"
```

## Supported providers
- **openai**
  - `api_key:` in YAML, or set `OPENAI_API_KEY`
- **ollama**
  - local provider; configure `base_url` if needed
- **huggingface**
  - `api_key:` in YAML, or set `HUGGINGFACEHUB_API_TOKEN`

## Legacy fallback (optional)
If an agent is missing a `primary` model config, SCANUE falls back to legacy env vars:
- `DLPFC_MODEL`, `VMPFC_MODEL`, `OFC_MODEL`, `ACC_MODEL`, `MPFC_MODEL`

If those are set, the legacy fallback uses the **OpenAI** provider.


