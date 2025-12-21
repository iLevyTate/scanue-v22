---
permalink: /docs/troubleshooting/
title: "Troubleshooting"
excerpt: "Common setup and runtime issues."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## “OPENAI_API_KEY is required…”
This only happens if one of your configured models uses `provider: "openai"` and you did not provide `api_key:` in YAML and did not set `OPENAI_API_KEY`.

Fix:
- set `OPENAI_API_KEY`, or
- set `api_key:` for the OpenAI model entry in `config/agents.yaml`, or
- switch that model’s provider to `ollama`/`huggingface`.

## “HUGGINGFACEHUB_API_TOKEN is required…”
Same idea as above, but for `provider: "huggingface"`.

## Ollama connection errors
- Ensure Ollama is installed and running.
- Confirm `base_url` is correct (default: `http://localhost:11434`).
- Ensure the model name exists locally (e.g., `llama3.2:latest`).

## YAML config errors
- Validate indentation in `config/agents.yaml`.
- Ensure each model entry includes `provider` and `name`.


