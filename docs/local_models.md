---
permalink: /docs/local_models/
title: "Local & Multi-Model Configuration"
excerpt: "How to configure SCANUE-V22 to use local models (Ollama, HuggingFace) and specialized models for each brain region."
last_modified_at: 2025-05-20T10:00:00-05:00
toc: true
---

# Local & Multi-Model Configuration

SCANUE-V22 supports a flexible multi-model architecture that allows you to assign specific AI models to different "brain regions" (agents). This includes support for local inference using Ollama and HuggingFace, as well as cloud providers like OpenAI.

## Configuration File

The primary configuration is located at `config/agents.yaml`. This file defines which models are used by which agents.

### Basic Structure

```yaml
agents:
  # Agent Name (DLPFC, VMPFC, OFC, ACC, MPFC)
  DLPFC:
    description: "Executive Function"
    models:
      # Primary model used for main processing
      primary:
        provider: "ollama"       # valid: openai, ollama, huggingface
        name: "llama3:70b"       # model identifier
        temperature: 0.1         # creativity level (0.0 - 1.0)
        base_url: "http://localhost:11434" # optional for local
      
      # Optional specialized models (future proofing)
      fast:
        provider: "ollama"
        name: "llama3:8b"
        temperature: 0.0
```

## Supported Providers

### 1. OpenAI (Cloud)
Standard integration with OpenAI API.
- **Provider**: `openai`
- **Requirements**: `OPENAI_API_KEY` environment variable.
- **Config**:
  ```yaml
  primary:
    provider: "openai"
    name: "gpt-4o"
  ```

### 2. Ollama (Local)
Run open-source models locally.
- **Provider**: `ollama`
- **Requirements**: [Ollama](https://ollama.com/) installed and running.
- **Config**:
  ```yaml
  primary:
    provider: "ollama"
    name: "llama3"
    base_url: "http://localhost:11434" # Default
  ```

### 3. HuggingFace (Local/Cloud)
Use HuggingFace Inference Endpoints or local TGI.
- **Provider**: `huggingface`
- **Requirements**: `HUGGINGFACEHUB_API_TOKEN` environment variable.
- **Config**:
  ```yaml
  primary:
    provider: "huggingface"
    name: "mistralai/Mistral-7B-Instruct-v0.2"
  ```

## Brain Region Recommendations

Each agent mimics a specific cognitive function. Here are recommended model types for each:

| Agent | Function | Recommended Model Characteristics |
|-------|----------|-----------------------------------|
| **DLPFC** | Executive Control, Logic | High reasoning capability (e.g., GPT-4, Llama-3-70b) |
| **VMPFC** | Emotional Regulation | Emotional intelligence, uncensored or social-tuned models |
| **OFC** | Reward Processing | Analytical, math-capable models |
| **ACC** | Conflict Monitoring | Strict, logical models with low temperature |
| **MPFC** | Integration | High context window models to synthesize all inputs |

## Backward Compatibility

If `config/agents.yaml` is missing or incomplete, the system falls back to the legacy environment variables:
- `DLPFC_MODEL`
- `VMPFC_MODEL`
- `OFC_MODEL`
- `ACC_MODEL`
- `MPFC_MODEL`

These will default to using the OpenAI provider.

## Credential requirements
- If you use any `provider: "openai"` models, you must set `OPENAI_API_KEY` (or set `api_key:` in the YAML entry).
- If you use any `provider: "huggingface"` models, you must set `HUGGINGFACEHUB_API_TOKEN` (or set `api_key:` in the YAML entry).
- If you use only `provider: "ollama"` models, you do not need either credential.

