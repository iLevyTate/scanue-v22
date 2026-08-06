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
        timeout: 120             # seconds for one LLM call (default: 30)
        max_retries: 3           # retries after the first attempt (0 disables)
        num_ctx: 8192            # ollama only: context window in tokens
        max_tokens: 1024         # cap on output length (ollama: num_predict)

      # Optional specialized models (future proofing)
      fast:
        provider: "ollama"
        name: "llama3:8b"
        temperature: 0.0
```

### Options that matter most for local models

- **`num_ctx` — set it explicitly.** Left unset, the Ollama server applies its
  own default (commonly 2048 tokens) and **silently drops** anything past it —
  no error, no log line, just a truncated prompt. Match it to your model's real
  capability. SCANUE logs each prompt's approximate token count (visible with
  `SCANUE_LOG_LEVEL=DEBUG`) so you can see how close you are.
- **`timeout`** — local models on CPU can be slow; the default 30 s per call is
  tuned for cloud APIs. 120 s is a sensible floor for CPU inference.
- **`max_retries`** — retries with exponential backoff apply to every provider,
  so a transient blip on a local server no longer fails the whole stage.
- **`max_tokens`** — if a response hits this cap, the run summary flags the
  stage (`finish_reason: length`) instead of passing a cut-off answer as
  complete.

### Verifying your setup

After configuring, run one real task with pass/fail checks — schema compliance,
token capture, context headroom:

```bash
python scripts/validate.py
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

## Small models and delegation

DLPFC decides which specialists run, and it is asked for a **schema-validated**
decision (`with_structured_output`). Small local models sometimes cannot honor
the schema, in which case SCANUE falls back to parsing the text reply and
records how the decision was made in `logs/session_*.json` as
`delegation_source`. If your runs show anything other than `structured_output`
there, routing is being inferred rather than stated — a larger DLPFC model
usually fixes it. See the README's Delegation section for the full table.

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

