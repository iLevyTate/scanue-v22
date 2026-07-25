# **SCANUE-V22**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18023657.svg)](https://doi.org/10.5281/zenodo.18023657)

<img width="2752" height="1536" alt="SCANUE" src="https://github.com/user-attachments/assets/7b4ec095-8238-4b62-9dcb-1531dff5c8a9" />

## **Overview**
SCANUE v22 is a brain-inspired, **multi-agent** CLI that orchestrates specialized “PFC region” agents using **LangGraph**. It focuses on decomposing a task (DLPFC) and then conditionally invoking only the necessary specialist agents (VMPFC/OFC/ACC) before final integration (MPFC).

## **Name Change Notification**
This repository was previously referred to as SCANJS, a deprecated project by another developer. To reflect the enhancements introduced—such as human-in-the-loop (HITL) mechanisms and customized fine-tuned models—the project has been rebranded as SCANUE-V22.

For clarity:
- Instances of "SCANJS" in older documentation or code refer to pre-rebranding materials
- The current version reflects multiple iterations leading to this enhanced release

## **Cognitive Agents**
- **DLPFC Agent:** Task delegation and executive control
- **VMPFC Agent:** Emotional regulation and risk assessment
- **OFC Agent:** Reward processing and outcome evaluation
- **ACC Agent:** Conflict detection and error monitoring
- **MPFC Agent:** Value-based decision-making

## **Technical Requirements**
- **Python:** 3.11+
- **An LLM provider**: OpenAI, Ollama (local), or HuggingFace (endpoint/TGI)
- **Environment variables (only if needed by your provider)**:
  - OpenAI: `OPENAI_API_KEY`
  - HuggingFace: `HUGGINGFACEHUB_API_TOKEN`
  - Legacy fallback model names (optional): `DLPFC_MODEL`, `VMPFC_MODEL`, `OFC_MODEL`, `ACC_MODEL`, `MPFC_MODEL`

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iLevyTate/scanue-v22.git
   cd scanue-v22
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .          # or: pip install -e ".[dev]" to also get pytest/ruff/mypy
   ```
   `pip install -r requirements.txt` still works, but installs the dependencies
   without the package itself (so the `scanue` command is not created).

3. **(Optional) Set up environment variables** in a `.env` file (recommended)

4. **Run the application:**
   ```bash
   # Interactive mode (prompts for a task, then offers to collect feedback)
   python main.py

   # One-shot mode (runs a single task non-interactively and exits)
   python main.py "How should I structure my team's weekly meetings?"
   ```

## **Configuration**
The primary configuration is `config/agents.yaml` (a ready-to-edit copy is
provided at `config/agents.example.yaml`). Each agent can use a different
provider/model:

- **Ollama (local)**: set `provider: "ollama"` and (optionally) `base_url` (default is `http://localhost:11434`)
- **OpenAI**: set `provider: "openai"` and either set `OPENAI_API_KEY` or put `api_key:` in the YAML
- **HuggingFace**: set `provider: "huggingface"` and either set `HUGGINGFACEHUB_API_TOKEN` or put `api_key:` in the YAML

See `docs/local_models.md` for examples and recommendations.

## **Workflow**
1. User inputs a task or problem
2. **DLPFC Agent:** Breaks down the task and delegates which specialist agents are needed
3. Specialized agents run (only if delegated):
   - **VMPFC:** Emotional regulation
   - **OFC:** Reward processing
   - **ACC:** Conflict detection
4. **MPFC:** Integrates all prior insights into the final response
5. (Optional) User provides feedback (persisted to `feedback_history.json`)

## **Testing**
```bash
pip install -e ".[dev]"
pytest tests/       # 217 tests, fully offline — no provider, no API key
ruff check .
mypy main.py workflow.py agents utils scripts
```

CI runs all three on every push and pull request, across Python 3.11–3.13.

The suite proves the logic but never contacts a model. To validate against your
actual provider — schema compliance, token capture, context headroom — run:

```bash
python scripts/validate.py
```

It runs one real task in a temporary state directory (your `feedback_history.json`
and `logs/` are untouched) and prints a pass/fail report. Exit code 0 means every
hard check passed.

## **Partial results**
Specialists fail independently: if VMPFC cannot reach its model, the run
continues without it rather than aborting. When that happens the CLI says so
explicitly, and the session log records `degraded: true` alongside
`agent_errors`. A failed agent's output is **excluded** from MPFC's synthesis
and MPFC is told which perspective is missing, so a partial analysis is never
presented as a complete one.

## **Delegation**
DLPFC decides which specialists a task needs. It first asks the model for a
**schema-validated** decision (`with_structured_output`), which the provider
constrains during generation — nothing has to be parsed out of prose. If the
model or provider can't do that, it falls back to parsing the text reply.

Every run records how the decision was made, in `logs/session_*.json` under the
`task_delegation` stage:

| `delegation_source` | Meaning |
|---|---|
| `structured_output` | Schema-validated — the model stated its decision |
| `structured_text` | Parsed from `- VMPFC Agent: YES` lines |
| `semantic` | Inferred from keywords in the reply |
| `pattern` | Inferred from loose regex matches |
| `heuristic` | Nothing matched; task-complexity guess |
| `resilient_fallback` | DLPFC failed; safe default set was used |

Only `structured_output` reflects an explicit choice by the model — everything
else is inference, and a fallback also emits a `WARNING`. To check how your
models are behaving:

```bash
grep -h '"delegation_source"' logs/session_*.json | sort | uniq -c | sort -rn
```

A high fallback rate usually means the model is too small to follow the schema;
DLPFC drives all routing, so it benefits most from your strongest model.

## **What a run records**
Each run writes `logs/session_*.json` (kept to the 50 most recent) containing,
per stage: the resolved model and provider, the rendered prompt, token usage and
finish reason, duration, and any error — plus run-level totals and
`wall_clock_ms`. Token counts and the elapsed time are printed after each run.

A `finish_reason` of `length` means the response was cut off mid-generation;
those stages are named in the summary rather than passing as complete answers.

## **Troubleshooting**
Diagnostics are logged to stderr. The default level is `WARNING`; raise it to see
prompt construction, routing decisions, and provider traffic:

```bash
SCANUE_LOG_LEVEL=DEBUG python main.py "your task"
```

## **Architecture**

Key modules:

- `main.py`: CLI entrypoint, feedback persistence, session logging
- `workflow.py`: LangGraph workflow graph (stages + dynamic delegation)
- `agents/`: agent implementations (`base.py`, `dlpfc.py`, `specialized.py`) and the provider/model `factory.py`
- `utils/config.py`: YAML config loader with legacy env-var fallback
- `config/agents.yaml`: per-agent model/provider configuration
- `docs/local_models.md`: guide for Ollama / HuggingFace / OpenAI configuration
- `tests/`: pytest suite covering agents, workflow, HITL, and CLI
- `scripts/validate.py`: one-command validation against a real provider
- `feedback_history.json`: persistent Human-in-the-Loop (HITL) feedback (gitignored)
- `logs/`: per-run session logs (gitignored)

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## **Acknowledgments**
- Thanks to all contributors who have helped shape SCANUE-V22
- Special thanks to the cognitive science community for their research and insights
