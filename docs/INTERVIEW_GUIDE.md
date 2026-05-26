# SCANUE-V22 — Interview Reference Guide

A complete walkthrough of the project so you can confidently explain what it is,
how it works, the decisions behind it, and the hard problems you solved. Each
section pairs a plain-English explanation with the actual code so you can speak
to specifics.

---

## 1. Elevator Pitch (30 seconds)

> "SCANUE-V22 is a brain-inspired, multi-agent decision-making system. Instead of
> sending a prompt to one large model, it splits cognition across five specialized
> agents, each modeled on a region of the prefrontal cortex. A central
> 'executive' agent (DLPFC) reads the user's task, decides which specialists are
> actually needed, and dynamically routes the task through only those agents
> before a final integration step. It's built in Python on **LangGraph**, supports
> **OpenAI, Ollama (local), and HuggingFace** providers per-agent, and includes a
> human-in-the-loop feedback loop that persists across sessions."

**One-liner:** A dynamic, provider-agnostic multi-agent orchestration CLI with
cognitive-science-inspired routing, built on a LangGraph state machine.

---

## 2. The Core Idea (Why "Brain-Inspired"?)

The system maps five cognitive agents onto prefrontal cortex regions. This isn't
just naming — each region's real-world function dictates *when* that agent is
invoked and *how* its model is tuned (e.g., temperature).

| Agent | Brain Region | Cognitive Role | Invoked When | Temp |
|-------|-------------|----------------|--------------|------|
| **DLPFC** | Dorsolateral PFC | Executive control, task breakdown & delegation | **Always** (entry point) | 0.1 (deterministic) |
| **VMPFC** | Ventromedial PFC | Emotional regulation, risk, social/moral reasoning | Emotional/social tasks | 0.7 (nuance) |
| **OFC** | Orbitofrontal Cortex | Reward processing, cost/benefit, financial outcomes | Reward/financial tasks | 0.2 |
| **ACC** | Anterior Cingulate Cortex | Conflict detection, error monitoring | Tasks with competing options | 0.0 (strict logic) |
| **MPFC** | Medial PFC | Value-based decision-making, **integration** | **Always** (final step) | 0.4 |

**Talking point:** The temperature choices are deliberate and defensible — the
conflict-detection agent (ACC) runs at temperature `0.0` because you want strict,
repeatable logic for catching contradictions, while the emotional agent (VMPFC)
runs at `0.7` for nuance. This is configured in `config/agents.yaml`.

---

## 3. Architecture Overview

```
                 ┌─────────────────────────────────────────────┐
   User Task ───▶│  DLPFC (task_delegation)  — ALWAYS RUNS       │
                 │  • Breaks task down                           │
                 │  • Decides which specialists are needed       │
                 └───────────────────┬──────────────────────────┘
                                     │ parse_agent_assignments()
                                     │ → ["emotional_regulation",
                                     │     "conflict_detection",
                                     │     "value_assessment"]
                                     ▼
        ┌───────────── Dynamic conditional routing (LangGraph) ──────────────┐
        │                                                                     │
        ▼                ▼                  ▼                   ▼              │
   VMPFC (emotion)   OFC (reward)    ACC (conflict)     MPFC (value/integrate)│
        │                │                  │                   │             │
        └────────────────┴──────────────────┴───────────────────┘             │
                                     │  (only delegated agents run, in order)  │
                                     ▼                                         │
                            Final integrated response ◀──────────────────────┘
                                     │
                                     ▼
                    Optional Human-in-the-Loop feedback
                       (persisted to feedback_history.json)
```

### Key files

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point, session logging, HITL feedback persistence, credential validation |
| `workflow.py` | The LangGraph state machine: nodes, conditional edges, delegation parsing, per-stage logging |
| `agents/base.py` | Abstract `BaseAgent` — shared LLM invocation, timeout handling, response formatting |
| `agents/dlpfc.py` | The executive/delegation agent + subtask parsing |
| `agents/specialized.py` | The four specialist agents (VMPFC, OFC, ACC, MPFC) |
| `agents/factory.py` | `LLMFactory` — provider abstraction (OpenAI / Ollama / HuggingFace) |
| `utils/config.py` | `ConfigLoader` — YAML config with env-var fallback |
| `config/agents.yaml` | Per-agent model/provider/temperature configuration |
| `tests/` | ~2,000 lines of pytest covering agents, workflow, HITL, CLI |

---

## 4. Technical Deep Dives (with code)

### 4.1 The LangGraph State Machine

The whole orchestration is a `StateGraph` over a typed state dict. The clever part
is that **every node uses the same conditional-edge function** (`get_next_stage`),
which decides the next hop based on *progress*, not hard-coded transitions.

```python
# workflow.py
class AgentState(TypedDict, total=False):
    task: str
    stage: str
    response: str
    subtasks: list
    feedback: str
    feedback_history: list
    session_log: dict
    error: bool
    delegated_agents: list   # which agents to run, decided by DLPFC
    agent_responses: dict    # collected outputs — also used as a progress counter
    agent_errors: dict       # per-agent failures (resilience)

def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("task_delegation", process_task_delegation)
    workflow.add_node("emotional_regulation", process_emotional_regulation)
    workflow.add_node("reward_processing", process_reward_processing)
    workflow.add_node("conflict_detection", process_conflict_detection)
    workflow.add_node("value_assessment", process_value_assessment)

    def get_next_stage(state: Dict[str, Any]) -> str:
        delegated_agents = state.get("delegated_agents", [])
        if not delegated_agents:
            return END

        # Progress tracking: count completed agents vs. total delegated
        completed_count = len(state.get("agent_responses", {}))
        if completed_count < len(delegated_agents):
            return delegated_agents[completed_count]   # next agent in order
        return END

    # Universal edge mapping: any node can route to any other node
    all_stages = ["task_delegation", "emotional_regulation",
                  "reward_processing", "conflict_detection", "value_assessment"]
    comprehensive_mappings = {END: END, **{s: s for s in all_stages}}

    for stage in all_stages:
        workflow.add_conditional_edges(stage, get_next_stage, comprehensive_mappings)

    workflow.set_entry_point("task_delegation")
    return workflow.compile()
```

**Talking point — why progress-based routing?** My first version manually set
`state["stage"]` inside each node to point at the next one. That caused two bugs:
agents getting **skipped** and **infinite loops**. The fix was to make routing a
pure function of *how many agents have completed* (`len(agent_responses)`) versus
*how many were delegated*. This is idempotent and can't loop — once the count
reaches the list length, it returns `END`. (See §5 for the war story.)

---

### 4.2 Dynamic Delegation — `parse_agent_assignments`

The DLPFC agent is *told* to emit a structured `YES/NO` block per specialist. But
LLMs don't always comply, so I built a **3-tier parsing strategy with intelligent
fallback** to make routing robust.

```python
# workflow.py (condensed)
def parse_agent_assignments(dlpfc_response: str) -> list:
    agent_map = {
        'VMPFC': 'emotional_regulation',
        'OFC':   'reward_processing',
        'ACC':   'conflict_detection',
        'MPFC':  'value_assessment',
    }
    response_lower = dlpfc_response.lower()
    agent_assignments = []

    # STRATEGY 1: Structured format — "- VMPFC Agent: YES"
    structured_found = False
    for name, stage in agent_map.items():
        if any(re.search(p, response_lower) for p in [
            rf"- {name.lower()} agent:\s*yes",
            rf"{name.lower()} agent:\s*yes",
            rf"- {name.lower()}:\s*yes",
        ]):
            agent_assignments.append(stage); structured_found = True

    # STRATEGY 2: Semantic keyword analysis (only if structure missing)
    if not structured_found:
        semantic = {
            'VMPFC': ['emotional', 'feeling', 'social', 'moral', 'risk', 'empathy'],
            'OFC':   ['reward', 'benefit', 'cost', 'outcome', 'financial', 'value'],
            'ACC':   ['conflict', 'error', 'competing', 'contradiction', 'monitor'],
        }
        for name, keywords in semantic.items():
            if any(k in response_lower for k in keywords):
                agent_assignments.append(agent_map[name])

    # STRATEGY 3: Loose pattern matching (final text-based fallback)
    # ... regex like "assign.*vmpfc", "delegate.*ofc" ...

    # INTELLIGENT FALLBACK: nothing matched → infer from task complexity
    if not agent_assignments:
        if is_complex:   agent_assignments = [all four stages]
        elif emotional:  agent_assignments = ['emotional_regulation', 'value_assessment']
        elif decision:   agent_assignments = ['reward_processing', 'value_assessment']
        else:            agent_assignments = ['value_assessment']  # simple → MPFC only

    # INVARIANT: MPFC (value_assessment) always runs last for integration
    if 'value_assessment' not in agent_assignments:
        agent_assignments.append('value_assessment')

    return agent_assignments
```

**Talking point:** This is defensive engineering against non-deterministic LLM
output. Rather than trusting the model to always produce clean structured output,
I degrade gracefully: structured → semantic → loose patterns → complexity
heuristic. And there's a hard invariant: **MPFC always runs last** because it's
the integration step. This guarantees the workflow always produces a coherent
final answer no matter how the upstream parsing goes.

---

### 4.3 The Provider Abstraction — `LLMFactory`

To support local *and* cloud models without touching agent code, all model
instantiation goes through one factory. Agents never know which provider they're
talking to.

```python
# agents/factory.py
class LLMFactory:
    @staticmethod
    def create_llm(config: Dict[str, Any]) -> Any:
        provider = config.get("provider", "openai").lower()
        model_name = config.get("name")
        temperature = config.get("temperature", 0.7)

        if provider == "openai":
            return ChatOpenAI(
                model=model_name, temperature=temperature,
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                timeout=config.get("timeout", 30.0),
                max_retries=config.get("max_retries", 3),
            )
        elif provider == "ollama":
            return ChatOllama(
                model=model_name, temperature=temperature,
                base_url=config.get("base_url", "http://localhost:11434"),
                timeout=config.get("timeout", 120.0),  # local models are slower
            )
        elif provider == "huggingface":
            llm = HuggingFaceEndpoint(
                repo_id=model_name, temperature=temperature,
                huggingfacehub_api_token=config.get("api_key")
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                timeout=config.get("timeout", 120.0),
            )
            return ChatHuggingFace(llm=llm)
        raise ValueError(f"Unsupported LLM provider: {provider}")
```

**Talking point:** This is the classic **Factory pattern** for dependency
inversion. Because every provider returns a LangChain `ChatModel` with a common
`.ainvoke()` interface, the agents are completely decoupled from the provider.
You can run DLPFC on cloud GPT-4 for reasoning quality while running ACC on a tiny
local `tinyllama:1.1b` for cheap strict logic — all by editing YAML, no code
changes. The higher timeout for local providers reflects that local inference is
slower.

---

### 4.4 Configuration with Graceful Fallback — `ConfigLoader`

```python
# utils/config.py
class ConfigLoader:
    _config = None
    _config_path = Path("config/agents.yaml")

    @classmethod
    def load_config(cls):
        if cls._config is None:  # cached after first load
            if cls._config_path.exists():
                with open(cls._config_path) as f:
                    cls._config = yaml.safe_load(f)
            else:
                cls._config = {"agents": {}}
        return cls._config

    @classmethod
    def get_model_config(cls, agent_name, model_type="primary", env_var_fallback=None):
        model_config = cls.get_agent_config(agent_name).get("models", {}).get(model_type)
        if model_config:
            return model_config
        # Legacy env-var fallback (e.g., DLPFC_MODEL) → defaults to OpenAI
        if env_var_fallback and os.getenv(env_var_fallback):
            return {"provider": "openai", "name": os.getenv(env_var_fallback), "temperature": 0.7}
        # Final safety net
        return {"provider": "openai", "name": "gpt-3.5-turbo", "temperature": 0.7}
```

```yaml
# config/agents.yaml (excerpt)
agents:
  DLPFC:
    description: "Central controller: task breakdown, delegation, strategy"
    models:
      primary: { provider: "ollama", name: "llama3.2:latest", temperature: 0.1, base_url: "http://localhost:11434" }
      fast:    { provider: "ollama", name: "tinyllama:1.1b",  temperature: 0.0, base_url: "http://localhost:11434" }
  ACC:
    models:
      primary: { provider: "ollama", name: "tinyllama:1.1b", temperature: 0.0 }  # strict logic
```

**Talking point:** Three-layer fallback — YAML config → legacy env var → hard
default — means the app never hard-crashes on missing config. The `_config` class
variable is a simple **memoization cache** so the YAML is parsed once. There's
also a `fast` model slot wired into the schema for future cost-optimization (use a
small model for simple tasks).

---

### 4.5 The Agent Base Class (Template Method pattern)

```python
# agents/base.py
class BaseAgent(ABC):
    def __init__(self, agent_name, model_env_key=None):
        self.agent_name = agent_name
        self.models = {}
        # Build every configured model for this agent via the factory
        for model_type, config in ConfigLoader.get_agent_config(agent_name).get("models", {}).items():
            self.models[model_type] = LLMFactory.create_llm(config)
        # Guarantee a 'primary' model exists, with legacy fallback
        if "primary" not in self.models:
            fallback = ConfigLoader.get_model_config(agent_name, "primary", env_var_fallback=model_env_key)
            self.models["primary"] = LLMFactory.create_llm(fallback)
        self.llm = self.models["primary"]          # back-compat alias
        self.prompt = self._create_prompt()         # subclass-defined
        self.last_raw_response = None               # cached for logging

    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate: ...   # each agent's persona

    async def _process_with_timeout(self, state):
        messages = self.prompt.format_messages(
            task=state.get("task", ""), state=state,
            previous_response=state.get("previous_response", "No previous response"),
            feedback=state.get("feedback", "No feedback provided"),
            feedback_history=state.get("feedback_history", []),
        )
        response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=30.0)
        # capture raw response + metadata for the session log
        self.last_raw_response = {"model": ..., "prompt": ..., "response": response.content}
        return self._format_response(response.content)
```

**Talking point:** This is the **Template Method pattern**. `BaseAgent` owns the
shared mechanics — model construction, prompt formatting, 30s timeout, structured
response shape, raw-response capture for logging — while each subclass only
implements `_create_prompt()` to define its cognitive persona. Adding a new brain
region is ~10 lines (see VMPFC below). Note the use of `getattr` with multiple
fallbacks for model metadata, because OpenAI/Ollama/HF expose different attribute
names — defensive coding for the multi-provider reality.

```python
# agents/specialized.py — a whole specialist is this small
class VMPFCAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="VMPFC", model_env_key="VMPFC_MODEL")
    def _create_prompt(self):
        return ChatPromptTemplate.from_template(
            """You are the VMPFC Agent, responsible for emotional regulation and risk assessment.
            Task: {task}  ...  Analyze the emotional and risk components of the task.""")
    async def process(self, state):
        return await super().process(state)
```

---

### 4.6 Human-in-the-Loop (HITL) with Cross-Session Persistence

Feedback is collected after each interactive run, persisted to JSON, and **fed
back into agent prompts** on future runs so the system carries context forward.

```python
# main.py — collect + persist
new_feedback = {
    "response": response_content,
    "feedback": feedback,
    "stage": result.get("stage", "unknown"),
}
feedback_history.append(new_feedback)
save_feedback_history(feedback_history)   # → feedback_history.json

# main.py — load + inject into initial state on the next run
feedback_history = load_feedback_history()
state = { "task": task, "stage": "task_delegation",
          "feedback_history": feedback_history.copy(), ... }
```

```python
# agents/dlpfc.py — history is formatted into the prompt
def _format_feedback_history(self, history):
    if not history:
        return "No previous feedback"
    return "\n".join(
        f"Stage: {e.get('stage','unknown')}\nResponse: {e.get('response','')}\n"
        f"Feedback: {e.get('feedback','')}\n" for e in history)
```

**Talking point:** "Human-in-the-loop" here means two things: (1) the user can
steer/correct output via feedback, and (2) that feedback **persists across
sessions** in `feedback_history.json` and is re-injected into the DLPFC prompt, so
the executive agent's delegation decisions are informed by past user preferences.
It's a lightweight, file-based form of long-term memory — no vector DB needed for
the scale this targets.

---

### 4.7 Observability — Structured Session Logging

Every run produces a structured JSON log capturing each stage's input, output,
raw LLM response, timing, and errors.

```python
# main.py
def create_session_log(task):
    return {"task": task, "timestamp": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4()), "stages": [],
            "final_response": None, "user_feedback": None,
            "error": None, "completed": False}

# workflow.py — each node times itself
def log_stage_end(stage_log, result, error=None):
    stage_log["end_time"] = datetime.now().isoformat()
    start = datetime.fromisoformat(stage_log["start_time"])
    stage_log["duration_ms"] = int((datetime.fromisoformat(stage_log["end_time"]) - start).total_seconds() * 1000)
    stage_log["output"] = copy.deepcopy(result.get("response", {}))
    return stage_log
```

**Talking point:** Logs land in `logs/session_<timestamp>_<id>.json` with per-stage
`duration_ms`, raw prompts, and raw responses. This was essential for debugging the
routing bugs — I could see exactly which agents fired, in what order, and how the
DLPFC output was parsed. It doubles as an audit trail for a decision-making system.

---

## 5. The Hard Problems I Solved (War Stories)

These are the strongest interview material — concrete bugs with concrete fixes.

### 5.1 Infinite Loops & Skipped Agents
- **Symptom:** Agents were either being skipped entirely or the graph looped
  forever.
- **Root cause:** Each node was manually mutating `state["stage"]` to name its
  successor. With LangGraph's conditional edges *also* reading state, the two
  mechanisms fought each other — stale `stage` values caused re-entry and skips.
- **Fix:** Stop setting `stage` manually. Make routing a **pure function of
  progress**: `next = delegated_agents[len(agent_responses)]`, return `END` once
  the count hits the list length. Provable termination.
- **Code evidence:** Comments like *"CRITICAL FIX: This replaces the previous
  flawed approach that caused agent skipping and infinite loops by manually
  setting state['stage']"* in `get_next_stage`, and the bug where MPFC wasn't
  recording its response (*"This was missing and caused infinite loops"*) in
  `process_value_assessment`.

### 5.2 Resilience — One Agent Fails, the Workflow Survives
- **Goal:** A single specialist erroring (timeout, model hiccup) shouldn't kill the
  whole pipeline.
- **Fix:** Distinguish **individual agent failures** (recorded in `agent_errors`,
  workflow continues) from **critical system errors** (`error=True`, stop). On a
  DLPFC failure, fall back to a sensible default delegation rather than aborting.

```python
# workflow.py — DLPFC failure still yields a working pipeline
correct_delegated_agents = ["emotional_regulation", "conflict_detection", "value_assessment"]
return {**state,
        "agent_errors": {**state.get("agent_errors", {}), "DLPFC": str(e)},
        "response": error_response,
        "delegated_agents": correct_delegated_agents}
```
- **Test evidence:** `test_error_handling`, `test_timeout_handling`,
  `test_cancellation_handling` all assert the workflow *continues* and records the
  error in `agent_errors` rather than crashing.

### 5.3 Non-Deterministic LLM Output Breaking Routing
- **Problem:** Routing depended on the DLPFC model emitting a clean structured
  block — which LLMs don't reliably do.
- **Fix:** The 3-tier + heuristic parser in §4.2, with the hard "MPFC always runs"
  invariant.

### 5.4 Local-Only Setups Shouldn't Require Cloud Keys
- **Problem:** The app originally demanded `OPENAI_API_KEY` even for fully-local
  Ollama runs.
- **Fix:** Provider-aware credential validation at startup — only require a key for
  agents actually configured to use that provider.

```python
# main.py
for agent_name, agent_cfg in agents_cfg.items():
    for model_type, model_cfg in agent_cfg.get("models", {}).items():
        provider = model_cfg.get("provider", "openai").lower()
        if provider == "openai" and not (model_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")):
            openai_models_need_key.append(f"{agent_name}.{model_type}")
```

### 5.5 Cross-Platform Unicode (Windows)
- The CLI uses emoji-rich output. On Windows consoles defaulting to cp1252 this
  crashed. Fix: reconfigure stdout/stderr to UTF-8, wrapped in try/except so it
  never fails the app.

### 5.6 Non-Interactive Mode
- When run with CLI args (`python main.py "my task"`), the app must **never block
  on stdin** for feedback. Guarded by an `interactive` flag so it can be scripted/CI'd.

---

## 6. Testing Strategy

~2,000 lines of pytest across 11 files. Key patterns:

- **Async testing** via `pytest-asyncio` (`asyncio_mode = auto` in `pytest.ini`).
- **Dependency injection through mocking** — `ConfigLoader.load_config` and
  `ChatOpenAI` are patched so tests never hit a real LLM or need real keys.
- **State-machine tests** — `test_workflow_state_transitions` mocks every agent's
  `process()` and asserts the graph runs all delegated agents in order and
  terminates (MPFC present in `agent_responses`).
- **Failure-mode tests** — explicit tests for timeout, cancellation, and generic
  errors that assert *resilience* (workflow continues, error captured).
- **HITL tests** — verify feedback accumulates, persists, and updates
  `previous_response`.

```python
# tests/test_workflow.py — mocking out the LLM layer entirely
test_config = {"agents": {"DLPFC": {"models": {"primary": {"provider": "openai", "name": "test-model"}}}, ...}}
with patch('utils.config.ConfigLoader.load_config', return_value=test_config), \
     patch('agents.factory.ChatOpenAI', return_value=mock_chat_openai):
    yield
```

**Talking point:** Because I dependency-inverted the LLM behind a factory and the
config behind a loader, the entire system is testable without network access or
API keys — I just patch those two seams. That's the practical payoff of the
abstraction layers.

---

## 7. Design Decisions & Trade-offs (be ready to defend)

| Decision | Why | Trade-off / What I'd revisit |
|----------|-----|------------------------------|
| Progress-based routing (`len(agent_responses)`) | Guarantees termination, fixed the loop/skip bugs | Assumes strictly sequential execution; parallel specialists would need a different counter |
| Sequential, not parallel, specialists | Simpler state, easier to reason about, MPFC integrates last | Slower wall-clock; independent agents (VMPFC/OFC/ACC) *could* run concurrently |
| File-based feedback (`feedback_history.json`) | Zero infra, easy to inspect | Doesn't scale to many users; no semantic retrieval — a vector store would be the upgrade |
| Factory + YAML config | Per-agent provider/model flexibility, fully testable | More indirection than a single hard-coded client |
| Regex/heuristic delegation parsing | Robust to messy LLM output, no extra LLM call | Brittle to wholly unexpected phrasing; an LLM-based "router" with function-calling would be cleaner |
| Heavy `print()` debug output | Great for a CLI demo & live debugging | Should be a real logging framework with levels for production |

---

## 8. Likely Interview Questions & Crisp Answers

**Q: Why multiple agents instead of one big prompt?**
Separation of concerns and tunability. Each agent has a focused persona and its
own temperature (strict logic at 0.0 for conflict detection, nuance at 0.7 for
emotion). The DLPFC only invokes specialists the task actually needs, which saves
tokens/latency on simple tasks. It also makes the reasoning auditable — you can
see each cognitive "lens" separately in the session log.

**Q: How does it decide which agents to run?**
The DLPFC agent emits a structured delegation block; `parse_agent_assignments`
parses it with three fallback tiers (structured → semantic keywords → loose
patterns), then a complexity heuristic if all else fails. MPFC always runs last to
integrate.

**Q: What happens if an agent fails or times out?**
Individual failures are caught, recorded in `agent_errors`, and the workflow
continues — only critical/system errors stop it. There's a 30s timeout per agent
and a default delegation fallback if the DLPFC itself fails. There are dedicated
tests proving the graph survives timeouts and cancellations.

**Q: How did you avoid infinite loops in the graph?**
Routing is a pure function of progress: next agent = `delegated_agents[len(agent_
responses)]`, and it returns `END` once completed count equals delegated count.
Because each node appends exactly one entry to `agent_responses`, termination is
guaranteed. The earlier bug came from manually mutating `state["stage"]`, which I
removed.

**Q: How would you scale this?**
Run the independent specialists (VMPFC/OFC/ACC) in parallel and only join before
MPFC; replace file-based feedback with a vector store for semantic recall; swap
`print` for structured logging/metrics; add a proper router using model
function-calling instead of regex; and add caching for repeated subtasks.

**Q: How is it testable without hitting an LLM?**
Two seams: the `LLMFactory` and `ConfigLoader`. Patch both in tests and inject a
mock model — no keys, no network. That's the payoff of the dependency inversion.

**Q: Why LangGraph over LangChain agents or hand-rolled orchestration?**
LangGraph gives a first-class state machine with conditional edges, which maps
perfectly onto "dynamic routing through a subset of nodes." It handles the graph
execution and state merging; I just define nodes and the routing function.

---

## 9. Tech Stack Summary

- **Language:** Python 3.8+ (async/await throughout)
- **Orchestration:** LangGraph (`StateGraph`, conditional edges)
- **LLM layer:** LangChain Core, `langchain-openai`, `langchain-ollama`,
  `langchain-huggingface`
- **Providers:** OpenAI (cloud), Ollama (local), HuggingFace (endpoint/TGI)
- **Config:** PyYAML, python-dotenv
- **Testing:** pytest, pytest-asyncio, pytest-mock
- **Patterns:** Factory, Template Method, State Machine, Strategy (parsing
  fallbacks), Memoized config loading

---

## 10. 60-Second Closing Summary

> "SCANUE-V22 takes a cognitive-science metaphor and turns it into a working
> dynamic multi-agent system. The interesting engineering isn't the brain naming —
> it's the LangGraph state machine with progress-based routing that provably
> terminates, the three-tier delegation parser that's robust to messy LLM output,
> a factory-based provider abstraction that lets any agent run on cloud or local
> models from a YAML file, and a resilience model where one agent failing never
> takes down the pipeline. It's fully testable without an API key because the LLM
> and config layers are dependency-inverted, and it has cross-session human
> feedback and structured per-stage logging for observability."
