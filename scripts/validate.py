#!/usr/bin/env python3
"""Validate SCANUE against a REAL provider.

The test suite runs entirely offline against stubs, which leaves one dimension
unverified: how the app behaves with an actual model attached. This script runs
one real task and checks the things stubs cannot prove --

  * whether the configured model honors the delegation schema, or falls back to
    inferring routing from text
  * whether token usage is actually extracted from real provider metadata
    (the shapes differ between Ollama and OpenAI, and a mismatch silently
    reports zero rather than erroring)
  * whether specialists actually execute, and whether any of them failed
  * whether any response was truncated mid-generation
  * how much context headroom the largest prompt leaves

It writes to a temporary state directory, so your real feedback_history.json
and logs/ are untouched.

Usage:
    python scripts/validate.py
    python scripts/validate.py --task "Should I rewrite this service in Go?"

Exit code is 0 when every hard check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "Should I take a lower-paying job with better hours?"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    status: str
    name: str
    detail: str
    hint: str = ""


def analyze(session_log: dict[str, Any]) -> list[Check]:
    """Turn a session log into pass/warn/fail checks.

    Pure so it can be unit-tested without a provider.
    """
    checks: list[Check] = []
    stages = session_log.get("stages") or []
    by_stage = {s.get("stage"): s for s in stages}
    summary = session_log.get("summary") or {}

    # --- The run completed at all ---------------------------------------- #
    if not stages:
        return [Check(FAIL, "workflow ran", "no stages recorded",
                      "The run failed before any agent executed. Check the provider is reachable.")]

    # Did ANY agent actually reach its model? If not, every downstream check is
    # measuring the absence of a provider rather than a defect -- reporting
    # "0 tokens: extract_usage is broken" here would send you debugging the
    # wrong thing entirely.
    agent_errors = session_log.get("agent_errors") or {}
    stage_agents = {s.get("agent") for s in stages if s.get("agent")}
    no_call_succeeded = bool(stage_agents) and stage_agents <= set(agent_errors)

    if no_call_succeeded:
        sample = next(iter(agent_errors.values()), "")
        checks.append(Check(
            FAIL, "provider reachable", f"every agent failed -- {sample[:70]}",
            "Nothing reached a model, so the remaining checks cannot be evaluated. "
            "For Ollama, confirm the server is running (`ollama list`) and that the "
            "models in config/agents.yaml are pulled",
        ))
        for name in ("token capture", "output truncation", "context headroom", "delegation source"):
            checks.append(Check(SKIP, name, "not evaluated -- no successful call"))
        checks.append(Check(
            PASS, "degradation handling",
            "the run continued and was reported as failed rather than crashing",
        ))
        return checks

    checks.append(Check(
        PASS if session_log.get("completed") else FAIL,
        "run completed",
        f"{len(stages)} stages, error={session_log.get('error')!r}",
        "" if session_log.get("completed") else "The final synthesis stage failed.",
    ))

    # --- Delegation: schema-validated, or inferred? ----------------------- #
    delegation = by_stage.get("task_delegation") or {}
    source = delegation.get("delegation_source")
    delegated = delegation.get("delegated_agents") or []
    if source == "structured_output":
        checks.append(Check(PASS, "delegation source", f"{source} -> {delegated}"))
    elif source:
        checks.append(Check(
            WARN, "delegation source", f"{source} -> {delegated}",
            "The model did not honor the delegation schema, so routing was inferred "
            "from its prose. This is the designed fallback, not a break -- but routing "
            "quality now depends on keyword matching. A larger DLPFC model usually fixes it.",
        ))
    else:
        checks.append(Check(FAIL, "delegation source", "not recorded",
                            "Expected delegation_source on the task_delegation stage."))

    # --- Specialists actually executed ------------------------------------ #
    specialists = [
        s.get("stage") for s in stages
        if s.get("stage") not in ("task_delegation", "value_assessment")
        and s.get("agent") not in agent_errors
    ]
    if specialists:
        checks.append(Check(PASS, "specialists ran", ", ".join(sorted(specialists))))
    else:
        checks.append(Check(
            WARN, "specialists ran", "none -- MPFC only",
            "Valid if the task genuinely needed no specialist, but if this happens for "
            "every task the delegation signal is being lost.",
        ))

    # --- Per-agent failures ----------------------------------------------- #
    if agent_errors:
        detail = "; ".join(f"{k}: {v[:60]}" for k, v in sorted(agent_errors.items()))
        checks.append(Check(WARN, "agent failures", detail,
                            "The run continued without these agents; the answer is partial."))
    else:
        checks.append(Check(PASS, "agent failures", "none"))

    # --- Token capture ----------------------------------------------------- #
    # This is the check stubs cannot make: usage_metadata / response_metadata
    # shapes differ per provider, and a mismatch silently yields zero.
    tokens = (summary.get("tokens") or {}).get("total_tokens", 0)
    if tokens:
        checks.append(Check(PASS, "token capture", f"{tokens:,} tokens across the run"))
    else:
        checks.append(Check(
            FAIL, "token capture", "0 tokens recorded",
            "extract_usage() found no usage metadata on the responses. Inspect "
            "raw_llm_response.usage in the log and compare against what your provider "
            "returns -- the keys likely differ from usage_metadata/response_metadata.",
        ))

    # --- Truncation --------------------------------------------------------- #
    truncated = summary.get("truncated_stages") or []
    if truncated:
        checks.append(Check(
            WARN, "output truncation", f"cut off mid-generation: {', '.join(truncated)}",
            "Raise max_tokens for those agents, or their answers are incomplete.",
        ))
    else:
        checks.append(Check(PASS, "output truncation", "none"))

    # --- Context headroom ---------------------------------------------------- #
    worst = None
    for stage in stages:
        raw = stage.get("raw_llm_response") or {}
        chars = raw.get("prompt_chars")
        num_ctx = (raw.get("metadata") or {}).get("num_ctx")
        if not chars or not num_ctx:
            continue
        used = (chars / 4) / num_ctx
        if worst is None or used > worst[0]:
            worst = (used, stage.get("stage"), int(chars / 4), num_ctx)
    if worst:
        used, name, tok, ctx = worst
        status = FAIL if used >= 1 else (WARN if used > 0.8 else PASS)
        checks.append(Check(
            status, "context headroom",
            f"worst: {name} at ~{tok:,}/{ctx:,} tokens ({used:.0%} used)",
            "Ollama silently DROPS anything past num_ctx. Raise it for that agent."
            if status != PASS else "",
        ))

    # --- Model attribution ---------------------------------------------------- #
    unattributed = [s.get("stage") for s in stages if not s.get("model")]
    if unattributed:
        checks.append(Check(FAIL, "model attribution", f"missing on: {', '.join(unattributed)}",
                            "Every stage should record which model it used."))
    else:
        models = {f"{s.get('agent')}={s.get('model')}" for s in stages}
        checks.append(Check(PASS, "model attribution", ", ".join(sorted(models))))

    return checks


def _load_latest_log(state_dir: Path) -> dict[str, Any] | None:
    logs = sorted(glob.glob(str(state_dir / "logs" / "session_*.json")))
    if not logs:
        return None
    with open(logs[-1], encoding="utf-8") as f:
        return json.load(f)


def _print_report(checks: list[Check], session_log: dict[str, Any]) -> bool:
    icons = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭️ "}
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)
    for c in checks:
        print(f"{icons[c.status]} {c.name:<20} {c.detail}")
        if c.hint:
            for line in c.hint.split(". "):
                if line.strip():
                    print(f"      ↳ {line.strip().rstrip('.')}.")
    print("=" * 72)

    final = (session_log.get("final_response") or {})
    content = final.get("content") if isinstance(final, dict) else final
    if content:
        print("\nFinal answer (judge the prompt quality yourself -- this part is subjective):")
        print("-" * 72)
        print(content[:2000] + ("\n[...truncated for display]" if len(content) > 2000 else ""))
        print("-" * 72)

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    skipped = [c for c in checks if c.status == SKIP]
    passed = len(checks) - len(failures) - len(warnings) - len(skipped)
    print(f"\n{passed} passed, {len(warnings)} warning(s), "
          f"{len(failures)} failure(s), {len(skipped)} skipped")
    return not failures


async def _run(task: str, state_dir: Path) -> None:
    os.environ["SCANUE_STATE_DIR"] = str(state_dir)
    # Import AFTER the env var is set: main resolves its state paths at import.
    import main as main_mod

    try:
        await main_mod.main([task])
    except SystemExit as e:
        if e.code:
            print(f"\n(the CLI exited with code {e.code})")


def main_entry() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default=DEFAULT_TASK, help="task to run")
    parser.add_argument("--keep", action="store_true", help="keep the temp state dir")
    args = parser.parse_args()

    print(f"Running one real task against your configured providers:\n  {args.task!r}\n")

    with tempfile.TemporaryDirectory(prefix="scanue-validate-") as tmp:
        state_dir = Path(tmp)
        asyncio.run(_run(args.task, state_dir))

        session_log = _load_latest_log(state_dir)
        if session_log is None:
            print("\n❌ No session log was written -- the run did not get far enough.")
            print("   Check that your provider is reachable (for Ollama: `ollama list`).")
            return 1

        ok = _print_report(analyze(session_log), session_log)
        if args.keep:
            kept = Path.cwd() / "validate-session.json"
            kept.write_text(json.dumps(session_log, indent=2, default=str), encoding="utf-8")
            print(f"\nFull session log written to {kept}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_entry())
