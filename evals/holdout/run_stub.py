"""Code-only dependence probe: run a contract suite through the REAL desk (orchestrator, gates,
gateway, store) with a non-model policy in place of the local LLM, and write an adapter-shaped
results file that evals.score can grade.

    .venv/bin/python -m evals.holdout.run_stub --mode fake-policy   --input X --output Y [--profile F]
    .venv/bin/python -m evals.holdout.run_stub --mode always-clarify --input X --output Y [--profile F]

Modes
  fake-policy      desk.testing.fake_policy_llm.FakePolicyLLM: the repo's own deterministic double
                   (grounding regexes + documented tool order). Answers "how much of the suite does
                   code alone solve?"
  always-clarify   an unhelpful agent: Resolution only ever replies with a generic question,
                   Operations only ever reports `needs_info`. Whatever still succeeds is solved by
                   orchestrator fallbacks / vetoes, not by any agent decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from desk.agents.toolspecs import REPLY, REPORT
from desk.llm import RawCall, interpret_calls
from desk.profiles import get_profile
from desk.session import run_case
from desk.testing.fake_policy_llm import FakePolicyLLM


class AlwaysClarifyLLM:
    protocol = "stub-always-clarify"
    engine = "stub (no model)"

    def step(self, role, messages, tools, allowed, max_tokens):
        allowed = list(allowed)
        if role == "operations":
            free_text = any(t.name == REPORT and "summary" in t.parameters.get("properties", {}) for t in tools)
            if REPORT in allowed:
                name, args = REPORT, {"outcome": "needs_info", **({"summary": "Need more information."} if free_text else {})}
            else:
                name, args = allowed[0], {}
        else:
            name, args = (REPLY, {"message": "Could you tell me a bit more about what you need?"}) if REPLY in allowed \
                else (allowed[0], {})
        step = interpret_calls([RawCall(name, args)], tools, allowed)
        step.role = role
        step.text = json.dumps({"tool": name, "args": args}, sort_keys=True)
        step.prompt_tokens = 0
        step.completion_tokens = 1
        step.latency_ms = 0.0
        step.finish_reason = "stop"
        step.http_attempts = 1
        return step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fake-policy", "always-clarify"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="F")
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    doc = json.loads(raw)
    profile = get_profile(args.profile)
    llm = FakePolicyLLM() if args.mode == "fake-policy" else AlwaysClarifyLLM()
    results = []
    for case in doc["cases"]:
        result = run_case(case, llm, profile)
        results.append(result)
        print(case["case_id"], [t["decision"] for t in result["turns"]], flush=True)
    out = {
        "contract_version": 1,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "hardware": {"note": "code-only probe, no model"},
        "configuration": {"mode": args.mode, "profile": profile.name, "llm": getattr(llm, "engine", args.mode)},
        "results": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
