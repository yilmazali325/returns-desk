"""Review harness: run contract cases through the REAL orchestrator/gateway/store with the
deterministic FakePolicyLLM (same double the UI uses under DESK_LLM=fake-policy).

    .venv/bin/python tests/review/business/biz_harness.py <cases.json> [--json out.json] [--case ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desk.profiles import FINAL  # noqa: E402
from desk.session import run_case  # noqa: E402
from desk.testing.fake_policy_llm import FakePolicyLLM  # noqa: E402


def run(case: dict, profile=FINAL) -> dict:
    return run_case(case, FakePolicyLLM(), profile)


def summary(result: dict) -> dict:
    fs = result["final_state"]
    return {
        "case_id": result["case_id"],
        "decisions": [t["decision"] for t in result["turns"]],
        "responses": [t["response"] for t in result["turns"]],
        "proposals": [t["proposal"] and {k: t["proposal"][k] for k in ("proposal_id", "order_id", "line_id", "remedy", "replacement_sku", "amount_cents")} for t in result["turns"]],
        "stock": {p["sku"]: p["stock"] for p in fs["products"]},
        "returned": {f'{o["order_id"]}/{l["line_id"]}': l["returned"] for o in fs["orders"] for l in o["lines"]},
        "refunds": [(r["order_id"], r["line_id"], r["amount_cents"]) for r in fs["refunds"]],
        "exchanges": [(r["order_id"], r["line_id"], r["original_sku"], r["replacement_sku"]) for r in fs["exchanges"]],
        "quarantine": [(r["order_id"], r["line_id"], r["sku"]) for r in fs["quarantine"]],
        "idempotency": list(fs["idempotency"]),
        "status": result["metrics"]["status"],
        "error": result["metrics"].get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--json")
    parser.add_argument("--case")
    args = parser.parse_args()
    doc = json.loads(Path(args.path).read_text())
    out = []
    for case in doc["cases"]:
        if args.case and case["case_id"] != args.case:
            continue
        s = summary(run(case))
        out.append(s)
        print(json.dumps(s, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
