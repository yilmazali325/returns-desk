"""Build (or verify) the frozen held-out suite v3 from the hand-written case specs.

    .venv/bin/python -m evals.holdout_v3.build_v3           # write holdout_v3.json + labels_v3.json
    .venv/bin/python -m evals.holdout_v3.build_v3 --check   # exit 1 if the files differ from the specs

Every decision, reason, proposal binding and final money/stock effect is written by hand in
cases_terse.py / cases_natural.py. evals/oracle.py is used only to compute the final business
state from the grounded intents, and the build FAILS if the oracle disagrees with any hand label:
a disagreement must be resolved by a person (and noted in FROZEN.md), never silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.holdout_v3.testdata import cases_natural, cases_terse
from evals.holdout_v3.spec import CURRENCY, CaseSpec, Proposal
from evals.oracle import build_expectation, find_line, find_order

HERE = Path(__file__).resolve().parent
INPUT_NAME = "holdout_v3.json"
LABELS_NAME = "labels_v3.json"
SOURCE = ("Hand-decided from candidate_assets/CONTRACT.md and docs/CLARIFICATIONS.md defaults by a blind author "
          "(no desk/, server/, tests/ or review access). evals/oracle.py computed only the final state from the "
          "hand-grounded intents; build_v3.py asserts it agrees with every hand label. See FROZEN.md.")


class LabelMismatch(AssertionError):
    """The oracle's reading of a case differs from the hand label."""


def all_specs() -> list[CaseSpec]:
    return sorted(cases_terse.CASES + cases_natural.CASES, key=lambda spec: spec.case_id)


# --------------------------------------------------------------------------- hand label -> scorer shape

def proposal_view(spec: CaseSpec, proposal: Proposal) -> dict:
    """The binding a propose turn must carry, in the scorer's field order."""
    original = find_line(find_order(spec.initial_state, proposal.order_id), proposal.line_id)
    return {"seq": proposal.seq, "customer_id": spec.customer_id, "request_id": spec.resolved_request_id(),
            "order_id": proposal.order_id, "line_id": proposal.line_id, "remedy": proposal.remedy,
            "original_sku": original["sku"], "replacement_sku": proposal.replacement_sku,
            "amount_cents": proposal.amount_cents, "currency": CURRENCY}


def turn_mismatches(spec: CaseSpec, expected: dict) -> list[str]:
    problems = []
    for number, turn in enumerate(spec.turns, start=1):
        got = (expected["decisions"][number - 1], expected["reasons"][number - 1])
        if got != (turn.decision, turn.reason):
            problems.append(f"turn {number}: hand {turn.decision}/{turn.reason}, oracle {got[0]}/{got[1]}")
        want_proposal = proposal_view(spec, turn.proposal) if turn.proposal else None
        if expected["proposals"][number - 1] != want_proposal:
            problems.append(f"turn {number}: hand proposal {want_proposal}, oracle {expected['proposals'][number - 1]}")
        contested = expected["alternatives"].get(str(number), [])
        if not set(contested) <= set(turn.alternatives):
            problems.append(f"turn {number}: oracle marks {contested} as contested; add them to the hand alternatives")
    return problems


def effect_mismatches(spec: CaseSpec, expected: dict) -> list[str]:
    """Compare the oracle's final state with the hand-declared transaction and stock movements."""
    final = expected["final_business_state"]
    request_id = spec.resolved_request_id()
    refunds, exchanges = [], []
    if spec.effect and spec.effect.remedy == "refund":
        refunds = [{"request_id": request_id, "customer_id": spec.customer_id, "order_id": spec.effect.order_id,
                    "line_id": spec.effect.line_id, "amount_cents": spec.effect.amount_cents}]
    if spec.effect and spec.effect.remedy == "exchange":
        original = find_line(find_order(spec.initial_state, spec.effect.order_id), spec.effect.line_id)
        exchanges = [{"request_id": request_id, "customer_id": spec.customer_id, "order_id": spec.effect.order_id,
                      "line_id": spec.effect.line_id, "original_sku": original["sku"],
                      "replacement_sku": spec.effect.replacement_sku}]
    problems = []
    if final["refunds"] != refunds:
        problems.append(f"refunds: hand {refunds}, oracle {final['refunds']}")
    if final["exchanges"] != exchanges:
        problems.append(f"exchanges: hand {exchanges}, oracle {final['exchanges']}")
    returned = [(o["order_id"], ln["line_id"]) for o in final["orders"] for ln in o["lines"] if ln["returned"]]
    initially = [(o["order_id"], ln["line_id"]) for o in spec.initial_state["orders"] for ln in o["lines"] if ln["returned"]]
    newly = sorted(set(returned) - set(initially))
    want_newly = [(spec.effect.order_id, spec.effect.line_id)] if spec.effect else []
    if newly != want_newly:
        problems.append(f"returned lines: hand {want_newly}, oracle {newly}")
    initial_stock = {p["sku"]: p["stock"] for p in spec.initial_state["products"]}
    moved = {p["sku"]: p["stock"] for p in final["products"] if p["stock"] != initial_stock[p["sku"]]}
    if moved != spec.stock_after:
        problems.append(f"stock: hand {spec.stock_after}, oracle {moved}")
    if expected["state_unchanged"] != (spec.effect is None):
        problems.append(f"state_unchanged: oracle {expected['state_unchanged']} but hand effect is {spec.effect}")
    if not all(event["applied"] for event in expected["events"]):
        problems.append("a stock_change fault did not apply (unknown SKU or turn out of range)")
    return problems


def label_for(spec: CaseSpec) -> dict:
    case = spec.contract_case()
    intents = [turn.intent for turn in spec.turns]
    expected = build_expectation(case, intents)
    problems = turn_mismatches(spec, expected) + effect_mismatches(spec, expected)
    if problems:
        raise LabelMismatch(f"{spec.case_id}: " + "; ".join(problems))
    expected["alternatives"] = {str(number): list(turn.alternatives)
                                for number, turn in enumerate(spec.turns, start=1) if turn.alternatives}
    return {"case_id": spec.case_id, "family": spec.family, "note": spec.note, "intents": intents, "expected": expected}


# --------------------------------------------------------------------------- documents

def build_documents(specs: list[CaseSpec]) -> tuple[dict, dict]:
    cases = {"contract_version": 1, "cases": [spec.contract_case() for spec in specs]}
    labels = {"labels_version": 1, "source": SOURCE, "cases": [label_for(spec) for spec in specs]}
    return cases, labels


def render(document: dict) -> str:
    # ensure_ascii=False keeps the natural messages readable (one contains an emoji); the file is UTF-8.
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the held-out v3 suite.")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--check", action="store_true", help="verify the files instead of writing them")
    args = parser.parse_args(argv)
    cases, labels = build_documents(all_specs())
    outputs = {args.out_dir / INPUT_NAME: render(cases), args.out_dir / LABELS_NAME: render(labels)}
    if args.check:
        stale = [path.name for path, text in outputs.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != text]
        print(f"STALE: {stale}" if stale else f"up to date: {len(cases['cases'])} cases")
        return 1 if stale else 0
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(cases['cases'])} cases to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
