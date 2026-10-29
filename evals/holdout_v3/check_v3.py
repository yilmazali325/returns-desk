"""Structural checker for the v3 suite. It never runs the desk, the adapter, a model or the scorer.

    .venv/bin/python -m evals.holdout_v3.check_v3

1. holdout_v3.json has exactly the shape of candidate_assets/public_cases.json: the same keys at
   every level (derived from the public file, not hard-coded), integer money/stock, boolean flags,
   ISO dates, empty transaction records, and faults that point at a real SKU and turn.
2. labels_v3.json has exactly the fields evals/score.py reads for --labels, one label per input
   case in the same order, and a 30 terse / 30 natural split.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from evals.score import EVENT_FIELDS, PROPOSAL_FIELDS

HERE = Path(__file__).resolve().parent
PUBLIC_CASES = HERE.parents[1] / "candidate_assets" / "public_cases.json"
DECISIONS = {"propose", "clarify", "deny", "completed", "error"}
LABEL_KEYS = {"case_id", "family", "note", "intents", "expected"}
EXPECTED_KEYS = {"decisions", "reasons", "proposals", "events", "final_business_state", "state_unchanged", "alternatives"}
STYLE_COUNTS = {"terse": 30, "natural": 30}
RECORD_LISTS = ("returns", "refunds", "exchanges", "quarantine")


def need(errors: list[str], ok: bool, message: str) -> None:
    if not ok:
        errors.append(message)


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_iso_date(value: object) -> bool:
    try:
        date.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- input shape

def reference_shape(public: dict) -> dict[str, set[str]]:
    """Key sets at every level of the public fixtures (the union over all public cases)."""
    shape: dict[str, set[str]] = {"root": set(public)}

    def add(level: str, items: list[dict]) -> None:
        for item in items:
            shape.setdefault(level, set()).update(item)

    cases = public["cases"]
    add("case", cases)
    add("state", [case["initial_state"] for case in cases])
    add("product", [p for case in cases for p in case["initial_state"]["products"]])
    add("order", [o for case in cases for o in case["initial_state"]["orders"]])
    add("line", [ln for case in cases for o in case["initial_state"]["orders"] for ln in o["lines"]])
    add("turn", [t for case in cases for t in case["turns"]])
    for fault in (f for case in cases for f in case["faults"]):
        add(f"fault:{fault['type']}", [fault])
    return shape


def product_errors(where: str, products: list, shape: dict) -> list[str]:
    errors: list[str] = []
    skus = [p.get("sku") for p in products]
    need(errors, len(skus) == len(set(skus)), f"{where}: duplicate SKU")
    for p in products:
        need(errors, set(p) == shape["product"], f"{where}: product keys {sorted(p)}")
        need(errors, is_int(p.get("price_cents")) and is_int(p.get("stock")), f"{where}: {p.get('sku')} money/stock not int")
    return errors


def order_errors(where: str, orders: list, skus: set, shape: dict) -> list[str]:
    errors: list[str] = []
    order_ids = [o.get("order_id") for o in orders]
    need(errors, len(order_ids) == len(set(order_ids)), f"{where}: duplicate order_id")
    for o in orders:
        label = f"{where}/{o.get('order_id')}"
        need(errors, set(o) == shape["order"], f"{label}: order keys {sorted(o)}")
        delivered = o.get("status") == "delivered"
        need(errors, is_iso_date(o.get("delivered_on")) if delivered else o.get("delivered_on") is None,
             f"{label}: delivered_on must be an ISO date when delivered, else null")
        line_ids = [ln.get("line_id") for ln in o.get("lines", [])]
        need(errors, bool(line_ids) and len(line_ids) == len(set(line_ids)), f"{label}: missing or duplicate line_id")
        for ln in o.get("lines", []):
            need(errors, set(ln) == shape["line"], f"{label}: line keys {sorted(ln)}")
            need(errors, is_int(ln.get("paid_cents")), f"{label}: paid_cents not int")
            need(errors, all(isinstance(ln.get(k), bool) for k in ("unopened", "final_sale", "returned")),
                 f"{label}: line flags not boolean")
            need(errors, ln.get("sku") in skus, f"{label}: line SKU {ln.get('sku')!r} not in catalog")
    return errors


def turn_and_fault_errors(case: dict, skus: set, shape: dict) -> list[str]:
    errors: list[str] = []
    where = case.get("case_id")
    turns = case.get("turns") or []
    need(errors, bool(turns), f"{where}: no turns")
    for t in turns:
        need(errors, set(t) <= shape["turn"] and isinstance(t.get("message"), str) and bool(t.get("message")),
             f"{where}: bad turn {t}")
        need(errors, t.get("confirm_proposal", "last") == "last", f"{where}: confirm_proposal must be 'last'")
    for f in case.get("faults", []):
        level = f"fault:{f.get('type')}"
        need(errors, level in shape and set(f) == shape.get(level), f"{where}: fault shape {f}")
        if f.get("type") == "stock_change":
            need(errors, f.get("sku") in skus and is_int(f.get("new_stock")), f"{where}: stock_change target {f}")
            need(errors, is_int(f.get("before_turn")) and 1 <= f["before_turn"] <= len(turns),
                 f"{where}: before_turn outside 1..{len(turns)}")
    return errors


def case_errors(case: dict, shape: dict) -> list[str]:
    where = case.get("case_id")
    errors: list[str] = []
    need(errors, set(case) == shape["case"], f"{where}: case keys {sorted(case)}")
    need(errors, all(isinstance(case.get(k), str) and case.get(k) for k in ("case_id", "customer_id", "request_id")),
         f"{where}: ids must be non-empty strings")
    st = case.get("initial_state", {})
    need(errors, set(st) == shape["state"], f"{where}: state keys {sorted(st)}")
    need(errors, is_iso_date(st.get("as_of")) and st.get("currency") == "USD", f"{where}: as_of/currency")
    need(errors, all(st.get(name) == [] for name in RECORD_LISTS) and st.get("idempotency") == {},
         f"{where}: transaction records must start empty")
    products = st.get("products", [])
    skus = {p.get("sku") for p in products}
    errors += product_errors(where, products, shape)
    errors += order_errors(where, st.get("orders", []), skus, shape)
    errors += turn_and_fault_errors(case, skus, shape)
    return errors


def suite_errors(suite: dict, public: dict) -> list[str]:
    shape = reference_shape(public)
    errors: list[str] = []
    need(errors, set(suite) == shape["root"] and suite.get("contract_version") == 1, "root keys / contract_version")
    cases = suite.get("cases", [])
    for key in ("case_id", "request_id"):
        values = [case.get(key) for case in cases]
        need(errors, len(values) == len(set(values)), f"duplicate {key} across cases")
    for case in cases:
        errors += case_errors(case, shape)
    return errors


# --------------------------------------------------------------------------- label format (what evals/score.py reads)

def proposal_errors(where: str, case: dict, decisions: list, proposals: list) -> list[str]:
    errors: list[str] = []
    for number, (decision, proposal) in enumerate(zip(decisions, proposals), start=1):
        need(errors, (proposal is not None) == (decision == "propose"), f"{where} turn {number}: proposal iff propose")
        if proposal is not None:
            need(errors, set(proposal) == {"seq", *PROPOSAL_FIELDS}, f"{where} turn {number}: proposal keys")
            need(errors, proposal.get("customer_id") == case["customer_id"]
                 and proposal.get("request_id") == case["request_id"], f"{where} turn {number}: proposal not bound to case")
            need(errors, is_int(proposal.get("amount_cents")) and is_int(proposal.get("seq")),
                 f"{where} turn {number}: amount/seq not int")
    return errors


def expected_errors(where: str, case: dict, expected: dict) -> list[str]:
    errors: list[str] = []
    turns = len(case["turns"])
    decisions = expected.get("decisions", [])
    for name in ("decisions", "reasons", "proposals"):
        need(errors, len(expected.get(name, [])) == turns, f"{where}: {name} length != {turns} turns")
    need(errors, set(decisions) <= DECISIONS, f"{where}: unknown decision in {decisions}")
    errors += proposal_errors(where, case, decisions, expected.get("proposals", []))
    for number, accepted in expected.get("alternatives", {}).items():
        valid_turn = number.isdigit() and 1 <= int(number) <= len(decisions)
        need(errors, valid_turn and set(accepted) <= DECISIONS and decisions[int(number) - 1] not in accepted,
             f"{where}: bad alternatives for turn {number}")
    need(errors, all(set(e) == set(EVENT_FIELDS) for e in expected.get("events", [])), f"{where}: event keys")
    final = expected.get("final_business_state", {})
    need(errors, final.get("top_level_keys") == sorted(case["initial_state"]), f"{where}: final state keys")
    need(errors, isinstance(expected.get("state_unchanged"), bool), f"{where}: state_unchanged not bool")
    return errors


def label_errors(labels: dict, suite: dict) -> list[str]:
    errors: list[str] = []
    need(errors, labels.get("labels_version") == 1, "labels_version must be 1")
    cases = suite.get("cases", [])
    entries = labels.get("cases", [])
    need(errors, [e.get("case_id") for e in entries] == [c.get("case_id") for c in cases],
         "label case_ids differ from the input cases or their order")
    styles = {style: 0 for style in STYLE_COUNTS}
    for entry, case in zip(entries, cases):
        where = entry.get("case_id")
        need(errors, set(entry) == LABEL_KEYS and set(entry.get("expected", {})) == EXPECTED_KEYS,
             f"{where}: label keys")
        style = str(entry.get("family", "")).split("_")[0]
        need(errors, style in styles, f"{where}: family must start with terse_ or natural_")
        styles[style] = styles.get(style, 0) + 1
        need(errors, len(entry.get("intents", [])) == len(case["turns"]), f"{where}: one intent per turn")
        errors += expected_errors(where, case, entry.get("expected", {}))
    need(errors, styles == STYLE_COUNTS, f"style split {styles}, expected {STYLE_COUNTS}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the v3 suite's input shape and label format.")
    parser.add_argument("--input", type=Path, default=HERE / "holdout_v3.json")
    parser.add_argument("--labels", type=Path, default=HERE / "labels_v3.json")
    parser.add_argument("--public", type=Path, default=PUBLIC_CASES)
    args = parser.parse_args(argv)
    try:
        suite = json.loads(args.input.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        public = json.loads(args.public.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)
        return 2
    errors = suite_errors(suite, public) + label_errors(labels, suite)
    for line in errors:
        print(f"ERROR {line}")
    turns = sum(len(case["turns"]) for case in suite.get("cases", []))
    print(f"{'FAILED' if errors else 'OK'}: {len(suite.get('cases', []))} cases, {turns} turns, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
