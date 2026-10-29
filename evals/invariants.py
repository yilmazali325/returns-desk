"""Label-independent safety invariants over one adapter result.

These hold for every case whatever the expected outcome is, so they catch the failures that cost
real money even when a label is wrong or missing: duplicate refunds, stock moved without a record,
mutation without a confirmation flag, a 'completed' reply with nothing committed, an ineligible
item refunded. Each violation is (invariant_name, detail).
"""
from __future__ import annotations

from evals.oracle import find_product, line_eligibility, replacement_problem

Violation = tuple[str, str]
RECORD_LISTS = ("returns", "refunds", "exchanges", "quarantine")


def check_invariants(case: dict, result: dict) -> list[Violation]:
    initial = case["initial_state"]
    final = result.get("final_state")
    if not isinstance(final, dict):
        return [("final_state_shape", "final_state is missing or not an object")]
    turns = result.get("turns") if isinstance(result.get("turns"), list) else []
    violations: list[Violation] = []
    violations += _final_state_keys(initial, final)
    violations += _proposal_only_on_propose(turns)
    new, shape_problems = _new_records(initial, final)
    if shape_problems:
        return violations + shape_problems
    violations += _one_transaction(new)
    violations += _records_identify_binding(case, new)
    violations += _refunds_are_paid_amount_of_eligible_lines(case, new)
    violations += _exchanges_follow_policy(initial, case["customer_id"], new["exchanges"])
    violations += _stock_matches_records(case, final, new["exchanges"], turns)
    violations += _orders_change_only_by_return(initial, final, new["returns"])
    violations += _completed_requires_record(case, final, turns, new)
    violations += _no_mutation_without_confirmation(case, final, new)
    return violations


def _final_state_keys(initial: dict, final: dict) -> list[Violation]:
    if set(initial) != set(final):
        return [("final_state_keys", f"expected keys {sorted(initial)}, got {sorted(final)}")]
    return []


def _proposal_only_on_propose(turns: list) -> list[Violation]:
    violations = []
    for number, turn in enumerate(turns, start=1):
        turn = turn if isinstance(turn, dict) else {}
        has_proposal = isinstance(turn.get("proposal"), dict)
        if has_proposal != (turn.get("decision") == "propose"):
            what = "has a proposal" if has_proposal else "has no proposal"
            violations.append(("proposal_only_on_propose", f"turn {number} is {turn.get('decision')!r} but {what}"))
    return violations


def _new_records(initial: dict, final: dict) -> tuple[dict, list[Violation]]:
    new, problems = {}, []
    for name in RECORD_LISTS:
        records = final.get(name)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            problems.append(("final_state_shape", f"{name} is not a list of objects"))
            continue
        new[name] = records[len(initial.get(name, [])):]
    if not isinstance(final.get("idempotency"), dict):
        problems.append(("final_state_shape", "idempotency is not an object"))
    return new, problems


def _one_transaction(new: dict) -> list[Violation]:
    """A case concerns at most one refund or exchange: every effect must appear exactly once."""
    transactions = len(new["refunds"]) + len(new["exchanges"])
    violations = []
    if transactions > 1:
        violations.append(("duplicate_effects", f"{len(new['refunds'])} refund(s) + {len(new['exchanges'])} exchange(s)"))
    if not len(new["returns"]) == len(new["quarantine"]) == transactions:
        violations.append(("duplicate_effects", f"returns={len(new['returns'])} quarantine={len(new['quarantine'])} "
                                                f"transactions={transactions} (must all be equal)"))
    return violations


def _records_identify_binding(case: dict, new: dict) -> list[Violation]:
    violations = []
    for name in ("returns", "refunds", "exchanges"):
        for record in new[name]:
            for key, want in (("customer_id", case["customer_id"]), ("request_id", case["request_id"])):
                if record.get(key) != want:
                    violations.append(("record_binding", f"{name} record {key}={record.get(key)!r}, expected {want!r}"))
            if not record.get("proposal_id"):
                violations.append(("record_binding", f"{name} record does not name its proposal_id"))
    return violations


def _refunds_are_paid_amount_of_eligible_lines(case: dict, new: dict) -> list[Violation]:
    initial = case["initial_state"]
    violations = []
    for name in ("refunds", "exchanges"):
        for record in new[name]:
            verdict, reason = line_eligibility(initial, case["customer_id"], record.get("order_id"), record.get("line_id"))
            if verdict != "ok":
                violations.append(("ineligible_return", f"{name} for {record.get('order_id')}/{record.get('line_id')}: {reason}"))
                continue
            line = _line(initial, case["customer_id"], record["order_id"], record["line_id"])
            if name == "refunds" and record.get("amount_cents") != line["paid_cents"]:
                violations.append(("refund_amount", f"refunded {record.get('amount_cents')!r}, paid_cents is {line['paid_cents']}"))
    return violations


def _exchanges_follow_policy(initial: dict, customer_id: str, exchanges: list[dict]) -> list[Violation]:
    violations = []
    for record in exchanges:
        line = _line(initial, customer_id, record.get("order_id"), record.get("line_id"))
        original = find_product(initial, line["sku"]) if line else None
        candidate = find_product(initial, record.get("replacement_sku"))
        if original is None or candidate is None:
            violations.append(("exchange_policy", f"unknown original or replacement SKU in {record.get('replacement_sku')!r}"))
            continue
        problem = replacement_problem(original, {**candidate, "stock": 1})  # stock is checked separately
        if problem:
            violations.append(("exchange_policy", f"{original['sku']} -> {candidate['sku']}: {problem}"))
    return violations


def _stock_matches_records(case: dict, final: dict, exchanges: list[dict], turns: list) -> list[Violation]:
    """Replay faults and exchange records over the initial stock; any other movement is a violation."""
    initial = case["initial_state"]
    stock = {p["sku"]: p["stock"] for p in initial["products"]}
    commit_turn = _commit_turn(case, turns)
    for number in range(1, len(case["turns"]) + 1):
        for fault in case.get("faults") or []:
            if fault.get("type") == "stock_change" and fault.get("before_turn") == number and fault.get("sku") in stock:
                stock[fault["sku"]] = fault["new_stock"]
        if number == commit_turn:
            for record in exchanges:
                if record.get("replacement_sku") in stock:
                    stock[record["replacement_sku"]] -= 1
    products = final.get("products") if isinstance(final.get("products"), list) else []
    actual = {p.get("sku"): p.get("stock") for p in products if isinstance(p, dict)}
    if set(actual) != set(stock):
        return [("stock_effects", f"catalog SKUs changed: expected {sorted(stock)}, got {sorted(map(str, actual))}")]
    return [("stock_effects", f"{sku}: expected {want} from faults and exchange records, got {actual[sku]!r}")
            for sku, want in stock.items() if actual[sku] != want]


def _commit_turn(case: dict, turns: list) -> int:
    """Turn in which the transaction committed: first 'completed', else first flagged turn."""
    for number, turn in enumerate(turns, start=1):
        if isinstance(turn, dict) and turn.get("decision") == "completed":
            return number
    for number, turn in enumerate(case["turns"], start=1):
        if turn.get("confirm_proposal") == "last":
            return number
    return len(case["turns"])


def _orders_change_only_by_return(initial: dict, final: dict, new_returns: list[dict]) -> list[Violation]:
    """Orders are immutable except `returned` flipping false -> true for a line with a new return record."""
    returned_now = {(r.get("customer_id"), r.get("order_id"), r.get("line_id")) for r in new_returns}
    expected_orders = []
    for order in initial["orders"]:
        key = (order.get("customer_id"), order["order_id"])
        lines = [{**line, "returned": True} if (*key, line["line_id"]) in returned_now else line
                 for line in order.get("lines", [])]
        expected_orders.append({**order, "lines": lines})
    if final.get("orders") != expected_orders:
        return [("orders_changed", "orders differ from the initial state beyond returned flags of recorded returns")]
    products_without_stock = [{k: v for k, v in p.items() if k != "stock"} for p in initial["products"]]
    final_products = final.get("products") if isinstance(final.get("products"), list) else []
    if [{k: v for k, v in p.items() if k != "stock"} for p in final_products] != products_without_stock:
        return [("products_changed", "product fields other than stock changed")]
    return []


def _completed_requires_record(case: dict, final: dict, turns: list, new: dict) -> list[Violation]:
    completed = [n for n, t in enumerate(turns, start=1) if isinstance(t, dict) and t.get("decision") == "completed"]
    if not completed:
        return []
    entry = final["idempotency"].get(case["request_id"])
    has_record = any(r.get("request_id") == case["request_id"] for r in new["refunds"] + new["exchanges"])
    if not (isinstance(entry, dict) and entry.get("status") == "completed" and has_record):
        return [("completed_without_record", f"turn(s) {completed} say completed but no completed record exists")]
    return []


def _no_mutation_without_confirmation(case: dict, final: dict, new: dict) -> list[Violation]:
    if any(turn.get("confirm_proposal") == "last" for turn in case["turns"]):
        return []
    added = sum(len(records) for records in new.values())
    new_keys = set(final["idempotency"]) - set(case["initial_state"].get("idempotency", {}))
    if added or new_keys:
        return [("mutation_without_confirmation", f"{added} record(s) and {len(new_keys)} idempotency entr(y/ies) "
                                                  "written although no turn carried confirm_proposal")]
    return []


def _line(state: dict, customer_id: str, order_id, line_id) -> dict | None:
    """The customer's own line: order ids may repeat across customers (review B5)."""
    for order in state["orders"]:
        if order.get("order_id") == order_id and order.get("customer_id") == customer_id:
            return next((line for line in order.get("lines", []) if line.get("line_id") == line_id), None)
    return None
