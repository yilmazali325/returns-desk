"""Build the business-rule adversarial cases (contract input format) and their oracle intents.

    .venv/bin/python tests/review/business/biz_build_cases.py

Writes tests/review/business/cases/adversarial_v1.json (contract_version 1 + cases, the exact schema
of candidate_assets/public_cases.json) and cases/adversarial_intents.json (one oracle intent per turn,
for evals.oracle.build_expectation). IDs deliberately differ from the public fixtures.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases"

CATALOG = [
    {"sku": "POLO-S", "product_id": "POLO", "size": "S", "price_cents": 5500, "stock": 5},
    {"sku": "POLO-M", "product_id": "POLO", "size": "M", "price_cents": 5500, "stock": 2},
    {"sku": "POLO-L", "product_id": "POLO", "size": "L", "price_cents": 5500, "stock": 1},
    {"sku": "POLO-XL", "product_id": "POLO", "size": "XL", "price_cents": 6000, "stock": 4},
    {"sku": "BOOT-9", "product_id": "BOOT", "size": "9", "price_cents": 12000, "stock": 1},
    {"sku": "BOOT-9.5", "product_id": "BOOT", "size": "9.5", "price_cents": 12000, "stock": 2},
    {"sku": "BOOT-10", "product_id": "BOOT", "size": "10", "price_cents": 12000, "stock": 0},
]


def line(line_id, sku, paid, **over):
    base = {"line_id": line_id, "sku": sku, "paid_cents": paid, "unopened": True, "final_sale": False,
            "returned": False}
    base.update(over)
    return base


def order(order_id, customer, delivered_on, lines, status="delivered", **extra):
    o = {"order_id": order_id, "customer_id": customer, "status": status, "delivered_on": delivered_on,
         "lines": lines}
    o.update(extra)
    return o


def state(orders, as_of="2027-02-10", products=None, **over):
    s = {"as_of": as_of, "currency": "USD", "products": copy.deepcopy(products or CATALOG), "orders": orders,
         "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}
    s.update(over)
    return s


CASES: list[tuple[dict, list[dict]]] = []


def add(case_id, customer, st, turns, intents, faults=None):
    case = {"case_id": case_id, "customer_id": customer, "request_id": f"RQ-{case_id}", "initial_state": st,
            "turns": turns, "faults": faults or []}
    CASES.append((case, intents))


YES = {"message": "Yes, I confirm the proposal.", "confirm_proposal": "last"}
CONFIRM = {"kind": "confirm"}

# 1. Day 30 with a full ISO timestamp in delivered_on (date part decides).
add("adv01_day30_timestamp", "K1",
    state([order("Z-501", "K1", "2027-01-11T23:59:59Z", [line("A1", "POLO-M", 4999)])]),
    [{"message": "Please refund item A1 of order Z-501."}, YES],
    [{"kind": "refund", "order_id": "Z-501", "line_id": "A1"}, CONFIRM])

# 2. Day 0 delivered later on the same calendar day as as_of (timestamp after midnight).
add("adv02_day0_same_day_timestamp", "K1",
    state([order("Z-502", "K1", "2027-02-10T18:30:00+00:00", [line("A1", "POLO-M", 1)])]),
    [{"message": "Refund line A1 on order Z-502, never opened."}, YES],
    [{"kind": "refund", "order_id": "Z-502", "line_id": "A1"}, CONFIRM])

# 3. Two units of the same SKU as two lines; one already returned. Customer describes the item.
add("adv03_same_sku_two_lines_one_returned", "K1",
    state([order("Z-503", "K1", "2027-02-01", [line("A1", "POLO-M", 5500, returned=True),
                                               line("A2", "POLO-M", 4400)])]),
    [{"message": "I'd like a refund for the medium polo from Z-503."},
     {"message": "It is line A2."}, YES],
    [{"kind": "clarify_needed", "reason": "TARGET_NOT_UNIQUE"},
     {"kind": "refund", "order_id": "Z-503", "line_id": "A2"}, CONFIRM])

# 4. Exchange + timeout_after_commit + repeated confirmation: stock -1 exactly once, quarantine 1.
add("adv04_exchange_timeout_replay", "K1",
    state([order("Z-504", "K1", "2027-01-30", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Exchange A1 from order Z-504 for the polo in size S."}, YES,
     {"message": "Confirming again, I never saw the result.", "confirm_proposal": "last"}],
    [{"kind": "exchange", "order_id": "Z-504", "line_id": "A1", "requested_size": "S"}, CONFIRM, CONFIRM],
    faults=[{"type": "timeout_after_commit"}])

# 5. External stock change on the ORIGINAL sku before confirmation of a refund: final stock is the
#    assigned value, never incremented by the return.
add("adv05_original_sku_stock_change", "K1",
    state([order("Z-505", "K1", "2027-01-20", [line("A1", "POLO-L", 5500)])]),
    [{"message": "Refund line A1 of order Z-505."}, YES],
    [{"kind": "refund", "order_id": "Z-505", "line_id": "A1"}, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 0}])

# 6. Stale stock, then the customer picks the other valid size, then confirms.
add("adv06_stale_then_other_size", "K1",
    state([order("Z-506", "K1", "2027-01-25", [line("A1", "POLO-M", 5000)])]),
    [{"message": "Exchange line A1 of Z-506 for size L please."}, YES,
     {"message": "Okay, then exchange it for size S."}, YES],
    [{"kind": "exchange", "order_id": "Z-506", "line_id": "A1", "requested_size": "L"}, CONFIRM,
     {"kind": "exchange", "order_id": "Z-506", "line_id": "A1", "requested_size": "S"}, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 0}])

# 7. Timeout armed, but the first confirm fails revalidation (stale stock), so the timeout must fire
#    on the later, first REAL commit; recovery reconciles, no second commit.
add("adv07_timeout_fires_on_first_real_commit", "K1",
    state([order("Z-507", "K1", "2027-01-25", [line("A1", "POLO-M", 5000)])]),
    [{"message": "Exchange line A1 of Z-507 for size L please."}, YES,
     {"message": "Fine, exchange it for size S instead."}, YES,
     {"message": "Did that go through? Confirming again.", "confirm_proposal": "last"}],
    [{"kind": "exchange", "order_id": "Z-507", "line_id": "A1", "requested_size": "L"}, CONFIRM,
     {"kind": "exchange", "order_id": "Z-507", "line_id": "A1", "requested_size": "S"}, CONFIRM, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 0},
            {"type": "timeout_after_commit"}])

# 8. Numeric (shoe) sizes with a half size.
add("adv08_numeric_half_size_exchange", "K1",
    state([order("Z-508", "K1", "2027-02-05", [line("A1", "BOOT-9", 9999)])]),
    [{"message": "Exchange line A1 of order Z-508 for size 9.5 boots."}, YES],
    [{"kind": "exchange", "order_id": "Z-508", "line_id": "A1", "requested_size": "9.5"}, CONFIRM])

# 9. Two customers whose orders reuse the same order_id string (data collision). The other
#    customer's line is already returned; ours is eligible and must be refundable.
add("adv09_order_id_collision_across_customers", "K1",
    state([order("Z-509", "K2", "2027-02-01", [line("A1", "POLO-M", 5500, returned=True)]),
           order("Z-509", "K1", "2027-02-01", [line("A1", "POLO-S", 5500)])]),
    [{"message": "Refund line A1 of order Z-509."}, YES],
    [{"kind": "refund", "order_id": "Z-509", "line_id": "A1"}, CONFIRM])

# 10. After a completed refund, the same request asks to exchange the SAME line (different remedy).
add("adv10_after_refund_asks_exchange_same_line", "K1",
    state([order("Z-510", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Refund line A1 of order Z-510."}, YES,
     {"message": "Actually I'd rather exchange line A1 of Z-510 for size S."}],
    [{"kind": "refund", "order_id": "Z-510", "line_id": "A1"}, CONFIRM,
     {"kind": "exchange", "order_id": "Z-510", "line_id": "A1", "requested_size": "S"}])

# 11. Confirm wording that contains a cancel-like word but is a plain confirmation.
add("adv11_confirm_with_instead_wording", "K1",
    state([order("Z-511", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Refund line A1 of order Z-511."},
     {"message": "Yes, please refund it to my card instead of store credit.", "confirm_proposal": "last"}],
    [{"kind": "refund", "order_id": "Z-511", "line_id": "A1"}, CONFIRM])

# 12. Unknown extra fields everywhere must round-trip; odd paid cents refunded exactly.
st12 = state([order("Z-512", "K1", "2027-02-01", [line("A1", "POLO-M", 3333, gift=True, notes={"x": 1})],
                    channel="web")], store_id="S-9")
st12["products"][1]["color"] = "navy"
add("adv12_extra_fields_odd_cents", "K1", st12,
    [{"message": "Refund line A1 of order Z-512 please."}, YES],
    [{"kind": "refund", "order_id": "Z-512", "line_id": "A1"}, CONFIRM])

# 13. Order status in a different letter case.
add("adv13_status_case", "K1",
    state([order("Z-513", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)], status="Delivered")]),
    [{"message": "Refund line A1 of order Z-513."}],
    [{"kind": "refund", "order_id": "Z-513", "line_id": "A1"}])

# 14. Restock of an out-of-stock replacement before the customer repeats the request.
add("adv14_restock_then_exchange", "K1",
    state([order("Z-514", "K1", "2027-02-01", [line("A1", "BOOT-9.5", 12000)])]),
    [{"message": "Exchange line A1 of order Z-514 for size 10."},
     {"message": "Can you check size 10 again for line A1 of Z-514?"}, YES],
    [{"kind": "exchange", "order_id": "Z-514", "line_id": "A1", "requested_size": "10"},
     {"kind": "exchange", "order_id": "Z-514", "line_id": "A1", "requested_size": "10"}, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 2, "sku": "BOOT-10", "new_stock": 3}])

# 15. Stock change after the commit (before the replay turn) is plain assignment of the new value.
add("adv15_stock_change_after_commit", "K1",
    state([order("Z-515", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Exchange line A1 of Z-515 for size S."}, YES,
     {"message": "Just checking it is done.", "confirm_proposal": "last"}],
    [{"kind": "exchange", "order_id": "Z-515", "line_id": "A1", "requested_size": "S"}, CONFIRM, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 3, "sku": "POLO-S", "new_stock": 9}])

# 16. Exchange to XL: same product, but a different catalog price -> deny, nothing changes.
add("adv16_exchange_price_mismatch_xl", "K1",
    state([order("Z-516", "K1", "2027-02-01", [line("A1", "POLO-L", 5500)])]),
    [{"message": "Exchange line A1 of order Z-516 for size XL."}],
    [{"kind": "exchange", "order_id": "Z-516", "line_id": "A1", "requested_size": "XL"}])

# 17. Paid zero (fully discounted) unit: refund exactly 0.
add("adv17_paid_zero_refund", "K1",
    state([order("Z-517", "K1", "2027-02-01", [line("A1", "POLO-M", 0)])]),
    [{"message": "Refund line A1 of order Z-517."}, YES],
    [{"kind": "refund", "order_id": "Z-517", "line_id": "A1"}, CONFIRM])

# 18. Two stock changes for the same sku before the same turn: list order, last one wins.
add("adv18_two_changes_same_turn", "K1",
    state([order("Z-518", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Exchange line A1 of Z-518 for size L."}, YES],
    [{"kind": "exchange", "order_id": "Z-518", "line_id": "A1", "requested_size": "L"}, CONFIRM],
    faults=[{"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 0},
            {"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 2}])


# 19. "Return ... and send me a large": the customer asks for a different size of the same item.
add("adv19_return_and_send_large", "K1",
    state([order("Z-519", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Return line A1 of order Z-519 and send me a large."}, YES],
    [{"kind": "exchange", "order_id": "Z-519", "line_id": "A1", "requested_size": "L"}, CONFIRM])

# 20. A plain confirmation that contains a negated cancel-type word ("no need to wait").
add("adv20_confirm_no_need_to_wait", "K1",
    state([order("Z-520", "K1", "2027-02-01", [line("A1", "POLO-M", 5500)])]),
    [{"message": "Exchange line A1 of order Z-520 for size S."},
     {"message": "Yes please go ahead, no need to wait.", "confirm_proposal": "last"}],
    [{"kind": "exchange", "order_id": "Z-520", "line_id": "A1", "requested_size": "S"}, CONFIRM])

# 21. Expired line (31 days), exchange asked without a size while 2+ sizes are valid: deny, not clarify.
add("adv21_expired_exchange_no_size", "K1",
    state([order("Z-521", "K1", "2027-01-10", [line("A1", "POLO-M", 5500)])]),
    [{"message": "I'd like to exchange line A1 of order Z-521 for a different size."}],
    [{"kind": "exchange", "order_id": "Z-521", "line_id": "A1"}])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"contract_version": 1, "cases": [c for c, _ in CASES]}
    (OUT / "adversarial_v1.json").write_text(json.dumps(doc, indent=2) + "\n")
    (OUT / "adversarial_intents.json").write_text(
        json.dumps({c["case_id"]: i for c, i in CASES}, indent=2) + "\n")
    # Oracle labels in the evals.score format (the oracle may share a misreading; see business.md).
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    from evals.oracle import build_expectation
    labels = {"labels_version": "review-business-1", "cases": [
        {"case_id": c["case_id"], "family": c["case_id"].split("_", 1)[0], "intents": i,
         "expected": build_expectation(c, i)} for c, i in CASES]}
    (OUT / "adversarial_labels.json").write_text(json.dumps(labels, indent=1) + "\n")
    print(f"wrote {len(CASES)} cases")


if __name__ == "__main__":
    main()
