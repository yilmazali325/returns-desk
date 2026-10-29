"""Business-rules review (tests/review/business): adversarial contract cases through the REAL
orchestrator, gateway and store, with the deterministic FakePolicyLLM (no model, no server).

Two kinds of tests:
- guards that hold today (window edges, duplicate prevention across request ids, timeout
  semantics, stock arithmetic, exact-shape final_state) -- evidence the core is right;
- repros of findings, marked ``xfail(strict=True)``: they assert the behaviour the contract asks
  for and fail today. When a fix lands, the strict xfail turns into an XPASS failure, which is
  the signal to delete the marker. See docs/review/business.md for each finding.

Run:  .venv/bin/pytest tests/review/business -q -rxX
Rebuild the cases:  .venv/bin/python tests/review/business/biz_build_cases.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from biz_harness import run, summary  # noqa: E402
from desk import policy  # noqa: E402
from desk.audit import AuditLog  # noqa: E402
from desk.faults import FaultPlan  # noqa: E402
from desk.llm import FakeLLM, RawCall, interpret_calls  # noqa: E402
from desk.profiles import FINAL  # noqa: E402
from desk.session import run_case  # noqa: E402
from desk.store import Store  # noqa: E402
from desk.testing.fake_policy_llm import FakePolicyLLM  # noqa: E402
from desk.tools import BusinessTools  # noqa: E402
from evals.oracle import build_expectation  # noqa: E402
from evals.score import score_case  # noqa: E402

CASES = {c["case_id"]: c for c in json.loads((HERE / "cases" / "adversarial_v1.json").read_text())["cases"]}
INTENTS = json.loads((HERE / "cases" / "adversarial_intents.json").read_text())
LABELS = {lb["case_id"]: lb for lb in json.loads((HERE / "cases" / "adversarial_labels.json").read_text())["cases"]}
PUBLIC = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]}
STATE_KEYS = ["as_of", "currency", "products", "orders", "returns", "refunds", "exchanges", "quarantine", "idempotency"]
YES = {"message": "Yes, I confirm the proposal.", "confirm_proposal": "last"}


def case(case_id: str) -> dict[str, Any]:
    return copy.deepcopy(CASES[case_id])


def public(case_id: str, turns: list[Any] | None = None, faults: list[Any] | None = None) -> dict[str, Any]:
    c = copy.deepcopy(PUBLIC[case_id])
    if turns is not None:
        c["turns"] = [t if isinstance(t, dict) else {"message": t} for t in turns]
    if faults is not None:
        c["faults"] = faults
    return c


def strict(case_id: str) -> tuple[bool, list[str]]:
    c = case(case_id)
    score = score_case(c, LABELS[case_id], run(c))
    return score.strict, score.diff_lines()


def stock(result: dict[str, Any]) -> dict[str, int]:
    return {p["sku"]: p["stock"] for p in result["final_state"]["products"]}


# ================================================================================================
# The case file itself is contract-shaped (same schema as candidate_assets/public_cases.json)


def test_adversarial_cases_use_the_contract_input_schema() -> None:
    assert len(CASES) >= 10
    for c in CASES.values():
        assert set(c) == {"case_id", "customer_id", "request_id", "initial_state", "turns", "faults"}
        assert list(c["initial_state"])[:9] == STATE_KEYS
        for o in c["initial_state"]["orders"]:
            assert {"order_id", "customer_id", "status", "delivered_on", "lines"} <= set(o)
            for ln in o["lines"]:
                assert {"line_id", "sku", "paid_cents", "unopened", "final_sale", "returned"} <= set(ln)
        assert all(set(t) <= {"message", "confirm_proposal"} for t in c["turns"])


# ================================================================================================
# Guards that hold today (desk == oracle == contract reading)

AGREE = [
    "adv01_day30_timestamp", "adv02_day0_same_day_timestamp", "adv03_same_sku_two_lines_one_returned",
    "adv04_exchange_timeout_replay", "adv05_original_sku_stock_change", "adv06_stale_then_other_size",
    "adv07_timeout_fires_on_first_real_commit", "adv08_numeric_half_size_exchange", "adv12_extra_fields_odd_cents",
    "adv14_restock_then_exchange", "adv15_stock_change_after_commit", "adv16_exchange_price_mismatch_xl",
    "adv17_paid_zero_refund", "adv18_two_changes_same_turn", "adv21_expired_exchange_no_size",
]


@pytest.mark.parametrize("case_id", AGREE)
def test_case_scores_strictly_against_the_oracle(case_id: str) -> None:
    ok, diffs = strict(case_id)
    assert ok, diffs


def test_window_is_0_to_30_inclusive_from_as_of_with_timestamps() -> None:
    def elig(delivered_on: str) -> list[str]:
        st = {"as_of": "2027-03-01", "products": [{"sku": "A", "product_id": "P", "size": "M", "price_cents": 1, "stock": 1}],
              "orders": [{"order_id": "O", "customer_id": "C", "status": "delivered", "delivered_on": delivered_on,
                          "lines": [{"line_id": "L", "sku": "A", "paid_cents": 1, "unopened": True,
                                     "final_sale": False, "returned": False}]}]}
        return policy.eligibility(st, "C", "O", "L").reasons

    assert elig("2027-03-01") == []                       # day 0
    assert elig("2027-03-01T23:00:00Z") == []             # day 0, later the same day
    assert elig("2027-01-30") == []                       # day 30 (Feb 2027 has 28 days)
    assert elig("2027-01-29") == ["RETURN_WINDOW_EXPIRED"]  # day 31
    assert elig("2027-03-02") == ["DELIVERY_IN_FUTURE"]   # delivered after as_of
    assert elig("03/01/2027") == ["MISSING_DELIVERY_DATE"]  # non-ISO: fail closed


def test_exchange_timeout_moves_one_unit_and_never_restocks_the_original() -> None:
    result = run(case("adv04_exchange_timeout_replay"))
    assert [t["decision"] for t in result["turns"]] == ["propose", "completed", "completed"]
    assert stock(result)["POLO-S"] == 4 and stock(result)["POLO-M"] == 2  # 5 -> 4 once; original untouched
    fs = result["final_state"]
    assert len(fs["exchanges"]) == len(fs["returns"]) == len(fs["quarantine"]) == 1
    executes = [e for e in result["action_audit"] if e.get("kind") == "tool" and e.get("tool") == "execute_proposal"]
    assert len(executes) == 1 and executes[0]["error"]["code"] == "TIMEOUT" and executes[0]["committed_effects"]


def test_unknown_fields_and_key_order_round_trip() -> None:
    c = case("adv12_extra_fields_odd_cents")
    fs = run(c)["final_state"]
    assert list(fs) == list(c["initial_state"])
    assert fs["store_id"] == "S-9" and fs["products"][1]["color"] == "navy"
    order = fs["orders"][0]
    assert order["channel"] == "web" and order["lines"][0]["gift"] is True and order["lines"][0]["notes"] == {"x": 1}
    assert fs["refunds"][0]["amount_cents"] == 3333


def test_new_request_id_on_an_already_refunded_line_never_refunds_twice() -> None:
    first = run(public("public_01_refund_boundary"))
    assert first["final_state"]["refunds"][0]["amount_cents"] == 3000
    again = public("public_01_refund_boundary")
    again["request_id"] = "REQ-second-attempt"
    again["initial_state"] = copy.deepcopy(first["final_state"])
    second = run(again)
    assert [t["decision"] for t in second["turns"]] == ["deny", "deny"]
    assert len(second["final_state"]["refunds"]) == 1 and second["final_state"]["idempotency"].keys() == {
        "REQ-public_01_refund_boundary"}


def test_two_request_ids_racing_on_one_store_commit_once() -> None:
    """A proposal made under REQ-B before REQ-A committed must be refused at B's commit."""
    st = copy.deepcopy(PUBLIC["public_01_refund_boundary"]["initial_state"])
    store = Store(st)
    a = BusinessTools(store, "C100", "REQ-A", FaultPlan([]), AuditLog())
    b = BusinessTools(store, "C100", "REQ-B", FaultPlan([]), AuditLog())
    for tools in (a, b):
        tools.begin_turn(1)
        tools.set_target("O100", "L1", "refund")
        assert tools.create_proposal("refund").ok
    pa, pb = a.open_proposal()["proposal_id"], b.open_proposal()["proposal_id"]
    a.begin_turn(2), a.set_grant(pa)
    assert a.execute_proposal().ok
    b.begin_turn(2), b.set_grant(pb)
    refused = b.execute_proposal()
    assert not refused.ok and refused.code == "ALREADY_RETURNED"
    fs = store.export()
    assert len(fs["refunds"]) == 1 and list(fs["idempotency"]) == ["REQ-A"]


# ================================================================================================
# Finding B1 -- the confirm flag is discarded on plain confirmations that contain a negated or
# harmless cancel-type word ("no need to wait", "don't cancel", "instead of store credit").

NEGATED_CANCEL_CONFIRMATIONS = [
    "Yes please go ahead, no need to wait.",
    "Yes, go ahead. Don't cancel it.",
    "Yes, I confirm. Please don't stop the exchange.",
    "Yes, the proposal is right, no need to cancel anything.",
    "Yes, I accept, I won't change my mind.",
    "Yes, I don't want to wait any longer.",
    "Yes, please stop asking and just process it.",
]


@pytest.mark.parametrize("message", NEGATED_CANCEL_CONFIRMATIONS)
def test_confirmed_exchange_executes_despite_negated_cancel_words(message: str) -> None:
    c = public("public_02_exchange", ["Exchange line L1 in order O100 for the same tee in size L.",
                                      {"message": message, "confirm_proposal": "last"}])
    result = run(c)
    assert [t["decision"] for t in result["turns"]] == ["propose", "completed"]
    assert stock(result)["TEE-L"] == 2


def test_adv11_refund_confirmation_with_instead_of_store_credit() -> None:
    ok, diffs = strict("adv11_confirm_with_instead_wording")
    assert ok, diffs


def test_adv20_confirmation_no_need_to_wait() -> None:
    ok, diffs = strict("adv20_confirm_no_need_to_wait")
    assert ok, diffs


# ================================================================================================
# Finding B2 -- "return ... and send me a large" becomes a REFUND; the code gate even overrides a
# model that correctly chose the exchange.


def test_adv19_return_and_send_me_a_large_is_an_exchange() -> None:
    ok, diffs = strict("adv19_return_and_send_large")
    assert ok, diffs


class _ExchangeFirstLLM:
    """Resolution hands off the CORRECT exchange (twice); everything else is the fake policy."""

    protocol = "fake"
    engine = "review double"

    def __init__(self) -> None:
        self.policy = FakePolicyLLM()
        self.exchange_handoffs = 0

    def step(self, role, messages, tools, allowed, max_tokens):  # same signature as LLMClient.step
        if role == "resolution" and "handoff_to_operations" in allowed and self.exchange_handoffs < 2:
            self.exchange_handoffs += 1
            call = RawCall("handoff_to_operations", {"intent": "propose_exchange", "order_id": "O100", "line_id": "L1",
                                                     "requested_size": "L", "requested_product": "SAME"})
            step = interpret_calls([call], tools, allowed)
            step.role = role
            return step
        return self.policy.step(role, messages, tools, allowed, max_tokens)


def test_gate_does_not_turn_a_correct_exchange_handoff_into_a_refund() -> None:
    c = public("public_02_exchange", ["Return L1 of O100 and send me a large."])
    result = run_case(c, _ExchangeFirstLLM(), FINAL)
    gates = [e["error"]["code"] for e in result["action_audit"] if e["kind"] == "gate"]
    proposal = result["turns"][0]["proposal"]
    assert gates == [] and proposal["remedy"] == "exchange" and proposal["replacement_sku"] == "TEE-L", (gates, proposal)


# ================================================================================================
# Finding B3 -- an INELIGIBLE line with "exchange for a different size" is not forced to a deny:
# a model that asks "which size?" is accepted, and the reply lists sizes "available for an exchange".


def test_ineligible_exchange_without_size_is_denied_even_when_the_model_asks_for_a_size() -> None:
    c = public("public_03_expired", ["I'd like to exchange line L1 of order O100 for a different size."])
    llm = FakeLLM({"resolution": [("reply_to_customer", {"message": "Which size would you like instead?"})] * 4,
                   "operations": []})
    result = run_case(c, llm, FINAL)
    expected = build_expectation(c, [{"kind": "exchange", "order_id": "O100", "line_id": "L1"}])["decisions"]
    turn = result["turns"][0]
    assert expected == ["deny"]
    assert turn["decision"] == "deny" and "available for an exchange" not in turn["response"], turn


# ================================================================================================
# Lower-impact findings and places where the oracle shares (or makes) a misreading


def test_adv09_order_id_reused_by_two_customers() -> None:
    result = run(case("adv09_order_id_collision_across_customers"))
    assert [t["decision"] for t in result["turns"]] == ["propose", "completed"], result["turns"][1]["response"]


def test_oracle_adv09_grades_the_customers_own_order() -> None:
    c = case("adv09_order_id_collision_across_customers")
    assert build_expectation(c, INTENTS[c["case_id"]])["decisions"] == ["propose", "completed"]


def test_adv13_desk_and_oracle_agree_on_status_spelling() -> None:
    c = case("adv13_status_case")
    desk = [t["decision"] for t in run(c)["turns"]]
    oracle = build_expectation(c, INTENTS[c["case_id"]])["decisions"]
    assert desk == oracle, (desk, oracle)


def test_adv10_exchange_request_after_completed_refund_is_not_reported_as_completed() -> None:
    c = case("adv10_after_refund_asks_exchange_same_line")
    result = run(c)
    # What the gateway itself says for that target once the refund is completed:
    store = Store(result["final_state"])
    tools = BusinessTools(store, c["customer_id"], c["request_id"], FaultPlan([]), AuditLog())
    tools.begin_turn(1)
    tools.set_target("Z-510", "A1", "exchange", requested_size="S")
    gateway = tools.create_proposal("exchange")
    assert gateway.code == "REQUEST_ALREADY_COMPLETED"
    assert result["turns"][2]["decision"] == "deny", summary(result)["responses"][2]
