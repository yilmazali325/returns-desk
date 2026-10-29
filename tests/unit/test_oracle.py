"""Tests for the independent oracle, the public expectations and the variant generator.

The oracle is the yardstick for every benchmark number, so it is tested on its own terms: each
disclosed rule and default (success and refusal paths), the hand-written public expectations, and
the generated suite (determinism, contract shape, validator compatibility, paraphrase variety,
metamorphic invariance).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pytest's rootdir is not importable by default in this repo
    sys.path.insert(0, str(ROOT))

from evals import oracle  # noqa: E402
from evals.oracle import (  # noqa: E402
    build_expectation, exchange_choice, expected_outcome, faults_only_state, line_eligibility, resolve_size,
    size_rank, validate_intent,
)
from evals.reference import main as reference_main, reference_results  # noqa: E402
from evals.variants import generate  # noqa: E402
from evals.variants.metamorphic import rename_ids, shift_dates  # noqa: E402

PUBLIC_CASES = ROOT / "candidate_assets" / "public_cases.json"
PUBLIC_EXPECTED = ROOT / "evals" / "expected" / "public_expected.json"
VARIANTS = ROOT / "evals" / "variants" / "variants_v1.json"
LABELS = ROOT / "evals" / "variants" / "labels_v1.json"


# --------------------------------------------------------------------------- fixtures

def catalog() -> list[dict]:
    return [
        {"sku": "TEE-S", "product_id": "TEE", "size": "S", "price_cents": 4000, "stock": 4},
        {"sku": "TEE-M", "product_id": "TEE", "size": "M", "price_cents": 4000, "stock": 2},
        {"sku": "TEE-L", "product_id": "TEE", "size": "L", "price_cents": 4000, "stock": 3},
        {"sku": "TEE-XL", "product_id": "TEE", "size": "XL", "price_cents": 4500, "stock": 5},
        {"sku": "JACKET-M", "product_id": "JACKET", "size": "M", "price_cents": 9000, "stock": 1},
        {"sku": "JACKET-L", "product_id": "JACKET", "size": "L", "price_cents": 9000, "stock": 0},
    ]


def state(delivered_days: int = 10, **line_flags) -> dict:
    as_of = date(2026, 10, 26)
    line = {"line_id": "LN-3", "sku": "TEE-M", "paid_cents": 2999, "unopened": True, "final_sale": False, "returned": False}
    line.update(line_flags)
    return {
        "as_of": as_of.isoformat(), "currency": "USD", "products": catalog(),
        "orders": [
            {"order_id": "ORD-7Q", "customer_id": "CU-1", "status": "delivered",
             "delivered_on": (as_of - timedelta(days=delivered_days)).isoformat(),
             "lines": [line, {"line_id": "LN-4", "sku": "JACKET-M", "paid_cents": 9000, "unopened": True,
                              "final_sale": False, "returned": False}]},
            {"order_id": "ORD-8Z", "customer_id": "CU-2", "status": "delivered", "delivered_on": "2026-10-20",
             "lines": [{"line_id": "LN-3", "sku": "TEE-S", "paid_cents": 4000, "unopened": True,
                        "final_sale": False, "returned": False}]},
        ],
        "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {},
    }


def case(turns: list[tuple[str, bool]], faults: list | None = None, **state_kwargs) -> dict:
    return {"case_id": "t", "customer_id": "CU-1", "request_id": "REQ-t", "initial_state": state(**state_kwargs),
            "turns": [{"message": m, **({"confirm_proposal": "last"} if flag else {})} for m, flag in turns],
            "faults": faults or []}


REFUND = {"kind": "refund", "order_id": "ORD-7Q", "line_id": "LN-3"}
CONFIRM = {"kind": "confirm"}


def exchange(**requested) -> dict:
    return {"kind": "exchange", "order_id": "ORD-7Q", "line_id": "LN-3", **requested}


def run(intents: list[dict], flags: list[bool], faults: list | None = None, **state_kwargs):
    return expected_outcome(case([("m", f) for f in flags], faults, **state_kwargs), intents)


def decisions(outcome) -> list[str]:
    return [t.decision for t in outcome.turns]


# --------------------------------------------------------------------------- pure rules

@pytest.mark.parametrize("days, verdict", [(0, "ok"), (30, "ok"), (31, "deny"), (-1, "deny")])
def test_window_is_zero_to_thirty_inclusive(days, verdict):
    assert line_eligibility(state(days), "CU-1", "ORD-7Q", "LN-3")[0] == verdict


@pytest.mark.parametrize("as_of, delivered, days", [
    ("2028-03-01", "2028-01-31", 30), ("2028-03-01", "2028-01-30", 31), ("2026-03-01", "2026-01-29", 31),
    ("2027-01-05", "2026-12-06", 30), ("2026-11-30", "2026-10-31", 30), ("2026-10-26T12:00:00", "2026-10-26", 0),
])
def test_calendar_boundaries(as_of, delivered, days):
    assert oracle.days_since_delivery(as_of, delivered) == days


@pytest.mark.parametrize("value", [None, "", "not-a-date", "2026-13-01"])
def test_missing_or_invalid_delivery_date(value):
    s = state()
    s["orders"][0]["delivered_on"] = value
    assert line_eligibility(s, "CU-1", "ORD-7Q", "LN-3") == ("deny", "MISSING_DELIVERY_DATE")


@settings(max_examples=200, deadline=None)
@given(st.dates(min_value=date(2020, 1, 1), max_value=date(2035, 12, 31)), st.integers(min_value=-40, max_value=400))
def test_eligibility_depends_only_on_the_day_difference(as_of, days):
    s = state()
    s["as_of"] = as_of.isoformat()
    s["orders"][0]["delivered_on"] = (as_of - timedelta(days=days)).isoformat()
    assert (line_eligibility(s, "CU-1", "ORD-7Q", "LN-3")[0] == "ok") == (0 <= days <= 30)


@pytest.mark.parametrize("flags, reason", [
    ({"unopened": False}, "OPENED"), ({"unopened": 1}, "OPENED"), ({"final_sale": True}, "FINAL_SALE"),
    ({"final_sale": None}, "FINAL_SALE"), ({"returned": True}, "ALREADY_RETURNED"),
    ({"unopened": False, "final_sale": True}, "OPENED+FINAL_SALE"),
])
def test_line_flags_are_checked_strictly(flags, reason):
    assert line_eligibility(state(**flags), "CU-1", "ORD-7Q", "LN-3") == ("deny", reason)


def test_foreign_and_missing_orders_are_indistinguishable():
    s = state()
    assert line_eligibility(s, "CU-1", "ORD-8Z", "LN-3") == line_eligibility(s, "CU-1", "NOPE", "LN-3") == ("deny", "ORDER_NOT_FOUND")


def test_unknown_line_in_owned_order_asks_to_clarify():
    assert line_eligibility(state(), "CU-1", "ORD-7Q", "LN-99") == ("clarify", "LINE_NOT_FOUND")


def test_not_delivered_status():
    s = state()
    s["orders"][0].update(status="shipped", delivered_on=None)
    assert line_eligibility(s, "CU-1", "ORD-7Q", "LN-3") == ("deny", "NOT_DELIVERED+MISSING_DELIVERY_DATE")


def test_size_ranking_and_relative_sizes():
    assert size_rank("xs") < size_rank("M") < size_rank("XL")
    assert size_rank("9") < size_rank("10") < size_rank("ONE-SIZE")
    family = [p for p in catalog() if p["product_id"] == "TEE"]
    medium = family[1]
    assert resolve_size("UP", medium, family) == "L"
    assert resolve_size("down", medium, family) == "S"
    assert resolve_size("UP", family[3], family) is None  # XL is the largest offered
    assert resolve_size("DOWN", family[0], family) is None
    assert resolve_size("l", medium, family) == "l"


def line_of(sku: str = "TEE-M") -> dict:
    return {"line_id": "x", "sku": sku, "paid_cents": 1, "unopened": True, "final_sale": False, "returned": False}


@pytest.mark.parametrize("requested, expected", [
    ({"requested_size": "L"}, ("propose", "EXCHANGE_OK", "TEE-L")),
    ({"requested_size": "s"}, ("propose", "EXCHANGE_OK", "TEE-S")),
    ({"requested_size": "UP"}, ("propose", "EXCHANGE_OK", "TEE-L")),
    ({"requested_size": "M"}, ("deny", "SAME_SIZE", None)),
    ({"requested_size": "XXL"}, ("deny", "SIZE_NOT_OFFERED", None)),
    ({"requested_size": "XL"}, ("deny", "PRICE_MISMATCH", None)),
    ({"requested_sku": "JACKET-M"}, ("deny", "DIFFERENT_PRODUCT", None)),
    ({"requested_sku": "NOPE-1"}, ("deny", "SKU_NOT_FOUND", None)),
    ({"requested_product": "JACKET"}, ("deny", "DIFFERENT_PRODUCT", None)),
    ({"requested_product": "TEE", "requested_size": "S"}, ("propose", "EXCHANGE_OK", "TEE-S")),
    ({}, ("clarify", "SIZE_UNSPECIFIED", None)),
])
def test_exchange_choice(requested, expected):
    assert exchange_choice(state(), line_of(), requested) == expected


def test_exchange_choice_stock_and_option_counts():
    s = state()
    by_sku = {p["sku"]: p for p in s["products"]}
    by_sku["TEE-L"]["stock"] = 0
    assert exchange_choice(s, line_of(), {"requested_size": "L"}) == ("deny", "OUT_OF_STOCK", None)
    assert exchange_choice(s, line_of(), {}) == ("propose", "EXCHANGE_OK", "TEE-S")  # exactly one valid size
    by_sku["TEE-S"]["stock"] = "4"  # a non-integer stock is never treated as available
    assert exchange_choice(s, line_of(), {}) == ("deny", "NO_VALID_SIZE", None)
    assert exchange_choice(s, line_of("GONE-M"), {"requested_size": "L"}) == ("deny", "SKU_NOT_FOUND", None)
    assert exchange_choice(s, line_of("TEE-XL"), {"requested_size": "UP"}) == ("deny", "SIZE_NOT_OFFERED", None)


# --------------------------------------------------------------------------- conversation semantics

def test_refund_happy_path_writes_every_record():
    outcome = run([REFUND, CONFIRM], [False, True])
    assert decisions(outcome) == ["propose", "completed"]
    assert outcome.turns[0].proposal["amount_cents"] == 2999  # paid_cents, not the 4000 catalog price
    final = outcome.final_state
    assert [len(final[k]) for k in ("returns", "refunds", "exchanges", "quarantine")] == [1, 1, 0, 1]
    assert final["orders"][0]["lines"][0]["returned"] is True
    assert final["idempotency"]["REQ-t"]["status"] == "completed"
    assert [p["stock"] for p in final["products"]] == [p["stock"] for p in catalog()]


def test_exchange_moves_only_the_replacement_stock():
    outcome = run([exchange(requested_size="S"), CONFIRM], [False, True])
    stock = {p["sku"]: p["stock"] for p in outcome.final_state["products"]}
    assert decisions(outcome) == ["propose", "completed"]
    assert (stock["TEE-S"], stock["TEE-M"]) == (3, 2)
    assert outcome.final_state["exchanges"][0]["replacement_sku"] == "TEE-S"
    assert outcome.final_state["refunds"] == []


def test_verbal_yes_re_presents_the_same_proposal_without_mutation():
    outcome = run([REFUND, {"kind": "verbal_yes"}], [False, False])
    assert decisions(outcome) == ["propose", "propose"]
    assert outcome.turns[0].proposal == outcome.turns[1].proposal
    assert outcome.final_state == state()


def test_confirm_without_a_proposal_re_reads_the_standing_request():
    assert decisions(run([REFUND, CONFIRM], [False, True], delivered_days=40)) == ["deny", "deny"]
    assert decisions(run([{"kind": "clarify_needed"}, CONFIRM], [False, True])) == ["clarify", "clarify"]
    first_turn = run([REFUND], [True])  # flag on turn 1: nothing earlier to resolve to
    assert decisions(first_turn) == ["propose"] and first_turn.final_state == state()


def test_change_of_mind_supersedes_and_only_the_latest_commits():
    outcome = run([REFUND, exchange(requested_size="L"), CONFIRM], [False, False, True])
    assert decisions(outcome) == ["propose", "propose", "completed"]
    assert [t.proposal["seq"] for t in outcome.turns[:2]] == [1, 2]
    assert [p["status"] for p in outcome.proposals] == ["superseded", "executed"]
    assert outcome.final_state["refunds"] == [] and len(outcome.final_state["exchanges"]) == 1


def test_identical_restatement_keeps_the_proposal_id():
    outcome = run([REFUND, REFUND], [False, False])
    assert [t.reason for t in outcome.turns] == ["PROPOSED", "REPRESENTED"]
    assert len(outcome.proposals) == 1


@pytest.mark.parametrize("second, expected", [
    (exchange(requested_size="S"), ["propose", "propose"]),  # flagged change: re-propose, no commit
    (exchange(), ["propose", "completed"]),  # restating "the exchange" executes the bound proposal
    (exchange(requested_size="L"), ["propose", "completed"]),  # restating the bound size executes
    ({"kind": "refund", "order_id": "ORD-7Q", "line_id": "LN-3"}, ["propose", "propose"]),  # remedy change
    ({"kind": "exchange", "order_id": "ORD-7Q", "line_id": "LN-4", "requested_size": "L"}, ["propose", "deny"]),
    ({"kind": "status"}, ["propose", "completed"]),  # any other wording executes (CLARIFICATIONS #9)
    ({"kind": "clarify_needed"}, ["propose", "completed"]),
])
def test_flagged_turn_executes_unless_the_text_changes_the_request(second, expected):
    assert decisions(run([exchange(requested_size="L"), second], [False, True])) == expected


def test_cancel_with_flag_blocks_execution():
    c = case([("refund it", False), ("no, cancel", True)])
    expectation = build_expectation(c, [REFUND, {"kind": "cancel"}])
    assert expectation["decisions"] == ["propose", "clarify"]
    assert expectation["state_unchanged"] is True
    assert expectation["alternatives"] == {"2": ["deny"]}


def test_replays_and_request_already_completed():
    turns = [REFUND, CONFIRM, CONFIRM, REFUND, {"kind": "verbal_yes"}, {"kind": "status"},
             {"kind": "refund", "order_id": "ORD-7Q", "line_id": "LN-4"}]
    outcome = run(turns, [False, True, True, False, False, False, False])
    assert decisions(outcome) == ["propose", "completed", "completed", "completed", "completed", "completed", "deny"]
    assert outcome.turns[-1].reason == "REQUEST_ALREADY_COMPLETED"
    assert len(outcome.final_state["refunds"]) == 1


def test_timeout_after_commit_fires_once_and_is_reconciled():
    expectation = build_expectation(case([("a", False), ("b", True), ("c", True)], [{"type": "timeout_after_commit"}]),
                                    [REFUND, CONFIRM, CONFIRM])
    assert expectation["decisions"] == ["propose", "completed", "completed"]
    assert expectation["reasons"][1:] == ["COMMITTED_TIMEOUT_RECONCILED", "REPLAY"]
    assert expectation["alternatives"] == {"2": ["error"]}
    assert len(expectation["final_business_state"]["refunds"]) == 1


def test_stale_stock_invalidates_and_a_reconfirm_is_not_open():
    faults = [{"type": "stock_change", "before_turn": 2, "sku": "TEE-L", "new_stock": 0}]
    outcome = run([exchange(requested_size="L"), CONFIRM, CONFIRM], [False, True, True], faults)
    assert [t.reason for t in outcome.turns] == ["PROPOSED", "OUT_OF_STOCK", "PROPOSAL_NOT_OPEN"]
    assert outcome.events[0] == {"type": "stock_change", "sku": "TEE-L", "old_stock": 3, "new_stock": 0,
                                 "before_turn": 2, "applied": True}
    # Without an open proposal a bare "yes" re-reads the standing request against today's stock.
    assert decisions(run([exchange(requested_size="L"), CONFIRM, {"kind": "verbal_yes"}], [False, True, False], faults)) \
        == ["propose", "deny", "deny"]


def test_revalidation_rechecks_eligibility_not_only_stock():
    conversation = oracle._Conversation(case([("a", False), ("b", True)]))
    conversation.handle(REFUND, False)
    conversation.state["orders"][0]["lines"][0]["returned"] = True  # simulate an out-of-band return
    assert conversation.handle(CONFIRM, True).reason == "ALREADY_RETURNED"


def test_fault_edge_cases():
    faults = [{"type": "stock_change", "before_turn": 1, "sku": "NOPE", "new_stock": 9},
              {"type": "stock_change", "before_turn": 5, "sku": "TEE-L", "new_stock": 0},  # never reached
              {"type": "mystery"},
              {"type": "stock_change", "before_turn": 2, "sku": "TEE-L", "new_stock": 0},
              {"type": "stock_change", "before_turn": 2, "sku": "TEE-L", "new_stock": 2}]  # list order: last wins
    outcome = run([exchange(requested_size="L"), CONFIRM], [False, True], faults)
    assert decisions(outcome) == ["propose", "completed"]
    assert [e["applied"] for e in outcome.events] == [False, True, True]
    assert {p["sku"]: p["stock"] for p in outcome.final_state["products"]}["TEE-L"] == 1
    baseline = {p["sku"]: p["stock"] for p in faults_only_state(case([("a", False), ("b", True)], faults))["products"]}
    assert baseline["TEE-L"] == 2


@pytest.mark.parametrize("intent, flag, message", [
    ({"kind": "teleport"}, False, "unknown intent kind"),
    ({"kind": "confirm"}, False, "needs confirm_proposal"),
    ({"kind": "verbal_yes"}, True, "must not carry"),
    ({"kind": "refund", "line_id": "LN-3"}, False, "needs an order_id"),
])
def test_label_mistakes_are_rejected(intent, flag, message):
    with pytest.raises(ValueError, match=message):
        validate_intent(intent, flag)


def test_intent_count_must_match_turns():
    with pytest.raises(ValueError, match="1 intents for 2 turns"):
        expected_outcome(case([("a", False), ("b", True)]), [REFUND])


# --------------------------------------------------------------------------- public expectations

def test_hand_written_public_expectations_match_the_oracle():
    cases = {c["case_id"]: c for c in json.loads(PUBLIC_CASES.read_text())["cases"]}
    labels = json.loads(PUBLIC_EXPECTED.read_text())["cases"]
    assert [label["case_id"] for label in labels] == list(cases)
    for label in labels:
        assert build_expectation(cases[label["case_id"]], label["intents"]) == label["expected"], label["case_id"]


def test_public_expectations_pin_the_headline_outcomes():
    labels = {label["case_id"]: label["expected"] for label in json.loads(PUBLIC_EXPECTED.read_text())["cases"]}
    assert labels["public_01_refund_boundary"]["final_business_state"]["refunds"][0]["amount_cents"] == 3000
    assert {p["sku"]: p["stock"] for p in labels["public_02_exchange"]["final_business_state"]["products"]}["TEE-L"] == 2
    assert labels["public_03_expired"]["state_unchanged"] and labels["public_05_stale_stock"]["state_unchanged"]
    assert labels["public_04_clarify"]["decisions"] == ["clarify", "propose", "completed"]
    assert labels["public_06_timeout_replay"]["decisions"] == ["propose", "completed", "completed"]


# --------------------------------------------------------------------------- generated suite

@pytest.fixture(scope="module")
def suite() -> tuple[dict, dict]:
    return generate.build_suite()


def test_generation_is_deterministic_and_the_committed_files_are_current(suite):
    cases, labels = suite
    again_cases, again_labels = generate.build_suite()
    assert generate.dumps(cases) == generate.dumps(again_cases) == VARIANTS.read_text()
    assert generate.dumps(labels) == generate.dumps(again_labels) == LABELS.read_text()


def test_suite_size_and_paraphrase_variety(suite):
    cases, labels = suite
    assert len(cases["cases"]) >= 150
    first_messages: dict[str, set[str]] = {}
    for c, label in zip(cases["cases"], labels["cases"]):
        assert c["case_id"] == label["case_id"]
        first_messages.setdefault(label["family"], set()).add(c["turns"][0]["message"])
    thin = {family: len(m) for family, m in first_messages.items() if len(m) < 3}
    assert not thin, f"families with fewer than 3 different phrasings: {thin}"
    required = {"refund_day0", "refund_mid_window", "refund_day30", "refund_expired", "future_delivery", "opened",
                "final_sale", "already_returned", "not_delivered", "foreign_order", "nonexistent_order",
                "nonexistent_line", "exchange_valid", "exchange_out_of_stock", "exchange_price_mismatch",
                "exchange_same_size", "exchange_different_product", "exchange_no_size_multi",
                "exchange_no_size_single", "exchange_relative_size", "clarify_multi_line", "clarify_multi_order",
                "description_unique", "description_ambiguous", "stale_stock_to_zero", "stale_stock_positive",
                "stock_change_refund", "timeout_refund_replay", "timeout_exchange_replay", "duplicate_confirm",
                "verbal_yes_no_flag", "confirm_without_proposal", "change_refund_to_exchange", "cancel_with_flag",
                "injection_eligible", "injection_ineligible", "claims_contradict_state", "multi_customer_state",
                "long_conversation", "metamorphic_rename", "metamorphic_dateshift"}
    assert required <= set(first_messages)


def test_cases_follow_the_contract_input_schema(suite):
    cases, labels = suite
    assert cases["contract_version"] == 1
    assert len({c["case_id"] for c in cases["cases"]}) == len({c["request_id"] for c in cases["cases"]}) == len(cases["cases"])
    for c, label in zip(cases["cases"], labels["cases"]):
        assert list(c) == ["case_id", "customer_id", "request_id", "initial_state", "turns", "faults"]
        assert tuple(c["initial_state"]) == generate.STATE_KEYS
        assert all(set(t) <= {"message", "confirm_proposal"} and t.get("confirm_proposal", "last") == "last"
                   for t in c["turns"])
        view = label["expected"]["final_business_state"]
        assert len(view["refunds"]) + len(view["exchanges"]) <= 1  # "a case concerns at most 1 refund or exchange"
        assert 1 <= len(c["turns"]) <= 6


def test_variant_inputs_pass_the_supplied_validator(tmp_path, suite):
    spec = importlib.util.spec_from_file_location("validate_submission", ROOT / "candidate_assets" / "validate_submission.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    output = tmp_path / "reference.json"
    assert reference_main(["--input", str(VARIANTS), "--labels", str(LABELS), "--output", str(output)]) == 0
    assert validator.check(str(VARIANTS), str(output)) == len(suite[0]["cases"])


def test_reference_output_mirrors_the_oracle():
    document = reference_results(PUBLIC_CASES.read_bytes(), json.loads(PUBLIC_EXPECTED.read_text()))
    first = document["results"][0]
    assert first["turns"][0]["proposal"]["proposal_id"] == "ORACLE-P-1"
    assert first["turns"][1]["proposal"] is None
    stale = document["results"][4]
    assert stale["external_events"][0]["event_id"] == "ORACLE-EV-1"


def test_metamorphic_copies_keep_every_decision(suite):
    cases, labels = suite
    by_id = {label["case_id"]: label for label in labels["cases"]}
    case_by_id = {c["case_id"]: c for c in cases["cases"]}
    for label in labels["cases"]:
        if not label["family"].startswith("metamorphic_"):
            continue
        source_id = label["note"].split(" from ")[-1]
        assert label["expected"]["decisions"] == by_id[source_id]["expected"]["decisions"]
        if label["family"] == "metamorphic_rename":
            source = case_by_id[source_id]
            text = json.dumps(case_by_id[label["case_id"]])
            for order in source["initial_state"]["orders"]:
                assert f'"{order["order_id"]}"' not in text and f'"{order["customer_id"]}"' not in text


def test_rename_and_shift_transforms_directly():
    c = case([("Refund ln-3 of ORD-7Q, I'm CU-1", False), ("yes", True)],
             [{"type": "stock_change", "before_turn": 2, "sku": "TEE-M", "new_stock": 1}])
    renamed, intents = rename_ids(c, [REFUND, CONFIRM], salt=7, new_case_id="r")
    assert renamed["turns"][0]["message"] == "Refund ln7-1 of ORD-7001, I'm K701"
    assert renamed["faults"][0]["sku"] == renamed["initial_state"]["products"][1]["sku"]
    assert intents[0]["order_id"] == "ORD-7001"
    assert renamed["initial_state"]["products"][0]["product_id"] == "BASIC-TEE"
    assert build_expectation(renamed, intents)["decisions"] == build_expectation(c, [REFUND, CONFIRM])["decisions"]
    shifted = shift_dates(c, -400, "s")
    assert shifted["initial_state"]["as_of"] == "2025-09-21"
    assert build_expectation(shifted, [REFUND, CONFIRM])["decisions"] == ["propose", "completed"]


def test_generator_refuses_a_case_whose_oracle_outcome_disagrees():
    builder = generate.SuiteBuilder()
    with pytest.raises(AssertionError, match="family expects"):
        builder.add("broken", customer="CU-1", as_of=date(2026, 10, 26), catalog=catalog(),
                    orders=copy.deepcopy(state(40)["orders"]),
                    turns=[generate.say("refund LN-3 of ORD-7Q", "refund", order_id="ORD-7Q", line_id="LN-3")],
                    expect=["propose"])


def test_generator_cli_writes_both_files(tmp_path, capsys):
    assert generate.main(["--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "variants_v1.json").read_text() == VARIANTS.read_text()
    assert (tmp_path / "labels_v1.json").read_text() == LABELS.read_text()
    assert "cases in" in capsys.readouterr().out


def test_text_helpers():
    assert generate.fmt_id("O100", "hash") == "#O100" and generate.fmt_id("O100", "lower") == "o100"
    assert generate.size_phrase("XL", 1) == "an extra large" and generate.size_phrase("32", 1) == "a size 32"
    assert generate.size_word("10") == "size 10" and generate.size_word("M") == "medium"
