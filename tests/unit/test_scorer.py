"""Tests for evals.score, evals.invariants and evals.state_view against hand-made adapter outputs.

The scorer is only trustworthy if it fails exactly the outputs a strict grader would fail. Each
test starts from a correct, hand-written result (with IDs deliberately different from the
oracle's) and breaks one thing: a decision, an amount, a duplicate refund, a stock effect, ...
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pytest's rootdir is not importable by default in this repo
    sys.path.insert(0, str(ROOT))

from evals.reference import reference_result, reference_results  # noqa: E402
from evals.score import build_report, main, render_text, score_case, score_run, wilson_interval  # noqa: E402
from evals.state_view import business_view, diff_views, snippet  # noqa: E402

PUBLIC_CASES = ROOT / "candidate_assets" / "public_cases.json"
PUBLIC_EXPECTED = ROOT / "evals" / "expected" / "public_expected.json"
VARIANTS = ROOT / "evals" / "variants" / "variants_v1.json"
LABELS = ROOT / "evals" / "variants" / "labels_v1.json"

CASES = {c["case_id"]: c for c in json.loads(PUBLIC_CASES.read_text())["cases"]}
LABELS_BY_ID = {label["case_id"]: label for label in json.loads(PUBLIC_EXPECTED.read_text())["cases"]}
P01, P02, P03, P05, P06 = ("public_01_refund_boundary", "public_02_exchange", "public_03_expired",
                           "public_05_stale_stock", "public_06_timeout_replay")


# --------------------------------------------------------------------------- hand-made outputs

def committed_state(case_id: str, remedy: str = "refund", replacement: str | None = None) -> dict:
    """What a correct desk writes for a refund/exchange of O100/L1, with the desk's own ID scheme."""
    case = CASES[case_id]
    final = copy.deepcopy(case["initial_state"])
    final["orders"][0]["lines"][0]["returned"] = True
    common = {"transaction_id": "TX-0001", "request_id": case["request_id"], "proposal_id": "P-0001",
              "customer_id": "C100", "order_id": "O100", "line_id": "L1", "sku": "TEE-M",
              "replacement_sku": replacement, "amount_cents": 3000 if remedy == "refund" else 0, "currency": "USD",
              "business_date": "2026-10-26", "status": "completed"}
    final["returns"] = [{"return_id": "RET-0001", "remedy": remedy, "disposition": "quarantine", **common}]
    if remedy == "refund":
        final["refunds"] = [{"refund_id": "RF-0001", "return_id": "RET-0001", **common}]
    else:
        final["exchanges"] = [{"exchange_id": "EX-0001", "return_id": "RET-0001", "original_sku": "TEE-M", **common}]
        next(p for p in final["products"] if p["sku"] == replacement)["stock"] -= 1
    final["quarantine"] = [{"quarantine_id": "Q-0001", "return_id": "RET-0001", "quantity": 1,
                            "reason": "customer_return", **common}]
    final["idempotency"] = {case["request_id"]: {"action": remedy, **common, "result": {"transaction_id": "TX-0001"}}}
    return final


def proposal(case_id: str, remedy: str = "refund", replacement: str | None = None, pid: str = "P-0001") -> dict:
    return {"proposal_id": pid, "request_id": CASES[case_id]["request_id"], "customer_id": "C100",
            "order_id": "O100", "line_id": "L1", "remedy": remedy, "original_sku": "TEE-M",
            "replacement_sku": replacement, "amount_cents": 3000 if remedy == "refund" else 0, "currency": "USD"}


def turn(decision: str, prop: dict | None = None) -> dict:
    return {"decision": decision, "response": "(wording is never scored)", "proposal": prop}


def result(case_id: str, turns: list[dict], final: dict, events: list | None = None) -> dict:
    return {"case_id": case_id, "turns": turns, "action_audit": [], "final_state": final,
            "external_events": events or [], "metrics": {"elapsed_ms": 1, "peak_ram_mb": None, "peak_vram_mb": None}}


def correct_01() -> dict:
    return result(P01, [turn("propose", proposal(P01)), turn("completed")], committed_state(P01))


def correct_06() -> dict:
    return result(P06, [turn("propose", proposal(P06)), turn("completed"), turn("completed")], committed_state(P06))


def score(res: dict | None, case_id: str = P01):
    return score_case(CASES[case_id], LABELS_BY_ID[case_id], res)


def names(s) -> set[str]:
    return {name for name, _ in s.violations}


# --------------------------------------------------------------------------- correct outputs

def test_correct_outputs_pass_strictly_whatever_the_generated_ids():
    for res, case_id in ((correct_01(), P01), (correct_06(), P06)):
        s = score(res, case_id)
        assert s.strict and s.lenient, s.diff_lines()
    exchange = result(P02, [turn("propose", proposal(P02, "exchange", "TEE-L")), turn("completed")],
                      committed_state(P02, "exchange", "TEE-L"))
    assert score(exchange, P02).strict
    stale_final = copy.deepcopy(CASES[P05]["initial_state"])
    stale_final["products"][2]["stock"] = 0
    events = [{"event_id": "EV-0001", "type": "stock_change", "sku": "TEE-L", "old_stock": 3, "new_stock": 0,
               "before_turn": 2, "applied": True, "source": "external"}]
    stale = result(P05, [turn("propose", proposal(P05, "exchange", "TEE-L")), turn("deny")], stale_final, events)
    assert score(stale, P05).strict


def test_record_order_does_not_matter():
    res = correct_01()
    final = res["final_state"]
    final["returns"] = list(reversed(final["returns"]))
    assert score(res).strict


# --------------------------------------------------------------------------- one defect at a time

def test_wrong_decision_fails_only_the_decision():
    res = correct_01()
    res["turns"][1]["decision"] = "error"
    s = score(res)
    assert not s.strict and not s.lenient
    assert s.state_diffs == [] and s.violations == []
    assert s.diff_lines() == ["turn 2: decision expected 'completed' (COMMITTED), got 'error'"]


def test_timeout_reported_as_error_is_lenient_but_not_strict():
    res = correct_06()
    res["turns"][1]["decision"] = "error"
    s = score(res, P06)
    assert not s.strict and s.lenient
    assert "also accepted leniently: ['error']" in s.diff_lines()[0]


def test_wrong_refund_amount():
    res = correct_01()
    res["final_state"]["refunds"][0]["amount_cents"] = 4000  # catalog price instead of paid_cents
    s = score(res)
    assert not s.strict
    assert "state.refunds[O100].amount_cents: expected 3000, got 4000" in s.state_diffs
    assert names(s) == {"refund_amount"}


def test_wrong_proposal_amount_and_missing_fields():
    res = correct_01()
    res["turns"][0]["proposal"]["amount_cents"] = 50000
    del res["turns"][0]["proposal"]["original_sku"]
    s = score(res)
    assert s.proposal_diffs == ["turn 1: original_sku expected 'TEE-M', got '<missing>'",
                                "turn 1: amount_cents expected 3000, got 50000"]
    assert not s.strict and s.state_diffs == []


def test_duplicate_refund():
    res = correct_01()
    res["final_state"]["refunds"].append(dict(res["final_state"]["refunds"][0], refund_id="RF-0002"))
    s = score(res)
    assert "duplicate_effects" in names(s)
    assert "state.refunds: expected 1 item(s), got 2" in s.state_diffs


def test_extra_stock_change_restocks_the_quarantined_original():
    res = correct_01()
    res["final_state"]["products"][1]["stock"] = 3  # TEE-M put back on the shelf
    s = score(res)
    assert names(s) == {"stock_effects"}
    assert "state.products[TEE-M].stock: expected 2, got 3" in s.state_diffs


def test_mutation_on_a_no_op_case():
    res = result(P03, [turn("deny")], committed_state(P03))
    s = score(res, P03)
    assert {"mutation_without_confirmation", "ineligible_return", "unexpected_mutation"} <= names(s)


def test_completed_without_any_record():
    res = result(P01, [turn("propose", proposal(P01)), turn("completed")], copy.deepcopy(CASES[P01]["initial_state"]))
    s = score(res)
    assert "completed_without_record" in names(s)


def test_proposal_object_only_on_propose_turns():
    res = correct_01()
    res["turns"][1]["proposal"] = proposal(P01)
    res["turns"][0]["proposal"] = None
    s = score(res)
    assert {"proposal_only_on_propose"} == names(s)
    assert s.proposal_diffs == ["turn 1: proposal object missing"]


def test_proposal_identity_is_checked_without_comparing_id_values():
    verbal = json.loads(LABELS.read_text())["cases"]
    labels = {label["case_id"]: label for label in verbal}
    cases = {c["case_id"]: c for c in json.loads(VARIANTS.read_text())["cases"]}
    case_id = "v1_verbal_yes_no_flag_01"
    res = reference_result(cases[case_id], labels[case_id]["intents"])
    assert score_case(cases[case_id], labels[case_id], res).strict
    res["turns"][1]["proposal"]["proposal_id"] = "P-NEW"  # re-presented under a fresh ID
    s = score_case(cases[case_id], labels[case_id], res)
    assert s.proposal_diffs == ["turn 2: expected the same proposal 'ORACLE-P-1' re-presented, got 'P-NEW'"]
    case_id = "v1_change_refund_to_exchange_01"
    res = reference_result(cases[case_id], labels[case_id]["intents"])
    res["turns"][1]["proposal"]["proposal_id"] = res["turns"][0]["proposal"]["proposal_id"]
    s = score_case(cases[case_id], labels[case_id], res)
    assert any("reused for a different binding" in line for line in s.proposal_diffs)
    res["turns"][1]["proposal"]["proposal_id"] = ""
    assert "turn 2: proposal_id missing" in score_case(cases[case_id], labels[case_id], res).proposal_diffs


def test_record_binding_and_exchange_policy():
    res = result(P02, [turn("propose", proposal(P02, "exchange", "TEE-L")), turn("completed")],
                 committed_state(P02, "exchange", "JACKET-M"))
    res["final_state"]["returns"][0]["proposal_id"] = None
    res["final_state"]["exchanges"][0]["customer_id"] = "C200"
    s = score(res, P02)
    assert {"exchange_policy", "record_binding"} <= names(s)
    unknown = result(P02, res["turns"], committed_state(P02, "exchange", "TEE-L"))
    unknown["final_state"]["exchanges"][0]["replacement_sku"] = "TEE-XXL"
    assert "exchange_policy" in names(score(unknown, P02))
    foreign = result(P02, res["turns"], committed_state(P02, "exchange", "TEE-L"))
    foreign["final_state"]["exchanges"][0]["order_id"] = "O999"  # a record for an order the customer does not have
    assert {"exchange_policy", "ineligible_return"} <= names(score(foreign, P02))


def test_orders_and_products_are_otherwise_immutable():
    res = correct_01()
    res["final_state"]["orders"][0]["lines"][0]["paid_cents"] = 1
    assert "orders_changed" in names(score(res))
    res = correct_01()
    res["final_state"]["products"][0]["price_cents"] = 1
    assert "products_changed" in names(score(res))
    res = correct_01()
    res["final_state"]["products"].pop()
    assert "stock_effects" in names(score(res))


def test_final_state_shape_problems():
    res = correct_01()
    res["final_state"]["extra"] = []
    s = score(res)
    assert "final_state_keys" in names(s) and any("top_level_keys" in d for d in s.state_diffs)
    res = correct_01()
    res["final_state"]["refunds"] = "oops"
    res["final_state"]["idempotency"] = []
    assert names(score(res)) == {"final_state_shape"}
    res = correct_01()
    res["final_state"] = None
    s = score(res)
    assert s.state_diffs == ["final_state missing or not an object"] and names(s) == {"final_state_shape"}


def test_events_must_record_applied_stock_changes():
    final = copy.deepcopy(CASES[P05]["initial_state"])
    final["products"][2]["stock"] = 0
    turns = [turn("propose", proposal(P05, "exchange", "TEE-L")), turn("deny")]
    missing = result(P05, turns, final)
    missing["external_events"] = None
    assert score(missing, P05).event_diffs == ["external_events missing; expected 1 stock_change event(s)"]
    wrong = result(P05, turns, final, [{"type": "stock_change", "sku": "TEE-L", "old_stock": 3, "new_stock": 1,
                                         "before_turn": 2, "applied": True}])
    assert score(wrong, P05).event_diffs == ["external_events[TEE-L].new_stock: expected 0, got 1"]
    no_events_expected = correct_01()
    del no_events_expected["external_events"]
    assert score(no_events_expected).strict


def test_turn_count_and_missing_results():
    res = correct_01()
    res["turns"] = res["turns"][:1]
    s = score(res)
    assert s.actual_decisions == ["propose", None] and "turn_count" in names(s)
    res["turns"] = "not a list"
    assert score(res).actual_decisions == [None, None]
    s = score(None)
    assert not s.strict and s.state_diffs == ["no result for this case in the results file"]


# --------------------------------------------------------------------------- run level and CLI

def results_doc(results: list[dict], input_bytes: bytes) -> dict:
    return {"contract_version": 1, "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "hardware": {"x": 1}, "configuration": {"x": 1}, "results": results}


def test_run_level_report_and_warnings():
    raw = PUBLIC_CASES.read_bytes()
    labels = json.loads(PUBLIC_EXPECTED.read_text())
    broken = correct_01()
    broken["final_state"]["refunds"].append(dict(broken["final_state"]["refunds"][0]))
    doc = results_doc([broken, correct_06()], raw)
    doc["input_sha256"] = "0" * 64
    labels["cases"] = labels["cases"][:1] + labels["cases"][5:] + [{"case_id": "ghost", "expected": {}}]
    scores, warnings = score_run(raw, labels, doc)
    assert [s.case_id for s in scores] == [P01, P06]
    assert any("input_sha256" in w for w in warnings)
    assert any("4 input case(s) have no label" in w for w in warnings)
    assert any("label ghost has no input case" in w for w in warnings)
    report = build_report(scores, warnings)
    assert report["summary"]["strict_success"] == 1 and report["invariants"] == {"duplicate_effects": 2}  # count and ratio checks
    text = render_text(report)
    assert "FAIL public_01_refund_boundary [public]" in text and "WARNING:" in text


def test_long_diffs_are_truncated_in_text():
    res = correct_01()
    res["final_state"]["products"] = [dict(p, stock=99, price_cents=1) for p in res["final_state"]["products"]]
    report = build_report([score(res)], [])
    assert "more difference(s) in the --json report" in render_text(report)


def test_cli_scores_the_public_set(tmp_path, capsys):
    raw = PUBLIC_CASES.read_bytes()
    reference = reference_results(raw, json.loads(PUBLIC_EXPECTED.read_text()))
    reference["results"][1]["turns"][1]["decision"] = "deny"
    results_path, report_path = tmp_path / "results.json", tmp_path / "report.json"
    results_path.write_text(json.dumps(reference))
    argv = ["--input", str(PUBLIC_CASES), "--labels", str(PUBLIC_EXPECTED), "--output", str(results_path)]
    assert main(argv + ["--json", str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "Strict case success: 5/6" in out and "completed->deny x1" in out
    assert json.loads(report_path.read_text())["summary"]["cases"] == 6
    assert main(argv + ["--fail-under", "0.9"]) == 1
    assert main(argv + ["--fail-under", "0.5"]) == 0
    assert main(["--input", str(tmp_path / "missing.json"), "--labels", str(PUBLIC_EXPECTED), "--output", str(results_path)]) == 2


def test_variant_suite_discriminates():
    raw = VARIANTS.read_bytes()
    labels = json.loads(LABELS.read_text())
    perfect = build_report(*score_run(raw, labels, reference_results(raw, labels)))
    assert perfect["summary"]["strict_success"] == perfect["summary"]["cases"] >= 150
    assert perfect["summary"]["invariant_violations"] == 0
    # A desk that never acts: no mutation, so no safety violations, but it fails every actionable case.
    cases = json.loads(raw)["cases"]
    passive = results_doc([result(c["case_id"], [turn("clarify") for _ in c["turns"]], c["initial_state"])
                           for c in cases], raw)
    report = build_report(*score_run(raw, labels, passive))
    assert report["summary"]["strict_rate"] < 0.2
    assert report["summary"]["turn_decision_accuracy"] < 0.3


def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(18, 18)
    assert round(low, 3) == 0.824 and high == 1.0


# --------------------------------------------------------------------------- state view

def test_diff_views_is_type_strict_and_readable():
    assert diff_views(True, 1) == ["state: expected true, got 1"]
    assert diff_views(3000, 3000.0) == ["state: expected 3000, got 3000.0"]
    assert diff_views({"a": 1}, {"b": 1}) == ["state.a: missing (expected 1)", "state.b: unexpected 1"]
    keyed = diff_views([{"sku": "A", "n": 1}], [{"sku": "B", "n": 1}], "p")
    assert keyed == ["p[A]: missing", 'p[B]: unexpected {"n": 1, "sku": "B"}']
    positional = diff_views([1, 2], [1], "x") + diff_views([1], [1, 3], "y")
    assert positional == ["x: expected 2 item(s), got 1", "x[1]: missing 2", "y: expected 1 item(s), got 2", "y[1]: unexpected 3"]
    duplicates = diff_views([{"sku": "A"}], [{"sku": "A"}, {"sku": "A"}], "d")
    assert duplicates[0] == "d: expected 1 item(s), got 2"
    assert snippet("x" * 500).endswith("...") and len(snippet("x" * 500)) == 140


def test_business_view_keeps_malformed_parts_visible():
    view = business_view({"as_of": "2026-01-01", "refunds": None, "returns": ["junk"], "idempotency": "bad"})
    assert view["refunds"] is None and view["returns"] == ["junk"] and view["idempotency"] == "bad"
    assert view["exchanges"] is None and view["top_level_keys"] == ["as_of", "idempotency", "refunds", "returns"]


@pytest.mark.parametrize("case_id", list(CASES))
def test_every_public_reference_result_is_strict(case_id):
    labels = json.loads(PUBLIC_EXPECTED.read_text())
    res = reference_results(PUBLIC_CASES.read_bytes(), labels)["results"][list(CASES).index(case_id)]
    assert score(res, case_id).strict
