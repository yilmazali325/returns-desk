"""Tests for the v3 suite tooling: the frozen files are reproducible and well-formed, a hand label that
disagrees with the oracle is rejected, and the checker catches each kind of malformed input/label.

    .venv/bin/pytest evals/holdout_v3 --cov=evals.holdout_v3 --cov-branch
"""
from __future__ import annotations

import copy
import dataclasses
import json
from typing import Callable

import pytest

from evals.holdout_v3 import build_v3, check_v3
from evals.holdout_v3.spec import exchanged, refund_offer, refunded, stock_change

SUITE = json.loads((check_v3.HERE / "holdout_v3.json").read_text(encoding="utf-8"))
LABELS = json.loads((check_v3.HERE / "labels_v3.json").read_text(encoding="utf-8"))
PUBLIC = json.loads(check_v3.PUBLIC_CASES.read_text(encoding="utf-8"))


def spec(case_id: str):
    return next(s for s in build_v3.all_specs() if s.case_id == case_id)


def case(doc: dict, case_id: str) -> dict:
    return next(c for c in doc["cases"] if c["case_id"] == case_id)


# --------------------------------------------------------------------------- frozen files

def test_frozen_files_pass_the_checker(capsys):
    assert check_v3.main([]) == 0
    assert "OK: 60 cases" in capsys.readouterr().out


def test_frozen_files_match_the_hand_specs():
    assert build_v3.main(["--check"]) == 0


def test_public_cases_also_pass_the_input_checker():
    # The checker derives its shape from the public file, so the public file must satisfy it.
    assert check_v3.suite_errors(PUBLIC, PUBLIC) == []


def test_build_writes_files_and_detects_stale_ones(tmp_path, capsys):
    assert build_v3.main(["--out-dir", str(tmp_path), "--check"]) == 1  # nothing written yet
    assert build_v3.main(["--out-dir", str(tmp_path)]) == 0
    assert build_v3.main(["--out-dir", str(tmp_path), "--check"]) == 0
    (tmp_path / build_v3.LABELS_NAME).write_text("{}", encoding="utf-8")
    assert build_v3.main(["--out-dir", str(tmp_path), "--check"]) == 1
    assert "STALE: ['labels_v3.json']" in capsys.readouterr().out


# --------------------------------------------------------------------------- hand label vs oracle

def with_turn(case_id: str, number: int, **changes):
    s = spec(case_id)
    turns = list(s.turns)
    turns[number - 1] = dataclasses.replace(turns[number - 1], **changes)
    return dataclasses.replace(s, turns=turns)


@pytest.mark.parametrize(("make", "message"), [
    (lambda: with_turn("v3_t02", 1, decision="propose"), "turn 1: hand propose/RETURN_WINDOW_EXPIRED"),
    (lambda: with_turn("v3_t01", 1, proposal=refund_offer(1, "O5120", "L1", 3200)), "turn 1: hand proposal"),
    (lambda: with_turn("v3_t27", 2, alternatives=()), "marks ['error'] as contested"),
    (lambda: dataclasses.replace(spec("v3_t01"), effect=refunded("O5120", "L1", 3200)), "refunds: hand"),
    (lambda: dataclasses.replace(spec("v3_t11"), effect=exchanged("O1111", "L1", "JEANS-32")), "exchanges: hand"),
    (lambda: dataclasses.replace(spec("v3_t11"), stock_after={}), "stock: hand {}"),
    (lambda: dataclasses.replace(spec("v3_t30"), effect=None), "state_unchanged: oracle False"),
    (lambda: dataclasses.replace(spec("v3_t12"), effect=exchanged("O1213", "L1", "TEE-L")), "returned lines: hand"),
    (lambda: dataclasses.replace(spec("v3_t22"), faults=[stock_change(2, "NO-SUCH-SKU", 0)]), "did not apply"),
])
def test_a_hand_label_that_disagrees_with_the_oracle_is_rejected(make: Callable, message: str):
    with pytest.raises(build_v3.LabelMismatch) as raised:
        build_v3.label_for(make())
    assert message in str(raised.value)


def test_agreeing_label_carries_only_hand_alternatives():
    label = build_v3.label_for(spec("v3_n19"))
    assert label["expected"]["alternatives"] == {"1": ["propose"]}
    assert label["expected"]["decisions"] == ["clarify", "propose", "completed"]


# --------------------------------------------------------------------------- input checker error paths

def mutated(doc: dict, mutate: Callable[[dict], None]) -> dict:
    copy_ = copy.deepcopy(doc)
    mutate(copy_)
    return copy_


def t29(doc: dict) -> dict:
    return case(doc, "v3_t29")


def first_line(doc: dict) -> dict:
    return t29(doc)["initial_state"]["orders"][0]["lines"][0]


def set_(target: Callable[[dict], dict], key: str, value) -> Callable[[dict], None]:
    return lambda doc: target(doc).__setitem__(key, value)


INPUT_MUTATIONS = [
    (set_(lambda d: d, "contract_version", 2), "root keys / contract_version"),
    (lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])), "duplicate case_id"),
    (set_(t29, "extra", 1), "case keys"),
    (set_(t29, "customer_id", ""), "ids must be non-empty strings"),
    (set_(lambda d: t29(d)["initial_state"], "extra", 1), "state keys"),
    (set_(lambda d: t29(d)["initial_state"], "as_of", "14/11/2026"), "as_of/currency"),
    (set_(lambda d: t29(d)["initial_state"], "refunds", [{"x": 1}]), "transaction records must start empty"),
    (lambda d: t29(d)["initial_state"]["products"].append(copy.deepcopy(t29(d)["initial_state"]["products"][0])),
     "duplicate SKU"),
    (set_(lambda d: t29(d)["initial_state"]["products"][0], "color", "red"), "product keys"),
    (set_(lambda d: t29(d)["initial_state"]["products"][0], "stock", True), "money/stock not int"),
    (lambda d: t29(d)["initial_state"]["orders"].append(copy.deepcopy(t29(d)["initial_state"]["orders"][0])),
     "duplicate order_id"),
    (lambda d: t29(d)["initial_state"]["orders"][0].pop("status"), "order keys"),
    (set_(lambda d: t29(d)["initial_state"]["orders"][0], "delivered_on", None), "delivered_on must be an ISO date"),
    (lambda d: t29(d)["initial_state"]["orders"][0]["lines"].append(copy.deepcopy(first_line(d))),
     "missing or duplicate line_id"),
    (set_(first_line, "qty", 1), "line keys"),
    (set_(first_line, "paid_cents", "3200"), "paid_cents not int"),
    (set_(first_line, "unopened", "yes"), "line flags not boolean"),
    (set_(first_line, "sku", "TEE-XXL"), "not in catalog"),
    (set_(t29, "turns", []), "no turns"),
    (set_(lambda d: t29(d)["turns"][0], "speaker", "agent"), "bad turn"),
    (set_(lambda d: t29(d)["turns"][1], "confirm_proposal", "first"), "confirm_proposal must be 'last'"),
    (set_(lambda d: t29(d)["faults"][1], "type", "network_down"), "fault shape"),
    (set_(lambda d: t29(d)["faults"][0], "sku", "TEE-XXL"), "stock_change target"),
    (set_(lambda d: t29(d)["faults"][0], "before_turn", 4), "before_turn outside 1..3"),
]


@pytest.mark.parametrize(("mutate", "message"), INPUT_MUTATIONS)
def test_input_checker_reports_each_malformation(mutate: Callable, message: str):
    errors = check_v3.suite_errors(mutated(SUITE, mutate), PUBLIC)
    assert any(message in error for error in errors), errors


# --------------------------------------------------------------------------- label checker error paths

def t29_label(doc: dict) -> dict:
    return case(doc, "v3_t29")


def t29_expected(doc: dict) -> dict:
    return t29_label(doc)["expected"]


LABEL_MUTATIONS = [
    (set_(lambda d: d, "labels_version", 2), "labels_version must be 1"),
    (lambda d: d["cases"].reverse(), "differ from the input cases or their order"),
    (set_(t29_label, "extra", 1), "label keys"),
    (set_(t29_label, "family", "misc_timeout"), "family must start with terse_ or natural_"),
    (set_(t29_label, "family", "natural_timeout"), "style split"),
    (lambda d: t29_label(d)["intents"].pop(), "one intent per turn"),
    (lambda d: t29_expected(d)["reasons"].pop(), "reasons length != 3 turns"),
    (lambda d: t29_expected(d)["decisions"].__setitem__(2, "done"), "unknown decision"),
    (lambda d: t29_expected(d)["proposals"].__setitem__(1, {"seq": 1}), "proposal iff propose"),
    (lambda d: t29_expected(d)["proposals"][0].pop("currency"), "proposal keys"),
    (set_(lambda d: t29_expected(d)["proposals"][0], "customer_id", "C1"), "proposal not bound to case"),
    (set_(lambda d: t29_expected(d)["proposals"][0], "amount_cents", 0.0), "amount/seq not int"),
    (set_(lambda d: t29_expected(d)["alternatives"], "9", ["error"]), "bad alternatives for turn 9"),
    (set_(lambda d: t29_expected(d)["alternatives"], "2", ["completed"]), "bad alternatives for turn 2"),
    (lambda d: t29_expected(d)["events"][0].pop("applied"), "event keys"),
    (lambda d: t29_expected(d)["final_business_state"]["top_level_keys"].pop(), "final state keys"),
    (set_(t29_expected, "state_unchanged", "no"), "state_unchanged not bool"),
]


@pytest.mark.parametrize(("mutate", "message"), LABEL_MUTATIONS)
def test_label_checker_reports_each_malformation(mutate: Callable, message: str):
    errors = check_v3.label_errors(mutated(LABELS, mutate), SUITE)
    assert any(message in error for error in errors), errors


# --------------------------------------------------------------------------- checker CLI

def test_checker_cli_fails_on_bad_labels(tmp_path, capsys):
    bad = tmp_path / "labels.json"
    bad.write_text(json.dumps(mutated(LABELS, set_(lambda d: d, "labels_version", 2))), encoding="utf-8")
    assert check_v3.main(["--labels", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "ERROR labels_version must be 1" in out and "FAILED" in out


def test_checker_cli_reports_unreadable_input(tmp_path, capsys):
    assert check_v3.main(["--input", str(tmp_path / "missing.json")]) == 2
    assert "CHECK FAILED" in capsys.readouterr().err
