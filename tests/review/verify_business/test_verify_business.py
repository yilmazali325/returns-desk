"""Verifier repros for the business review (B1-B3 extensions and NEW-1).

Real orchestrator, gateway and store; the LLM is FakePolicyLLM or a scripted FakeLLM, so no model
or llama-server is needed. Each repro asserts the contract/CLARIFICATIONS behaviour and is a strict
xfail today: once the fix lands it XPASSes (a failure), which is the signal to drop the marker.

Run:  .venv/bin/pytest tests/review/verify_business -q -rxX
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desk.llm import FakeLLM  # noqa: E402
from desk.profiles import FINAL  # noqa: E402
from desk.session import run_case  # noqa: E402
from desk.testing.fake_policy_llm import FakePolicyLLM  # noqa: E402

PUBLIC = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]}


def _case(case_id: str, turns: list[Any]) -> dict[str, Any]:
    c = copy.deepcopy(PUBLIC[case_id])
    c["turns"] = [t if isinstance(t, dict) else {"message": t} for t in turns]
    return c


def _decisions(result: dict[str, Any]) -> list[str]:
    return [t["decision"] for t in result["turns"]]


def _stock(result: dict[str, Any]) -> dict[str, int]:
    return {p["sku"]: p["stock"] for p in result["final_state"]["products"]}


# --- B1 extension: on an EXCHANGE proposal, a confirmation that mentions returning the old item
# is read as "asks for a refund instead"; the grant is withheld and a REFUND proposal replaces it.

RETURN_WORDING_EXCHANGE_CONFIRMATIONS = [
    "Yes, go ahead. I'll return the old one.",
    "Yes, please go ahead with the return.",
    "Confirmed, start the return.",
    "Yes, I will send it back.",
]


@pytest.mark.parametrize("message", RETURN_WORDING_EXCHANGE_CONFIRMATIONS)
def test_exchange_confirmation_mentioning_the_return_executes(message: str) -> None:
    result = run_case(_case("public_02_exchange", ["Exchange line L1 in order O100 for the same tee in size L.",
                                                   {"message": message, "confirm_proposal": "last"}]),
                      FakePolicyLLM(), FINAL)
    assert _decisions(result) == ["propose", "completed"], result["turns"][1]["response"]
    assert _stock(result)["TEE-L"] == 2 and result["final_state"]["refunds"] == []


# --- NEW-1: the same cancel regex fires on UNCONFIRMED clarification answers. A public_04-shaped
# answer that names the line but carries 'instead'/'wait'/"don't cancel" becomes a pure cancel.

CLARIFY_ANSWERS_WITH_CANCEL_WORDS = [
    "It's the tee instead of the jacket, line L1.",
    "Wait, it's the tee, line L1.",
    "The tee, line L1. Sorry for the wait.",
    "Line L1, the tee. Please don't cancel my request.",
]


@pytest.mark.parametrize("answer", CLARIFY_ANSWERS_WITH_CANCEL_WORDS)
def test_clarification_answer_with_cancel_type_word_is_not_a_cancel(answer: str) -> None:
    result = run_case(_case("public_04_clarify", ["I want to return an item from O100.", answer,
                                                  {"message": "Yes, confirm the proposed refund.",
                                                   "confirm_proposal": "last"}]),
                      FakePolicyLLM(), FINAL)
    assert _decisions(result) == ["clarify", "propose", "completed"], result["turns"][1]["response"]
    assert [r["amount_cents"] for r in result["final_state"]["refunds"]] == [3000]


def test_clarification_answer_without_cancel_words_still_works() -> None:
    """Guard: the same answer without the cancel-type word completes (so the word is the cause)."""
    result = run_case(_case("public_04_clarify", ["I want to return an item from O100.", "It's the tee, line L1.",
                                                  {"message": "Yes, confirm the proposed refund.",
                                                   "confirm_proposal": "last"}]),
                      FakePolicyLLM(), FINAL)
    assert _decisions(result) == ["clarify", "propose", "completed"]


# --- B2 extension: other plausible "return it and get size X" wordings are code-forced refunds.

RETURN_AND_GET_SIZE = [
    "I'd like to return line L1 of order O100 and get a large.",
    "Return line L1 of O100, I need a large.",
    "Line L1 of O100 is too small, I'd like to return it and get size L.",
]


@pytest.mark.parametrize("message", RETURN_AND_GET_SIZE)
def test_return_and_get_a_size_is_not_a_refund(message: str) -> None:
    result = run_case(_case("public_02_exchange", [message]), FakePolicyLLM(), FINAL)
    proposal = result["turns"][0]["proposal"]
    assert result["turns"][0]["decision"] == "clarify" or (proposal and proposal["remedy"] == "exchange"), proposal


# --- B3 extension: the path a real model takes. It guesses a size, the gate answers SIZE_UNSPECIFIED
# with the hint "Ask which size; do not guess.", the model asks, and the clarify is accepted.


def test_expired_exchange_without_size_denies_after_a_guessed_size() -> None:
    llm = FakeLLM({"resolution": [
        ("handoff_to_operations", {"intent": "propose_exchange", "order_id": "O100", "line_id": "L1",
                                   "requested_size": "L", "requested_product": "SAME"}),
        ("reply_to_customer", {"message": "Which size would you like instead?"}),
        ("reply_to_customer", {"message": "Which size would you like instead?"}),
    ], "operations": [("create_proposal", {"remedy": "exchange", "replacement_sku": "TEE-L"}),
                      ("report_to_resolution", {"outcome": "denied"})]})  # the guessed size now reaches Ops
    result = run_case(_case("public_03_expired", ["I'd like to exchange line L1 of order O100 for a different size."]),
                      llm, FINAL)
    turn = result["turns"][0]
    assert turn["decision"] == "deny" and "available for an exchange" not in turn["response"], turn["response"]
