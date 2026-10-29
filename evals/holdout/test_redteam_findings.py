"""Red-team regression tests for the held-out v2 findings (docs/review/redteam.md).

Run explicitly (not in the default testpaths):
    .venv/bin/pytest evals/holdout/test_redteam_findings.py -q -p no:cacheprovider

Every test states the behaviour the contract + CLARIFICATIONS defaults require. On the code reviewed
on 2026-10-28 they FAIL; each failure is one finding. No phrase-specific expectations: each test
feeds the REAL desk (orchestrator, gates, gateway, store) a correct agent decision, or checks a
grounding property that must hold for any wording.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from desk.agents.toolspecs import HANDOFF
from desk.grounding import analyze
from desk.llm import RawCall, interpret_calls
from desk.profiles import get_profile
from desk.session import CaseSession, run_case
from desk.testing.fake_policy_llm import FakePolicyLLM

HERE = Path(__file__).resolve().parent
CASES = {c["case_id"]: c for c in json.loads((HERE / "holdout_v2.json").read_text())["cases"]}


class CorrectAgent(FakePolicyLLM):
    """The repo's own deterministic double, except that on the first Resolution step of the given
    turns it makes the handoff a competent agent makes (what Qwen3.5-4B actually emitted in out_F.json)."""

    def __init__(self, handoffs: dict[int, dict]):
        super().__init__()
        self.handoffs = handoffs
        self.turn = 0
        self.sent: set[int] = set()

    def step(self, role, messages, tools, allowed, max_tokens):
        if role == "resolution" and self.turn in self.handoffs and self.turn not in self.sent and HANDOFF in allowed:
            self.sent.add(self.turn)
            step = interpret_calls([RawCall(HANDOFF, self.handoffs[self.turn])], tools, list(allowed))
            step.role, step.text = role, json.dumps(self.handoffs[self.turn])
            step.prompt_tokens, step.completion_tokens, step.latency_ms = 0, 1, 0.0
            step.finish_reason, step.http_attempts = "stop", 1
            return step
        return super().step(role, messages, tools, allowed, max_tokens)


def run_with(case_id: str, handoffs: dict[int, dict]) -> list[dict]:
    case = copy.deepcopy(CASES[case_id])
    llm = CorrectAgent(handoffs)
    session = CaseSession.from_case(case, llm, get_profile("F"))
    outputs = []
    for number, turn in enumerate(case["turns"], start=1):
        llm.turn = number
        outputs.append(session.run_turn(turn["message"], turn.get("confirm_proposal")))
    return outputs


# --- Finding 2: the gates veto a correct agent decision ------------------------------------------

@pytest.mark.parametrize("case_id, handoff, line", [
    # "chaqueta" (Spanish for jacket): agent picks the jacket line of a 2-line order
    # Repro updated for SPEC §6.0: with two lines and no ids typed, a correct agent cites the customer's
    # words ("evidence"); code checks the citation. The handoffs below are the live ones plus that field.
    pytest.param("hx12_spanish_mix_refund_jacket", {"intent": "propose_refund", "order_id": "O-3001", "line_id": "L1",
                                                    "evidence": "la chaqueta"}, "L1",
                 # Known limitation, left open on purpose: code checks a citation against the English catalog
                 # words, so "chaqueta" cannot be verified and the gate asks which item (safe). Fixing it needs a
                 # multilingual garment lexicon; adding words for this one case would be a phrase patch.
                 marks=pytest.mark.xfail(strict=True, reason="hx12: a Spanish garment word cannot be verified")),
    # "sending the large back" (the small fits): agent picks the large tee
    ("hx04_two_sizes_keep_small", {"intent": "propose_refund", "order_id": "O-88", "line_id": "L2",
                                   "evidence": "sending the large back"}, "L2"),
    # three products mentioned, "money back for the cardigan"
    ("hx14_long_rambling_cardigan", {"intent": "propose_refund", "order_id": "O-9002", "line_id": "L1",
                                     "evidence": "my money back for the cardigan"}, "L1"),
])
def test_correct_target_choice_is_accepted(case_id, handoff, line):
    out = run_with(case_id, {1: handoff})
    assert out[0]["decision"] == "propose", out[0]["response"]
    assert out[0]["proposal"]["line_id"] == line


@pytest.mark.parametrize("case_id, handoff, sku", [
    ("hx05_typos_exchange", {"intent": "propose_exchange", "order_id": "O-1201", "line_id": "L1",
                             "requested_size": "M", "requested_product": "SAME"}, "HOODIE-M"),       # "a mediun"
    ("hx10_jeans_numeric_swap", {"intent": "propose_exchange", "order_id": "O-1206", "line_id": "L1",
                                 "requested_size": "32", "requested_product": "SAME"}, "JEANS-32"),   # "the 30s for 32s"
    ("hx13_spanglish_exchange_tee", {"intent": "propose_exchange", "order_id": "O-3002", "line_id": "L1",
                                     "requested_size": "L", "requested_product": "SAME"}, "TEE-L"),   # "por una talla L"
])
def test_correct_size_choice_is_accepted(case_id, handoff, sku):
    out = run_with(case_id, {1: handoff})
    assert out[0]["decision"] == "propose", out[0]["response"]
    assert out[0]["proposal"]["replacement_sku"] == sku


# --- Finding 3: a size word that matches no owned line removes the only line ---------------------

def _lines(case_id):
    case = CASES[case_id]
    cid = case["customer_id"]
    lines = [{"order_id": o["order_id"], "line_id": l["line_id"], "sku": l["sku"]}
             for o in case["initial_state"]["orders"] if o["customer_id"] == cid for l in o["lines"]]
    return lines, case["initial_state"]["products"]


def test_fit_description_does_not_unground_single_line():
    lines, catalog = _lines("hx08_size_up_already_largest")  # one HOODIE-XL line
    a = analyze(["This hoodie runs small, can I get one size bigger?"], lines, catalog)
    assert a.unique_target == ("O-1204", "L1"), a.to_dict()["explain"]


def test_follow_up_size_after_out_of_stock_deny_stays_grounded():
    lines, catalog = _lines("hx40_stock_zero_then_alternative_4turns")  # one HOODIE-M line
    msgs = ["can I swap my hoodie for an L?", "yes confirm", "dang. is the XL available instead?"]
    a = analyze(msgs, lines, catalog)
    assert a.unique_target == ("O-18", "L1"), a.to_dict()["explain"]
    assert a.requested_size == "XL"


def test_alternative_offered_in_deny_reply_can_be_taken():
    """End to end: T2 denies (stock went to 0) and its reply offers XL; the customer takes XL."""
    out = run_case(copy.deepcopy(CASES["hx40_stock_zero_then_alternative_4turns"]), FakePolicyLLM(), get_profile("F"))
    decisions = [t["decision"] for t in out["turns"]]
    assert "Available instead: HOODIE-XL" in out["turns"][1]["response"]
    assert decisions == ["propose", "deny", "propose", "completed"], decisions


# --- Finding 4: 'instead' inside a question becomes a cancel and the reply invents one -----------

def test_question_with_instead_is_not_reported_as_a_cancel():
    out = run_case(copy.deepcopy(CASES["hx40_stock_zero_then_alternative_4turns"]), FakePolicyLLM(), get_profile("F"))
    assert "You asked us not to go ahead" not in out["turns"][2]["response"], out["turns"][2]["response"]


# --- Finding 6: CLARIFICATIONS #4 says clarify when refund and exchange wording both appear -------

def test_return_or_swap_is_remedy_unclear():
    lines, catalog = _lines("hx48_remedy_unclear_then_swap")
    a = analyze(["Can I return the polo, or maybe swap it for an XL? Not sure which is better tbh"], lines, catalog)
    assert a.remedy_cue == "both", a.remedy_cue
