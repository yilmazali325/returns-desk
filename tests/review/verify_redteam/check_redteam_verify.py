"""Verifier tests for the redteam review (docs/review/redteam.md). Named check_*.py so the default
`pytest` run (testpaths = tests) does not collect them; run explicitly:

    .venv/bin/pytest tests/review/verify_redteam/check_redteam_verify.py -q -p no:cacheprovider -o addopts=""

Every test states behaviour required by CONTRACT.md + CLARIFICATIONS.md and FAILS on the code
reviewed on 2026-10-28. All use public_cases.json-style, ID-anchored wording (the shape hidden
fixture variants are most likely to have), and run the real desk with the repo's own
deterministic double, or with a double that makes the handoff Qwen3.5-4B actually emitted live
(tests/review/verify_redteam/out/live_F_*.json).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from desk.agents.toolspecs import HANDOFF
from desk.grounding import analyze
from desk.llm import RawCall, interpret_calls
from desk.profiles import get_profile
from desk.session import CaseSession, run_case
from desk.testing.fake_policy_llm import FakePolicyLLM

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
HOLDOUT = {c["case_id"].split("_")[0]: c for c in
           json.loads((ROOT / "evals/holdout/holdout_v2.json").read_text())["cases"]}


def _public(case_id: str, turns: list[dict]) -> dict:
    case = copy.deepcopy(PUBLIC[case_id])
    case["turns"], case["faults"] = turns, []
    return case


def _decisions(result: dict) -> list[str]:
    return [t["decision"] for t in result["turns"]]


class LiveHandoff(FakePolicyLLM):
    """FakePolicyLLM, except the first Resolution step of turn 1 is the handoff the live model made."""

    def __init__(self, handoff: dict):
        super().__init__()
        self.handoff, self.sent = handoff, False

    def step(self, role, messages, tools, allowed, max_tokens):
        if role == "resolution" and not self.sent and HANDOFF in allowed:
            self.sent = True
            step = interpret_calls([RawCall(HANDOFF, self.handoff)], tools, list(allowed))
            step.role, step.text = role, json.dumps(self.handoff)
            step.prompt_tokens, step.completion_tokens, step.latency_ms = 0, 1, 0.0
            step.finish_reason, step.http_attempts = "stop", 1
            return step
        return super().step(role, messages, tools, allowed, max_tokens)


# --- NEW-2: an exchange confirmation that mentions sending the item back becomes a REFUND --------

def test_exchange_confirm_mentioning_send_back_executes_the_exchange():
    """CLARIFICATIONS #9: only an explicit cancel/change blocks the flag. 'I'll send it back' is how
    every exchange works; it must not be read as 'asks for a refund instead'."""
    first = PUBLIC["public_02_exchange"]["turns"][0]
    case = _public("public_02_exchange", [first, {"message": "Yes, go ahead, I'll send it back.", "confirm_proposal": "last"},
                                          {"message": "Yes, confirm.", "confirm_proposal": "last"}])
    result = run_case(case, FakePolicyLLM(), get_profile("F"))
    state = result["final_state"]
    assert state["refunds"] == [], f"customer confirmed an exchange but got a refund: {state['refunds']}"
    assert len(state["exchanges"]) == 1, _decisions(result)
    assert _decisions(result)[1] == "completed", _decisions(result)


# --- NEW-1: a negated cancel word ('don't cancel', 'don't stop') is read as a cancel ---------------

def test_negated_cancel_word_does_not_block_the_confirmation():
    first = PUBLIC["public_01_refund_boundary"]["turns"][0]
    case = _public("public_01_refund_boundary",
                   [first, {"message": "Yes - please don't cancel, go ahead.", "confirm_proposal": "last"}])
    result = run_case(case, FakePolicyLLM(), get_profile("F"))
    assert "You asked us not to go ahead" not in result["turns"][1]["response"], result["turns"][1]["response"]
    assert _decisions(result) == ["propose", "completed"], _decisions(result)


# --- RT-2 (size gate), ID-anchored: a bare numeric size the model read correctly is rejected ------

def test_bare_numeric_size_with_ids_is_accepted():
    case = copy.deepcopy(HOLDOUT["hx10"])  # O-1206: L1 JEANS-30, L2 TEE-M
    case["faults"] = []
    case["turns"] = [{"message": "Exchange line L1 of order O-1206 for a 32."},
                     {"message": "Yes, confirm the proposed exchange.", "confirm_proposal": "last"}]
    llm = LiveHandoff({"intent": "propose_exchange", "order_id": "O-1206", "line_id": "L1",
                       "requested_size": "32", "requested_product": "SAME"})  # what Qwen3.5-4B emitted live
    result = run_case(case, llm, get_profile("F"))
    assert _decisions(result) == ["propose", "completed"], (_decisions(result), result["turns"][0]["response"])
    assert result["final_state"]["exchanges"][0]["replacement_sku"] == "JEANS-32"


# --- RT-3, ID-anchored: 'the small' after an out-of-stock deny is read as the owned size ----------

def test_alternative_offered_after_public_05_deny_can_be_taken():
    turns = PUBLIC["public_05_stale_stock"]["turns"] + [{"message": "What about the small then?"},
                                                        {"message": "Yes, confirm.", "confirm_proposal": "last"}]
    case = copy.deepcopy(PUBLIC["public_05_stale_stock"])
    case["turns"] = turns
    result = run_case(case, FakePolicyLLM(), get_profile("F"))
    assert "Available instead: TEE-S" in result["turns"][1]["response"]
    assert _decisions(result) == ["propose", "deny", "propose", "completed"], _decisions(result)


# --- RT-6, ID-anchored: 'return ... or swap' is remedy-unclear per CLARIFICATIONS #4 --------------

def test_return_or_swap_with_ids_is_remedy_unclear():
    lines = [{"order_id": "O100", "line_id": "L1", "sku": "TEE-M"}]
    catalog = PUBLIC["public_01_refund_boundary"]["initial_state"]["products"]
    a = analyze(["Should I return line L1 of O100 or swap it for size L?"], lines, catalog)
    assert a.remedy_cue == "both", a.remedy_cue
