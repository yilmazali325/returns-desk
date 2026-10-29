"""Verifier probe NEW-1: cancel-cue words on UNCONFIRMED clarification answers (public_04 shape)."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from desk.profiles import FINAL
from desk.session import run_case
from desk.testing.fake_policy_llm import FakePolicyLLM

PUB = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
SECOND = sys.argv[1:] or [
    "I mean line L1, the size M tee. Refund it please.",       # public wording
    "I mean line L1, the size M tee.",
    "Sorry, I meant the tee, line L1, not the jacket.",
    "The tee, line L1. Sorry for the wait.",
    "It's the tee instead of the jacket, line L1.",
    "Wait, it's the tee, line L1.",
    "Line L1, the tee. Please don't cancel my request.",
]
for second in SECOND:
    c = copy.deepcopy(PUB["public_04_clarify"])
    c["turns"] = [{"message": "I want to return an item from O100."}, {"message": second},
                  {"message": "Yes, confirm the proposed refund.", "confirm_proposal": "last"}]
    r = run_case(c, FakePolicyLLM(), FINAL)
    d = [t["decision"] for t in r["turns"]]
    print(d, repr(second), "\n    turn2:", r["turns"][1]["response"][:110], "| refunds:", [x["amount_cents"] for x in r["final_state"]["refunds"]])
