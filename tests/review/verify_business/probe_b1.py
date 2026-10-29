"""Verifier probe for B1: run confirmations through the real orchestrator with FakePolicyLLM."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from desk.profiles import FINAL
from desk.session import run_case
from desk.testing.fake_policy_llm import FakePolicyLLM

PUB = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}

def run(cid, msgs):
    c = copy.deepcopy(PUB[cid])
    c["turns"] = [m if isinstance(m, dict) else {"message": m} for m in msgs]
    r = run_case(c, FakePolicyLLM(), FINAL)
    withheld = [e["result"].get("reason") for e in r["action_audit"] if e.get("tool") == "grant_withheld"]
    return [t["decision"] for t in r["turns"]], withheld

PHRASES = sys.argv[1:] or [
    "Yes, confirm.",
    "Yes please go ahead, no need to wait.",
    "Yes, I can't wait to get it.",
    "Yes, go ahead. I can't wait.",
    "Yes, please proceed. Sorry for the wait.",
    "Yes, confirm. Thanks for waiting.",
    "Yes, go ahead. Don't cancel it.",
    "Yes, I accept, I won't change my mind.",
    "Yes, please refund it instead of giving store credit.",
    "Yes, that works, I'll take that instead.",
    "Yes, go ahead and stop the emails please.",
    "Okay, go ahead with the proposal.",
    "Yes. No changes, please proceed.",
    "No problem, go ahead.",
    "Yes, I'd like to return it as proposed.",
    "Yes, I will send it back.",
    "Yes, please process the return.",
    "Yes, confirm the return and refund.",
]
for cid, first in (("public_01_refund_boundary", "Refund line L1 of my order O100. It is unopened."),
                   ("public_02_exchange", "Exchange line L1 in order O100 for the same tee in size L.")):
    print("==", cid)
    for p in PHRASES:
        d, w = run(cid, [first, {"message": p, "confirm_proposal": "last"}])
        flag = "" if d[-1] == "completed" else "   <-- NOT EXECUTED"
        print(f"{d} withheld={w} :: {p!r}{flag}")
