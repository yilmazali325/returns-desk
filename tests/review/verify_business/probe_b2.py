"""Verifier probe for B2: 'return ... and send me a <size>' phrasing."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from desk.grounding import analyze
from desk.profiles import FINAL
from desk.session import run_case
from desk.store import Store
from desk.testing.fake_policy_llm import FakePolicyLLM

PUB = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
c0 = PUB["public_02_exchange"]
st = c0["initial_state"]
lines = [dict(order_id=o["order_id"], **l) for o in st["orders"] if o["customer_id"] == c0["customer_id"] for l in o["lines"]]
MSGS = sys.argv[1:] or [
    "Return L1 of O100 and send me a large.",
    "I'd like to return line L1 of order O100 and get a large.",
    "I want to return line L1 of order O100 for a large.",
    "Return line L1 of O100, I need a large.",
    "Line L1 of O100 is too small, I'd like to return it and get size L.",
    "Return line L1 of order O100 for the same tee in size L.",
    "Can I return line L1 of O100 and have size L sent?",
]
for m in MSGS:
    c = copy.deepcopy(c0); c["turns"] = [{"message": m}]
    r = run_case(c, FakePolicyLLM(), FINAL)
    # analysis via the session store's customer lines
    gate = [e["error"]["code"] for e in r["action_audit"] if e.get("kind") == "gate" and e.get("error")]
    grounding = next(e["result"] for e in r["action_audit"] if e.get("kind") == "grounding")
    p = r["turns"][0]["proposal"]
    print(f"{m!r}\n   remedy_cue={grounding['remedy_cue']} words={grounding['remedy_words']} req={grounding['requested_size']} orig_hints={grounding['original_size_hints']}"
          f"\n   -> {r['turns'][0]['decision']} {p and (p['remedy'], p['replacement_sku'], p['amount_cents'])} gates={gate}")
