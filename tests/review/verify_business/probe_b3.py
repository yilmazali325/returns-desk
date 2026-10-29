"""Verifier probe for B3: ineligible line + 'exchange for a different size' with a model that
follows the gate's own hint (guess a size -> SIZE_UNSPECIFIED 'Ask which size' -> ask)."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from desk.llm import FakeLLM
from desk.profiles import FINAL
from desk.session import run_case
from desk.testing.fake_policy_llm import FakePolicyLLM

PUB = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
c = copy.deepcopy(PUB["public_03_expired"])
c["turns"] = [{"message": "I'd like to exchange line L1 of order O100 for a different size."}]
scripts = {
    "fake_policy": FakePolicyLLM(),
    "guess_then_ask": FakeLLM({"resolution": [
        ("handoff_to_operations", {"intent": "propose_exchange", "order_id": "O100", "line_id": "L1",
                                   "requested_size": "L", "requested_product": "SAME"}),
        ("reply_to_customer", {"message": "Which size would you like instead?"}),
        ("reply_to_customer", {"message": "Which size would you like instead?"})], "operations": []}),
    "unspecified_handoff": None,
}
for name, llm in scripts.items():
    if llm is None:
        continue
    r = run_case(copy.deepcopy(c), llm, FINAL)
    gates = [(e["error"] or {}).get("code") for e in r["action_audit"] if e.get("kind") == "gate"]
    hints = [(e["error"] or {}).get("hint") for e in r["action_audit"] if e.get("kind") == "gate"]
    t = r["turns"][0]
    print(name, "->", t["decision"], "| gates", gates, hints, "\n   ", t["response"])
