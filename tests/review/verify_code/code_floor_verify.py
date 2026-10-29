"""Verifier copy of the code-only floor (C1). Stub 'model': Resolution only asks a vague question,
Operations only reports needs_info. Real run_case, profile F. Writes adapter-shaped output.
Usage: PYTHONPATH=. python tests/review/verify_code/code_floor_verify.py <cases.json> <out.json>"""
import hashlib, json, sys
from collections import Counter
from desk.llm import FakeLLM
from desk.session import run_case
from desk.profiles import get_profile
from desk.agents.toolspecs import REPLY, REPORT

def dumb(req):
    if REPLY in req.allowed:
        return (REPLY, {"message": "Could you tell me a bit more?"})
    if REPORT in req.allowed:
        return (REPORT, {"outcome": "needs_info"})
    return "no idea"

inp, out = sys.argv[1], sys.argv[2]
raw = open(inp, "rb").read()
results = [run_case(c, FakeLLM(dumb), get_profile("F")) for c in json.loads(raw)["cases"]]
kinds = Counter()
for r in results:
    for e in r["action_audit"]:
        if e.get("kind") in ("guard", "fallback"):
            kinds[(e.get("kind"), e.get("tool") or e.get("guard") or e.get("name"))] += 1
print("guard/fallback counts:", dict(kinds))
json.dump({"contract_version": 1, "input_sha256": hashlib.sha256(raw).hexdigest(), "hardware": {}, "configuration": {}, "results": results}, open(out, "w"))
