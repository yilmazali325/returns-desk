"""Third live subset: ID-anchored exchanges with a bare numeric size (RT-2 size gate on public-style text).
Expected: propose (replacement = the named size), then completed."""
import copy, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H = {c["case_id"].split("_")[0]: c for c in json.loads((ROOT / "evals/holdout/holdout_v2.json").read_text())["cases"]}
cases, EXPECTED = [], {}
def add(cid, base, msg, sku):
    c = copy.deepcopy(H[base]); c["case_id"] = cid; c["request_id"] = "REQ-" + cid.upper(); c["faults"] = []
    c["turns"] = [{"message": msg}, {"message": "Yes, confirm the proposed exchange.", "confirm_proposal": "last"}]
    cases.append(c); EXPECTED[cid] = {"decisions": ["propose", "completed"], "replacement_sku": sku}
add("num1_jeans_for_a_32", "hx10", "Exchange line L1 of order O-1206 for a 32.", "JEANS-32")
add("num2_jeans_to_34", "hx10", "Please swap line L1 in O-1206 to 34.", "JEANS-34")
add("num3_sneaker_for_a_10", "hx58", "Exchange line L1 of order O-36 for a 10.", "SNEAKER-10")
(HERE / "live3.json").write_text(json.dumps({"contract_version": 1, "cases": cases}, indent=1) + "\n")
(HERE / "live3_expected.json").write_text(json.dumps(EXPECTED, indent=1) + "\n")
