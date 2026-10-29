"""Public-style follow-ups after the public_05 out-of-stock deny (verification of RT-3 / RT-4).

Writes tests/review/verify_redteam/followups.json: public_05 + a 3rd turn taking the alternative the
deny reply offers (TEE-S) + a confirm turn. Expected for every case: propose, deny, propose, completed.
"""
import copy, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
pub = json.loads((ROOT / "candidate_assets/public_cases.json").read_text())
base = next(c for c in pub["cases"] if c["case_id"] == "public_05_stale_stock")
T3 = {
    "f1_small_instead": "OK then, the small instead please.",
    "f2_size_s_instead": "Fine, exchange it for size S instead.",
    "f3_take_small": "Then I'll take the small.",
    "f4_ids_size_s": "Exchange line L1 of O100 for size S instead.",
    "f5_what_about_small": "What about the small then?",
    "f6_ids_no_instead": "Exchange line L1 of order O100 for the same tee in size S.",
}
cases = []
for i, (cid, msg) in enumerate(T3.items()):
    c = copy.deepcopy(base)
    c["case_id"] = cid
    c["request_id"] = f"REQ-VR-{i}"
    c["turns"] = base["turns"] + [{"message": msg}, {"message": "Yes, confirm.", "confirm_proposal": "last"}]
    cases.append(c)
(HERE / "followups.json").write_text(json.dumps({"contract_version": 1, "cases": cases}, indent=1) + "\n")
print(len(cases))
