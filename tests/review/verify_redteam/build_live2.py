"""Second live subset (port 8422): public-style ID-anchored variants for RT-4 / RT-6 / NEW-1 / NEW-2,
plus the public_05 follow-ups. Expected decisions are in EXPECTED (per CONTRACT + CLARIFICATIONS #4/#9)."""
import copy, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
pub = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
fol = json.loads((HERE / "followups.json").read_text())["cases"]
cases, EXPECTED = [], {}

def add(cid, base, turns, expected):
    c = copy.deepcopy(pub[base]); c["case_id"] = cid; c["request_id"] = "REQ-" + cid.upper()
    c["turns"] = turns; c["faults"] = []; cases.append(c); EXPECTED[cid] = expected

add("n1_dont_cancel_go_ahead", "public_01_refund_boundary",
    [pub["public_01_refund_boundary"]["turns"][0], {"message": "Yes - please don't cancel, go ahead.", "confirm_proposal": "last"}],
    ["propose", "completed"])
add("n1_dont_stop_now", "public_01_refund_boundary",
    [pub["public_01_refund_boundary"]["turns"][0], {"message": "Yes, go ahead. Don't stop now!", "confirm_proposal": "last"}],
    ["propose", "completed"])
add("n2_exchange_confirm_send_back", "public_02_exchange",
    [pub["public_02_exchange"]["turns"][0], {"message": "Yes, confirm - I'll send back the M and get the L.", "confirm_proposal": "last"}],
    ["propose", "completed"])
add("n2_exchange_confirm_return_it", "public_02_exchange",
    [pub["public_02_exchange"]["turns"][0], {"message": "Yes, return it and send me the L.", "confirm_proposal": "last"}],
    ["propose", "completed"])
add("r6_return_or_swap_ids", "public_01_refund_boundary",
    [{"message": "Should I return line L1 of O100 or swap it for size L?"}, {"message": "Refund it please."},
     {"message": "Yes, confirm.", "confirm_proposal": "last"}],
    ["clarify", "propose", "completed"])
add("r6_return_or_exchange_ids", "public_01_refund_boundary",
    [{"message": "I want to return or exchange line L1 of order O100."}, {"message": "A refund, please."},
     {"message": "Yes, confirm.", "confirm_proposal": "last"}],
    ["clarify", "propose", "completed"])
for c in fol:
    c = copy.deepcopy(c); cases.append(c)
    EXPECTED[c["case_id"]] = ["propose", "deny", "propose", "completed"]
(HERE / "live2.json").write_text(json.dumps({"contract_version": 1, "cases": cases}, indent=1) + "\n")
(HERE / "live2_expected.json").write_text(json.dumps(EXPECTED, indent=1) + "\n")
print(len(cases))
