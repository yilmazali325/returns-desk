"""Fourth live subset (NEW-2): public_02 exchange, confirm turn mentions sending the original back.
Expected: propose, completed, completed (exchange TEE-M -> TEE-L), never a refund."""
import copy, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
pub = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
cases = []
for i, msg in enumerate(["Yes, go ahead, I'll send it back.", "Yes, confirm. I'll ship it back today.",
                         "Confirm - I'll return the old one."]):
    c = copy.deepcopy(pub["public_02_exchange"]); c["case_id"] = f"x{i}_exchange_confirm_send_back"
    c["request_id"] = f"REQ-X{i}"
    c["turns"] = [c["turns"][0], {"message": msg, "confirm_proposal": "last"}, {"message": "Yes, confirm.", "confirm_proposal": "last"}]
    cases.append(c)
(HERE / "live4.json").write_text(json.dumps({"contract_version": 1, "cases": cases}, indent=1) + "\n")
