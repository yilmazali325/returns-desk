"""Code-only E2E: public_01 (refund) and public_02 (exchange) with the confirm-turn text varied.
Every variant carries confirm_proposal 'last' and expresses consent; expected T2 'completed'."""
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from desk.session import run_case
from desk.profiles import get_profile
from desk.testing.fake_policy_llm import FakePolicyLLM

pub = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets/public_cases.json").read_text())["cases"]}
REFUND = ["Yes - please don't cancel, go ahead.", "Yes, go ahead. Don't stop now!", "Sure, no need to wait, go ahead.",
          "Yes, stop asking and just do it.", "Yes confirm, no need to hold off", "Nope, no changes, confirm it.",
          "Yes, confirm the refund of line L1.", "Yes, refund the proposed item and amount."]
EXCHANGE = ["Yes, confirm - I'll send back the M and get the L.", "Yes, return it and send me the L.",
            "Yes please, swap the M for the L.", "Yes, confirm this exchange.", "Yes, go ahead with the size L instead of my M."]
out = {}
for cid, texts in (("public_01_refund_boundary", REFUND), ("public_02_exchange", EXCHANGE)):
    for t in texts:
        case = copy.deepcopy(pub[cid])
        case["turns"][1]["message"] = t
        r = run_case(case, FakePolicyLLM(), get_profile("F"))
        withheld = [a["result"]["reason"] for a in r["action_audit"] if a.get("tool") == "grant_withheld"]
        d = [x["decision"] for x in r["turns"]]
        out[f"{cid} | {t}"] = d
        print(d, withheld, "|", cid[:9], "|", t, "|", r["turns"][1]["response"][:90])
