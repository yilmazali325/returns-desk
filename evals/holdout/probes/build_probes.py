"""Post-hoc probes (written AFTER reading desk/ code; NOT part of the frozen held-out score).
Each probe isolates one suspected generalizable weakness found while diagnosing holdout_v2."""
import json
from pathlib import Path
from evals.holdout.build_holdout import state, apparel, order, ln, ago, refund, exch, CONFIRM, YES, unclear
from evals.oracle import build_expectation

A = "2026-11-12"
P = []
def add(cid, cust, st, turns, expect, note):
    case = {"case_id": cid, "customer_id": cust, "request_id": f"REQ-{cid}", "initial_state": st,
            "turns": [({"message": m, "confirm_proposal": "last"} if f else {"message": m}) for m, f, _ in turns], "faults": []}
    exp = build_expectation(case, [i for _, _, i in turns])
    assert exp["decisions"] == expect, (cid, exp["decisions"])
    P.append((case, {"case_id": cid, "family": "probe", "note": note, "intents": [i for _, _, i in turns], "expected": exp}))

one = lambda cust, sku="TEE-M", paid=4000: state(A, apparel(), [order("O-1", cust, ago(A, 5), [ln("L1", sku, paid)])])
add("p01_policy_question_single_line", "C1", one("C1"),
    [("What's your return policy for t-shirts? Just asking.", False, unclear("POLICY_QUESTION"))], ["clarify"],
    "A policy question on a one-line account is not a refund request.")
add("p02_hypothetical_question", "C2", one("C2"),
    [("If I wanted to return something later, how long would I have?", False, unclear("POLICY_QUESTION"))], ["clarify"],
    "Hypothetical; no request.")
add("p03_confirm_ok_wait_yes", "C3", one("C3"),
    [("refund my tee please", False, refund("O-1", "L1")), ("ok wait yes, confirm it", True, CONFIRM)], ["propose", "completed"],
    "'wait' used as a filler inside a confirmation (CLAR #9: other wording executes).")
add("p04_confirm_mentions_other_size", "C4", one("C4"),
    [("refund my tee please", False, refund("O-1", "L1")), ("Yes please go ahead, I'll reorder it in a small later.", True, CONFIRM)],
    ["propose", "completed"], "Confirmation that mentions an unrelated future size.")
add("p05_confirm_hold_on_yes", "C5", one("C5"),
    [("refund my tee please", False, refund("O-1", "L1")), ("Hold on... yes, that's right. Confirm.", True, CONFIRM)],
    ["propose", "completed"], "'hold on' as hesitation then explicit yes.")
add("p06_runs_large_size_down", "C6", one("C6", "HOODIE-L", 6000),
    [("this hoodie runs large, can i get one size smaller?", False, exch("O-1", "L1", "DOWN")), ("yes", True, CONFIRM)],
    ["propose", "completed"], "'runs large' describes fit, not the owned size.")
add("p07_deny_then_offered_alternative", "C7", one("C7", "JACKET-M", 9000),
    [("can I exchange my jacket for a large?", False, exch("O-1", "L1", "L")),
     ("ok what about the small then?", False, exch("O-1", "L1", "S")), ("yes", True, CONFIRM)],
    ["deny", "propose", "completed"], "JACKET-L stock 0 -> deny listing S; customer takes S.")



def main(out_dir: Path = Path(__file__).parent) -> None:
    """Write probes.json and probe_labels.json (the oracle already agreed with every hand label above)."""
    (out_dir / "probes.json").write_text(json.dumps({"contract_version": 1, "cases": [c for c, _ in P]}, indent=1) + "\n")
    (out_dir / "probe_labels.json").write_text(json.dumps({"labels_version": 1, "cases": [l for _, l in P]}, indent=1) + "\n")
    print(len(P), "probes")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    main()
