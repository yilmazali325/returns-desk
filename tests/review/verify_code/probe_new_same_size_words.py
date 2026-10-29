"""NEW probe: 'the same size' (size not named) on a plain size-only catalog where exactly one other
size is valid. CLARIFICATIONS #2 / variants family exchange_same_size: a same-size exchange is deny.
The desk instead proposes the one other valid size, which the customer explicitly did not ask for."""
from desk.llm import FakeLLM
from desk.session import run_case
from desk.profiles import get_profile
from desk.testing.fake_policy_llm import FakePolicyLLM
from desk.agents.toolspecs import REPLY, REPORT
from desk import grounding

def state(stock_s):
    return {"as_of": "2026-05-10", "currency": "USD",
            "products": [{"sku": "TEE-S", "product_id": "TEE", "size": "S", "price_cents": 2000, "stock": stock_s},
                         {"sku": "TEE-M", "product_id": "TEE", "size": "M", "price_cents": 2000, "stock": 2},
                         {"sku": "TEE-L", "product_id": "TEE", "size": "L", "price_cents": 2000, "stock": 2}],
            "orders": [{"order_id": "O-1", "customer_id": "CU-1", "status": "delivered", "delivered_on": "2026-05-01",
                        "lines": [{"line_id": "L1", "sku": "TEE-M", "paid_cents": 2000, "unopened": True,
                                   "final_sale": False, "returned": False}]}],
            "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}

def dumb(req):
    if REPLY in req.allowed:
        return (REPLY, {"message": "Could you tell me a bit more?"})
    if REPORT in req.allowed:
        return (REPORT, {"outcome": "needs_info"})
    return "no idea"

msgs = ["Can I exchange my tee from order O-1 for a new one in the same size? The stitching is off.",
        "Please swap line L1 of O-1 for the same size, mine has a hole.",
        "Exchange line L1 of O-1 for a medium."]  # control: names the size -> deny SAME_SIZE expected
for stock_s in (0, 3):
    for msg in msgs:
        case = {"case_id": "x", "customer_id": "CU-1", "request_id": "REQ-x", "initial_state": state(stock_s),
                "turns": [{"message": msg}], "faults": []}
        for name, llm in (("fake-policy", FakePolicyLLM()), ("stub", FakeLLM(dumb))):
            t = run_case(case, llm, get_profile("F"))["turns"][0]
            p = t["proposal"] or {}
            print(f"S-stock={stock_s} {name:11} {msg[:60]!r:64} -> {t['decision']:8} repl={p.get('replacement_sku')}")

# Two turns, as a grader fixture would run it: request + explicit confirm of the 'last' proposal.
case = {"case_id": "x2", "customer_id": "CU-1", "request_id": "REQ-x2", "initial_state": state(0),
        "turns": [{"message": msgs[1]}, {"message": "Yes, go ahead.", "confirm_proposal": "last"}], "faults": []}
r = run_case(case, FakePolicyLLM(), get_profile("F"))
print("two-turn:", [t["decision"] for t in r["turns"]])
fs = r["final_state"]
print("  exchanges:", [(e.get("original_sku"), e.get("replacement_sku")) for e in fs["exchanges"]])
print("  stock:", {p["sku"]: p["stock"] for p in fs["products"]}, "| line returned:", fs["orders"][0]["lines"][0]["returned"])

# A model that understands correctly ("same size" = M) and hands that off: does code let it deny?
from desk.agents.toolspecs import HANDOFF, CREATE_PROPOSAL
def smart(req):
    if req.role == "resolution" and HANDOFF in req.allowed:
        return (HANDOFF, {"intent": "propose_exchange", "order_id": "O-1", "line_id": "L1",
                          "requested_size": "M", "requested_product": "SAME"})
    if req.role == "operations" and CREATE_PROPOSAL in req.allowed:
        return (CREATE_PROPOSAL, {"remedy": "exchange", "replacement_sku": "TEE-M"})
    if REPORT in req.allowed:
        return (REPORT, {"outcome": "denied"})
    return (REPLY, {"message": "Sorry, an exchange has to be for a different size."})
for stock_s in (0, 3):
    case = {"case_id": "x3", "customer_id": "CU-1", "request_id": "REQ-x3", "initial_state": state(stock_s),
            "turns": [{"message": msgs[1]}], "faults": []}
    r = run_case(case, FakeLLM(smart), get_profile("F"))
    gates = [e["error"]["code"] for e in r["action_audit"] if e.get("kind") == "gate" and e.get("error")]
    t = r["turns"][0]
    print(f"smart model S-stock={stock_s}: {t['decision']} repl={(t['proposal'] or {}).get('replacement_sku')} gates={gates}")
