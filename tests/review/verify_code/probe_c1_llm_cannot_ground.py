"""A model that understands correctly but whose target the regex grounding cannot see."""
from desk.llm import FakeLLM
from desk.session import run_case
from desk.profiles import get_profile
from desk.agents.toolspecs import REPLY, REPORT, HANDOFF, CREATE_PROPOSAL

st = {"as_of": "2026-05-10", "currency": "USD",
      "products": [{"sku": "PARKA-M", "product_id": "PARKA", "size": "M", "price_cents": 9000, "stock": 2},
                   {"sku": "BEANIE-S", "product_id": "BEANIE", "size": "S", "price_cents": 1500, "stock": 5}],
      "orders": [{"order_id": "Q-77", "customer_id": "CU-1", "status": "delivered", "delivered_on": "2026-05-01",
                  "lines": [{"line_id": "A1", "sku": "PARKA-M", "paid_cents": 8100, "unopened": True, "final_sale": False, "returned": False},
                            {"line_id": "A2", "sku": "BEANIE-S", "paid_cents": 1500, "unopened": True, "final_sale": False, "returned": False}]}],
      "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}

def smart(req):
    # Resolution: hand off the line the customer clearly means (the $15 one); Ops: propose; then reply.
    if req.role == "resolution" and HANDOFF in req.allowed:
        return (HANDOFF, {"intent": "propose_refund", "order_id": "Q-77", "line_id": "A2"})
    if req.role == "operations" and CREATE_PROPOSAL in req.allowed:
        return (CREATE_PROPOSAL, {"remedy": "refund"})
    if REPORT in req.allowed:
        return (REPORT, {"outcome": "proposed"})
    return (REPLY, {"message": "Here is the proposal for you to review."})

for msg in ["Please refund the cheap one I paid 15 dollars for.", "Refund the winter hat please.", "I'd like my money back for the second item."]:
    case = {"case_id": "x", "customer_id": "CU-1", "request_id": "REQ-x", "initial_state": st, "turns": [{"message": msg}], "faults": []}
    r = run_case(case, FakeLLM(smart), get_profile("F"))
    gates = [e["error"]["code"] for e in r["action_audit"] if e.get("kind") == "gate" and e.get("error")]
    print(repr(msg), "->", r["turns"][0]["decision"], "| gate rejections:", gates, "|", r["turns"][0]["response"][:140])
