"""C13 probe: a colour request ('blue one, same size') on a catalog where colour variants share
product_id. Contract: an exchange must be a DIFFERENT size of the same product_id. The customer
asked for the same size, so the only correct outcomes are deny or clarify; proposing another size
the customer did not ask for is wrong."""
import json
from desk.llm import FakeLLM
from desk.session import run_case
from desk.profiles import get_profile
from desk.testing.fake_policy_llm import FakePolicyLLM
from desk.agents.toolspecs import REPLY, REPORT

st = {"as_of": "2026-05-10", "currency": "USD",
      "products": [{"sku": "TEE-M-RED", "product_id": "TEE", "size": "M", "price_cents": 2000, "stock": 2},
                   {"sku": "TEE-L-RED", "product_id": "TEE", "size": "L", "price_cents": 2000, "stock": 2},
                   {"sku": "TEE-M-BLUE", "product_id": "TEE", "size": "M", "price_cents": 2000, "stock": 2}],
      "orders": [{"order_id": "Q-77", "customer_id": "CU-1", "status": "delivered", "delivered_on": "2026-05-01",
                  "lines": [{"line_id": "A1", "sku": "TEE-M-RED", "paid_cents": 2000, "unopened": True,
                             "final_sale": False, "returned": False}]}],
      "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}

def dumb(req):
    if REPLY in req.allowed:
        return (REPLY, {"message": "Could you tell me a bit more?"})
    if REPORT in req.allowed:
        return (REPORT, {"outcome": "needs_info"})
    return "no idea"

for msg in ["Exchange my red tee on order Q-77 for the blue one, same size.",
            "Can I swap my tee for the same size in blue?",
            "I'd like to exchange my tee for a different colour, keep the size M."]:
    case = {"case_id": "x", "customer_id": "CU-1", "request_id": "REQ-x", "initial_state": st,
            "turns": [{"message": msg}], "faults": []}
    for name, llm in (("fake-policy", FakePolicyLLM()), ("stub", FakeLLM(dumb))):
        r = run_case(case, llm, get_profile("F"))
        t = r["turns"][0]
        p = t["proposal"] or {}
        print(f"{name:11} {msg!r}\n            -> {t['decision']} replacement={p.get('replacement_sku')} | {t['response'][:150]}")
