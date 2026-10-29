"""C5 probe: policy.resolve_size (gateway) vs grounding.resolve_size (gate helper) on direct inputs,
plus what the orchestrator actually passes to the gateway."""
from desk import policy, grounding
cat = [
    {"sku": "TEE-S", "product_id": "TEE", "size": "S", "price_cents": 2000, "stock": 3},
    {"sku": "TEE-M", "product_id": "TEE", "size": "M", "price_cents": 2000, "stock": 3},
    {"sku": "TEE-L", "product_id": "TEE", "size": "L", "price_cents": 2000, "stock": 3},
    {"sku": "PANT-XL", "product_id": "PANT", "size": "XL", "price_cents": 5000, "stock": 3},
    {"sku": "PANT-32", "product_id": "PANT", "size": "32", "price_cents": 5000, "stock": 3},
]
probes = [("lrg", "TEE-M"), ("bigger", "TEE-M"), ("size up", "TEE-M"), ("teel", "TEE-M"), ("M", "TEE-M"),
          ("UP", "PANT-XL"), ("", "TEE-M"), ("L", "TEE-M"), ("UP", "TEE-M"), ("large", "TEE-M")]
diff = 0
for req, orig in probes:
    p = policy.resolve_size(req, orig, cat); g = grounding.resolve_size(req, orig, cat)
    diff += p != g
    print(f"{req!r:10} from {orig:8} policy={p} grounding={g} {'DIFF' if p != g else ''}")
print("differ:", diff, "of", len(probes))
# What the gateway actually receives: grounding's canonical analysis.requested_size
for req in ["lrg", "bigger", "size up", "teel", "large"]:
    print(f"canonical_size({req!r}) =", grounding.canonical_size(req), "| direction:", grounding._direction_of(req))
