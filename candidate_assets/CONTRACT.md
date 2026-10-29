# Candidate interface v1

Implement `python adapter.py --input cases.json --output results.json`. The adapter must invoke the same agents, local models and business tools as the UI. It runs all cases sequentially, resetting state before each. No answer lookup or special evaluation implementation. Framework, programming language and internal architecture remain your choice; a Python wrapper may launch them.

## Input

The root contains `contract_version: 1` and `cases`. Each case has `case_id`, `customer_id`, `request_id`, `initial_state`, `turns` and `faults`. Cases are independent. A case concerns at most 1 refund or exchange. Never hardcode IDs, phrases, counts or expected outcomes. Support unseen cases with the same schema and disclosed rules.

State contains `as_of` (ISO date), `currency` (`USD`), `products`, `orders`, and initially empty `returns`, `refunds`, `exchanges`, `quarantine`, `idempotency`. Products have `sku`, `product_id`, `size`, `price_cents`, `stock`. Orders have `order_id`, `customer_id`, `status`, `delivered_on`, and `lines`. Each line is 1 unit with `line_id`, `sku`, `paid_cents`, `unopened`, `final_sale`, `returned`. All monetary and stock values are integers.

Each turn contains `message` and optional `confirm_proposal: "last"`. Resolve `last` to the immediately preceding non-null proposal ID and pass this as explicit user confirmation. Without it, no business mutation is permitted. A confirmed proposal must bind customer, order line, remedy, amount and replacement SKU. Revalidate current state immediately before committing. Return clarification if the request lacks a unique target. Do not infer authorization from the message alone.

## Disclosed business rules

Use the supplied `as_of`, not the computer clock. Only the authenticated customer's delivered orders qualify. The delivery age must be 0–30 calendar days inclusive; the unit must be unopened, not final sale and not already returned. Refund exactly `paid_cents`; discounts are already allocated. No taxes, shipping, fees or price-difference payments are in scope.

An exchange must use a different size of the same `product_id`, with the same catalog `price_cents` as the original SKU and positive replacement stock. Decrement replacement stock by 1. A refunded or exchanged original goes into quarantine and must not increase sellable stock. Mark its order line returned. No-op refusals must leave business state unchanged.

Maintain an auditable refund/return or exchange record and a completed idempotency record keyed by `request_id`. A replay must return the original result without repeating money or stock effects. Records must identify the customer, order, line, request, proposal and any replacement SKU. Candidate-generated IDs may vary.

## Faults

Apply `stock_change` immediately before its 1-based `before_turn`, replacing the specified SKU's stock with `new_stock`. Record it as an external event. `timeout_after_commit` makes the first commit persist successfully but raises a simulated timeout before its result reaches the agent. Recovery must query/reconcile state rather than perform a second business action. Faults must occur at the business-tool boundary, not as instructions telling the LLM the correct answer.

## Output

Root: `contract_version`, `input_sha256` (SHA256 of exact input bytes), `hardware` (CPU, GPU, RAM, OS), `configuration` (model revision/hash, quantization, engine/version), `results`.

Each result: `case_id`, `turns`, `action_audit`, `final_state`, `metrics`. Each turn: `decision` (`propose`, `clarify`, `deny`, `completed`, `error`), `response`, `proposal` (object or null). Audit entries identify agent, tool, arguments, result/error and committed effects without exposing private chain of thought. Final state uses the input schema plus transaction records. Metrics include `elapsed_ms`, `peak_ram_mb`, `peak_vram_mb`; use null for unavailable memory readings and explain limitations. Include failed runs. The supplied validator checks structure only; it is not a correctness scorer or implemented retail backend.
