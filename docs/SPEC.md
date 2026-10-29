# Returns Desk — internal build spec v2 (layer contracts)

Source of truth for every module. External rules: `candidate_assets/CONTRACT.md` and
`candidate_assets/Retail_AI_Interview_Brief.pdf`; when this spec and CONTRACT.md disagree, CONTRACT.md
wins — note it in `docs/CLARIFICATIONS.md`. v2 folds in the adversarial review in
`docs/notes/spec_critique.json` (read it for measured evidence behind each rule).

Design principle: **agents say yes; code can say no** (§6.0). LLM agents read language, pick the line,
remedy and size, decide clarify-vs-act, start every business action and write replies. Code owns
identity, fact checks on the agents' choices, policy, money, stock, commits, idempotency, recovery and
the decision label; it may block, read or replay, never act for an agent. Every block or read by code
is audited (`gate`, `guard`, `fallback`) and counted in metrics.

Stack: Python 3.12 (uv, `.venv`), FastAPI + vanilla JS UI (no CDN; works offline), SQLite (stdlib),
llama.cpp `llama-server` 0.7.0 (Homebrew build 11669, commit 620d2225c, Metal), Qwen3.5-4B GGUF from
`unsloth/Qwen3.5-4B-GGUF` revision `e87f176479d0855a907a41277aca2f8ee7a09523`:
Q4_K_M sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
Q8_0 sha256 `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1`. Models live in
`models/` (git-ignored). Hardware: Apple M5 Pro (15 CPU cores, 16-core GPU), 24 GB unified memory,
macOS 26.4.1 (25E253). Run with `.venv/bin/python`; tests `.venv/bin/pytest` (default skips `-m llm`).
Never read the computer clock in business logic — use `state.as_of`. No randomness in business code.

## 1. Layout and ownership

```
adapter.py                 stdlib bootstrap shim → re-execs .venv python → desk.adapter_main
desk/
  domain.py                pydantic/dataclass types: LineSummary, Eligibility, ExchangeOption, Proposal,
                           ToolResult, OpsReport, TurnOutput; reason-code constants + CODE_CLASS table
  policy.py                pure rules: eligibility, exchange options, refund amount, date math
  store.py                 SQLite-backed store; exact-shape export; atomic commits; proposals/events tables
  faults.py                FaultPlan (stock_change before turn N; timeout_after_commit, fires once)
  tools.py                 BusinessTools gateway — the only path from agents to state; audit + faults
  audit.py                 AuditLog (append-only, listeners for SSE)
  grounding.py             deterministic text analyzer: IDs, products, sizes, remedy/cancel cues, candidates
  sanitize.py              neutralize control tokens / tool tags in customer text; cap length
  render.py                money formatting, reason-code wording, facts lines, templates, reply guard
  llm.py                   LLMClient (native | constrained protocol), Step, LLMError, FakeLLM
  llm_server.py            LlamaServer launch/health/stop, version + sha checks, cold-start timing
  profiles.py              named profiles (engine flags + app switches) — see §9
  telemetry.py             phys_footprint high-water (libproc), AGX unified GPU memory (ioreg), hardware
  agents/prompts.py        static system prompts (no fixture IDs; examples use ORD-7Q / LN-3 / SCARF)
  agents/toolspecs.py      tool JSON schemas per role (byte-stable, sort_keys)
  agents/resolution.py     Resolution agent loop
  agents/operations.py     Operations agent loop
  orchestrator.py          Desk: per-turn pipeline, grants, gates, decision derivation, response compose
  session.py               run_case(case, …) used by adapter, bench and UI
  adapter_main.py          contract adapter implementation (argparse, server lifecycle, output)
server/app.py + server/static/{index.html,app.js,styles.css}
bench/ (run.py, report.py, preflight.sh)   evals/ (independent oracle, expected, variants, scorer)
tests/unit, tests/e2e      scripts/ (setup.sh, run.sh, reset.sh, offline_check.sh, bench.sh)
```

## 2. State, export and records

Input `initial_state` keys: `as_of, currency, products[], orders[], returns[], refunds[], exchanges[],
quarantine[], idempotency{}`. **final_state has exactly the input keys in input order** and is built
from a deep copy of the input: stock and `line.returned` change in place, records are appended,
idempotency gets the entry. Unknown extra fields round-trip untouched. `json.dumps(export(load(s)))`
must equal `json.dumps(s)` for any untouched state. Proposals and external events are NOT in
final_state: they go to result-level `proposals[]` and `external_events[]` (and the audit).

Store: SQLite (`:memory:` for adapter cases, a file under `var/` for UI sessions). Tables:
`state_doc(version PK, json)` (canonical current document), `idempotency(request_id PK, json)`,
`returned_lines(order_id, line_id, PK(order_id,line_id))`, `proposals(proposal_id PK, json, status)`,
`events(event_id PK, json)`, `ledger(seq PK, transaction_id, type, json)`. A commit is ONE
`BEGIN IMMEDIATE … COMMIT`: insert idempotency row + returned_lines row (PK = duplicate guard) +
ledger rows + new state_doc version. The in-memory document is replaced only after COMMIT succeeds.

Deterministic IDs per session counter: `P-0001`, `TX-0001`, `RET-0001`, `RF-0001`, `EX-0001`, `Q-0001`,
`EV-0001`. Every record carries: `transaction_id, request_id, proposal_id, customer_id, order_id,
line_id, sku` (original), `replacement_sku` (null for refunds), `amount_cents` (0 for exchanges),
`currency, business_date` (= as_of), `status:"completed"`.
- refund: `returns[]` {return_id, remedy:"refund", disposition:"quarantine", …common}, `refunds[]`
  {refund_id, return_id, …common, amount_cents = line.paid_cents}, `quarantine[]` {quarantine_id,
  return_id, quantity:1, reason:"customer_return", …common}, line.returned = true.
- exchange: `returns[]` (remedy "exchange"), `exchanges[]` {exchange_id, return_id, original_sku,
  replacement_sku, …common}, `quarantine[]`, line.returned = true, replacement stock −1. The original
  SKU's stock never increases.
- `idempotency[request_id]` = {status:"completed", action:"refund"|"exchange", …common,
  result:{transaction_id, return_id, refund_id|exchange_id, amount_cents, replacement_sku}}.

## 3. Policy (pure)

Dates: `date.fromisoformat(s[:10])`; missing/invalid → `MISSING_DELIVERY_DATE`. days = as_of − delivered_on.
`eligibility(state, customer_id, order_id, line_id) -> Eligibility{eligible, reasons[], order_id, line_id,
sku, product_id, size, paid_cents, delivered_on, as_of, days_since_delivery, window_days:30}`.
Reasons (all failing, this order): `ORDER_NOT_FOUND` (missing OR another customer's — identical
wording, never leak), `LINE_NOT_FOUND`, `NOT_DELIVERED`, `MISSING_DELIVERY_DATE`, `DELIVERY_IN_FUTURE`
(days<0), `RETURN_WINDOW_EXPIRED` (days>30), `OPENED`, `FINAL_SALE`, `ALREADY_RETURNED`.
Booleans compared strictly (`unopened is True`, `final_sale is False`, `returned is False`).

`exchange_options(state, order_id, line_id) -> [ExchangeOption{sku, size, price_cents, stock, valid,
reasons[]}]` over every product with the same product_id and a different size. Valid iff same
product_id AND size.casefold() ≠ original size AND catalog price_cents == ORIGINAL SKU's catalog
price_cents AND stock > 0. Codes: `SAME_SIZE`, `DIFFERENT_PRODUCT`, `PRICE_MISMATCH`, `OUT_OF_STOCK`,
`SKU_NOT_FOUND` (original SKU not in catalog → exchange impossible; refund unaffected), `SIZE_NOT_OFFERED`.
Size order ladder: XXS<XS<S<M<L<XL<XXL<3XL<4XL<5XL; numeric sizes ascending. `resolve_size(requested,
original_sku, catalog)` maps "large"/"lg"/"L"/"UP"/"DOWN"/numeric to an offered size of that product or
returns a code. `refund_amount(line) = line.paid_cents`.

## 4. Business tools gateway (desk/tools.py)

`BusinessTools(store, customer_id, request_id, faults, audit)`. No tool takes customer_id; the LLM never
supplies proposal_id. Ops tools act on the **injected grounded target** (set by the orchestrator from
the validated handoff). Every call → `ToolResult{ok, data|None, error{code,message}|None}` + audit entry
`{agent, tool, arguments, result|error, committed_effects, visible_to_agent}`.

Idempotency short-circuit at the top of every Ops tool: if `request_id` has a completed record →
same target and same remedy → `data.replay=true, transaction=<original result>`; a different target or
the other remedy on that line → error `REQUEST_ALREADY_COMPLETED` (CLARIFICATIONS #7, review B4).

- `find_order_lines(order_ref="ALL")` → `{matches:[LineSummary], count, order_not_on_account?}` —
  LineSummary for Resolution hides policy fields: `{order_id, line_id, sku, product_id, size,
  aliases:[…], paid_cents}`.
- `account_card()` (orchestrator, not an LLM tool) → the same summaries for all the customer's lines +
  related catalog sizes; injected as "prefetch" in profiles that enable it (audited as agent
  `orchestrator`, tool `prefetch_account_card`).
- `check_eligibility(order_id?, line_id?)` → Eligibility (+ `completed_by_this_request` when replaying).
  Orchestrator only (no agent calls it): the facts behind a code-made deny (`guard/code_deny`).
- `check_proposal(remedy, replacement_sku=None)` → the answer `create_proposal` would give, but read-only:
  nothing is stored. Orchestrator only, after Operations ended a propose handoff without a decision
  (`fallback/check_proposal`): a refusal is reported, a proposal is never made.
- `list_exchange_options()` → `{original_sku, options:[ExchangeOption]}`.
- `create_proposal(remedy, replacement_sku=None)` → Proposal. Validates in code; amount computed in
  code; remedy must equal the handoff intent (`REMEDY_MISMATCH`); replacement must equal the resolved
  requested size (`REPLACEMENT_NOT_REQUESTED`) and be valid. **Idempotent on binding**: an identical
  binding returns the existing open proposal (no supersession). A different binding supersedes the
  open proposal.
- `execute_proposal()` → uses the turn's ConfirmationGrant (else `CONFIRMATION_REQUIRED`). One
  transaction: idempotency (completed + same proposal_id/binding → replay, no effects) → proposal must
  be open (`PROPOSAL_NOT_OPEN`) → binding hash (`BINDING_MISMATCH`) → re-run eligibility → exchange
  re-check (`OUT_OF_STOCK`, `PRICE_MISMATCH`, …) → write records → COMMIT → proposal `executed`.
  Revalidation failure → proposal `invalidated`, nothing else changes. Max one execute per turn.
  **timeout_after_commit**: after COMMIT succeeds, outside the transaction, if armed and not fired:
  mark fired, write the audit entry with real committed_effects and `visible_to_agent:false`, return
  error `TIMEOUT`. Replays never consume the fault.
- `get_request_status()` → `{request_id, status:"completed"|"not_found", record?}`.
- Faults: `apply_turn_faults(turn_no)` before each turn, list order, plain assignment of stock;
  unknown SKU → event `applied:false, reason:"SKU_NOT_FOUND"`; before_turn out of range → never applied
  (noted in audit); unknown fault types recorded and ignored. Event `{event_id, type:"stock_change",
  sku, old_stock, new_stock, before_turn, applied, source:"external"}`.

Proposal: `{proposal_id, request_id, customer_id, order_id, line_id, remedy, original_sku,
replacement_sku, amount_cents, currency:"USD", eligibility:{delivered_on, as_of, days_since_delivery,
window_days}, binding_hash (sha256 of canonical JSON {customer_id, order_id, line_id, remedy,
amount_cents, replacement_sku, request_id}), status: open|superseded|executed|invalidated,
created_turn}`.

## 5. Grounding (desk/grounding.py, pure) and sanitization

Grounding extracts what the customer literally typed and checks the agent's handoff against it. It
never chooses a line, a remedy or a size for the agent (§6.0). Sizes have ONE vocabulary and ONE
resolver: `policy.SIZE_WORDS`, `policy.normalize_size` and `policy.resolve_size`; grounding only
finds size words in text.

`analyze(messages, lines, catalog, as_of=None) -> Analysis` over ALL customer messages so far
(sanitized). Token extraction:
- IDs: casefold; compared after stripping `# - _` and spaces, whole-token only ("L1" ≠ "L10",
  "O100" ≠ "O1001"; "o-100", "#O100" == "O100"). A token equal to any ID is never a size. Money
  amounts ("$32", "30 dollars") are masked first, so they are never sizes; they are kept for
  citations. `order_refs` / `line_refs` (on the account), `unknown_order_refs` / `unknown_line_refs`
  (ID-shaped, not on the account). `id_candidates` = the lines the typed IDs allow (all lines when
  none was typed): facts only.
- Products: product_id / SKU tokens + a generic garment lexicon matched against the account's and
  catalog's product ids; garment words owned by no line are `unmatched_product_words` (and
  `requested_product_words` when asked for: "for the jacket").
- Sizes: size words (policy vocabulary), bare s/m/l only in fixed patterns, numbers after `size`, or
  after for/to/into when the catalog offers that number ("for a 32"). The words around a size decide
  its role: requested ("for a large", "get a large", "send me a large", "is the XL available?",
  "what about the small?", "the small instead") or original ("my medium", "the size M tee"). Fit words
  ("runs small", "a bit large") are directions (UP/DOWN), never sizes. "The same size" / "the same one"
  in an exchange is the requested size `SAME` (= the size the customer has; then SAME_SIZE).
  An original size is a HARD filter when it describes the item ("my size L tee", "the L tee"); a
  loose one that would remove every candidate is ignored and explained.
- `remedy_cue`: strong exchange (exchange, swap, different size, size up…) vs strong refund (refund,
  money back, reimburse) vs weak (return, send back; bigger/smaller; a size asked for with
  for/get/need/want/send me). Both strong, or any refund + any exchange wording joined by a hedge
  (or / maybe / not sure / which) → `both` (CLARIFICATIONS #4). A weak refund word never replaces an
  earlier strong exchange ("Yes, I'll send it back" keeps the exchange). `stated_remedy` = what the
  latest message that talks about the remedy states with STRONG words (`both`/`refund`/`exchange`/`none`).
- `cancel_cue` (last message only), per clause and negation-aware: an explicit cancel verb that is not
  negated (cancel, changed my mind, never mind, forget it, keep it, don't proceed / don't do it, don't
  want it / to go ahead), or a message that OPENS with no / stop / wait / hold on and never comes round
  to consenting. "Don't cancel", "no need to wait", "stop asking and do it", "sorry for the wait",
  "nope, no changes, confirm it" and "I'll take that instead" are not cancels.
- `inquiry` (last message): hypothetical wording ("just asking", "if I wanted") or rule/status words
  inside a question ("what's your policy…?", "did it go through?").
- A regex reading of the target (`candidates`, `unique_target`) is still computed. It is used for the
  one-time clarify feedback, for the replay/deny decision after completion, for clarify facts and as
  the test double's brain; it never overrides an agent's verified choice.

Checks over a handoff (each returns ok or a code the agent gets back with a hint and candidates):
- `check_target(analysis, order_id, line_id, evidence)` → `TargetCheck{ok, code, grounded_by, evidence,
  reason}`. Hard facts: the line is on the account (`ORDER_NOT_FOUND` / `LINE_NOT_FOUND`), no typed ID
  is unknown, every typed ID agrees (`TARGET_MISMATCH`). One line left by the typed IDs →
  `customer_ids`; one line on the account → `only_line` unless the customer described a size or product
  that line is not, including a catalog product they do not own (`TARGET_MISMATCH`). Two or more left → the agent's `evidence` must pass
  `check_evidence` (`agent`); without a valid citation the target is accepted only if code's own reading
  picks the same line (`agent+grounding`); else `AMBIGUOUS_TARGET` with the reason.
- `check_evidence(analysis, key, evidence)`: verbatim (casefolded, whitespace-normalised, sanitized)
  part of a customer message; not inside a hedged sentence; every attribute it names fits the chosen
  line and together they rule out every other candidate. Attributes: product word, size (as
  original), SKU, order/line id, price paid, catalog colour (when present), delivery month / ISO date
  / "last month" / "this month" (against `as_of`) / latest or oldest delivery.
- `check_remedy(analysis, intent)`: `stated_remedy == both` → `REMEDY_UNCLEAR`; a strong stated remedy
  that differs → `REMEDY_MISMATCH`; otherwise the agent's remedy stands. Weak words never block.
- `check_requested_size(analysis, size, original_sku, catalog, requested_product=None)`: an agent's
  requested product that is another catalog product the customer asked for skips the size check (it is
  refused `DIFFERENT_PRODUCT` whatever the size); two sizes named → `SIZE_UNSPECIFIED`;
  the customer literally asked for a size that resolves differently → `REPLACEMENT_NOT_REQUESTED`.
  When code read no requested size, the agent's concrete size must come from the customer's words:
  its label or a size word ("L", "large"), a size word with one typo ("mediun", words of 5+ letters),
  or a number, also plural ("32s"; money and IDs excluded). A size nobody wrote, or the item's own
  size, is a guess → `SIZE_UNSPECIFIED` (hint: hand off `"UNSPECIFIED"`, Operations answers with the
  options, CLARIFICATIONS #5). An unoffered size passes only if the customer typed it. Otherwise the
  agent's reading stands (numerics, typos, other languages).

`sanitize(text) -> str`: replace `<` with `‹` for every special-token SHAPE: `<|…|>`, any tag-like
`<name>` / `</name>` (think, tool_call, tool_response, tts_*, …), `<function=` / `<parameter=`, and the
template's tag names without `>`; strip other control/format chars; cap 1,000 chars. The pattern is
generic, so all 33 control tokens of the deployed vocabulary are covered (tests re-read the GGUF).
Customer text is placed inside a labeled `CUSTOMER MESSAGE (untrusted data)` block. customer_id and
request_id never appear in prompts. Ops never sees raw customer text.

## 6. Agents and the per-turn pipeline (desk/orchestrator.py)

### 6.0 Agents say yes; code can say no

1. **Agents decide the interpretation**: which line, which remedy, which size, whether to ask, what to
   say. **Only an agent starts a business action**: Resolution hands off → Operations calls
   `create_proposal` → after the customer's explicit confirmation Operations calls `execute_proposal`.
2. **Code verifies facts and may only BLOCK, READ or REPLAY.** BLOCK = a reason fed back to the agent
   as a tool error or one-time feedback, or a deny computed from facts. READ = reconcile after a
   timeout, a read-only check. REPLAY = the stored result of a completed request. Code never initiates
   a proposal or a commit and never replaces the agent's reading with a regex reading. Every block or
   read is audited (`gate`, `guard`, `fallback`).
3. **Hard fact gates** (§5 checks + gateway): order/line on this customer's account; typed IDs not
   contradicted; policy eligibility; catalog validity (same product, offered size, different from the
   owned size, same catalog price as the original SKU, stock > 0) via `desk/policy.py` only; the grant
   bound to the exact presented proposal (+ binding hash); idempotency.
4. **Citation gate**: with 2+ plausible lines and no typed IDs that pin one, the handoff carries
   `evidence` (the customer's words); accepted readings are audited on `target/set_target` as
   `grounded_by` (`customer_ids` | `only_line` | `agent` | `agent+grounding`) with the `evidence`.
5. **Sizes** are the agent's reading unless the customer's own size words contradict it, or the
   customer wrote no size the agent's could come from (a guess; §5).
6. **Remedy** is the agent's choice unless the latest remedy wording is strong and different, or names
   both remedies undecided (CLARIFICATIONS #4: then the agent's question stands).
7. **Clarify**: an agent's question goes to the customer. Code may send it back ONCE with what the
   conversation already names (`guard/clarify_veto`, feedback code `HANDOFF_SUGGESTED`; with an open
   proposal for that line the feedback says it is not done and how to re-present it; for an exchange
   with no size it says to hand off `"UNSPECIFIED"`); never for a rule or status question, after a
   status check, on the agent's last step, or once handoffs are closed. Then the agent's choice
   stands. There is no code handoff.
8. **Operations**: the option table (`list_exchange_options`) names the call that decides (`next:
   create_proposal {...} decides, also when that option is not valid`), so the agent asks the gateway
   instead of judging the table. When Ops ends a propose handoff without a decisive call, code runs the read-only
   `check_proposal` dry run (`fallback/check_proposal`): a refusal is reported (deny); a would-be
   proposal is NOT created — the handoff ends `NO_DECISION` (clarify) and Resolution may hand off again.
   After a TIMEOUT the only allowed Ops tool is `get_request_status`; a code status READ remains only as
   a safety net. On a confirmation turn Operations executes; there is no code execute: scripted FakeLLM
   runs show a correct agent never needs one (tests/unit/test_agents.py), and a missing execute ends the
   turn `NOT_EXECUTED` (error, nothing changed, the proposal stays open for the next confirmation).
9. **Code-made denies (facts only)**: an order the customer typed that is not theirs (`ORDER_NOT_FOUND`);
   a line pinned by typed IDs (or the only line, not contradicted by the description) whose policy
   check refuses it when the agent only asked a question (`guard/code_deny`, reply from the facts
   template; no exchange sizes are ever listed for such a line).

Roles. **Resolution** (customer-facing): interprets the conversation, picks the target, remedy and
requested size, chooses clarify vs handoff, writes the reply. Tools: `find_order_lines(order_ref)`,
`handoff_to_operations` (oneOf per intent: `propose_refund{order_id,line_id,evidence?}`,
`propose_exchange{order_id,line_id,requested_size,requested_product?,evidence?}`, `check_status{}`,
`execute_confirmed{}`), `reply_to_customer{message}`. **Operations** (back office): options, maps the
requested size to a SKU, proposes, executes, reconciles. Tools: `list_exchange_options{}`,
`create_proposal{remedy, replacement_sku?}`, `execute_proposal{}`, `get_request_status{}`,
`report_to_resolution{outcome}`. `create_proposal` runs the eligibility rules; there is no agent
eligibility tool (tuning iteration 2). The gateway's `check_eligibility(order_id?, line_id?)` and
`check_proposal(remedy)` are read-only calls for the orchestrator's blocking checks (item 8, 9).

Context (cache-friendly, append-only within a turn): `[system: static role prompt (+ tool docs in
constrained mode)] → [user: account card (stable per case: each line's ids, SKU, product words, size
words, price paid, delivery date; catalog sizes; today's date; no eligibility verdict) + memo (prior
customer messages verbatim & sanitized, prior decisions, open proposal summary) + current message
block + "Allowed now: …"]` → assistant/tool steps appended. Section headers are constants in
`desk/agents/prompts.py` (the test double imports them). Tool lists are byte-identical every step for
a role; per-step restrictions are enforced by code (and by response_format in constrained mode), never
by changing the tools block. Ops context per handoff = static prompt + structured brief (target
summary, intent, requested size/product, grant flag) — no customer text.

Turn pipeline:
1. Apply faults for this turn. Resolve `confirm_proposal:"last"` → the proposal_id of the latest
   EARLIER turn output with non-null proposal → ConfirmationGrant (this turn only). None → no grant
   (audit `confirmation_unresolved`). The grant is withheld (audit `guard/grant_withheld`) only for a
   real cancel or change (CLARIFICATIONS #9): the cancel cue (§5); both remedies; a STRONG cue for the
   other remedy; a requested size that resolves to another replacement; an exchange message naming a
   size that is neither the item's nor the replacement's, or a refund message naming the size of
   another owned line (`UNREAD_SIZE`); an unknown or different line/order id. A pure cancel (cancel cue
   and nothing new in the message) is reply-only; only then does the reply say nothing was processed.
2. Idempotency preamble: request already completed → the same request (grant for the executed
   proposal, or no different named line/order and no strong wording for the other remedy) →
   deterministic replay: decision completed, original transaction, no effects; otherwise deny
   `REQUEST_ALREADY_COMPLETED`. (Resolution still writes the reply.)
3. Grant turn: Resolution may only `handoff(execute_confirmed)` or reply (a reply before the handoff
   gets `CONFIRMATION_PENDING` feedback); even its last step may hand off. Ops may only
   `execute_proposal`, `get_request_status`, `report_to_resolution`; no proposals this turn.
   Exception (held-out v3, v3_n22/v3_n28): when Resolution proposes another line, remedy or size than
   the bound proposal, that is the agent saying the message changed the request. The grant is
   withdrawn (`guard/grant_withheld`, reason "the agent read a changed request") and the turn runs
   as a normal, unconfirmed turn. A proposal for the same thing is still refused (`TOOL_NOT_ALLOWED`).
   The open exchange offer's own size does not compete with a new size written in the latest
   message (`grounding.changes_the_open_offer`).
   Code also withholds the grant itself when the message names a product on the account other
   than the bound line's (`_grant_conflict`, reason "names a different item"; post-fix bench: B
   executed the jeans refund on v3_n22 because its agent did not read the change).
4. Normal turn: Resolution steps; a propose handoff passes `check_target` → `check_remedy` →
   `check_requested_size`; a block is fed back with code, hint, reason and candidates. A size the
   agent proposes that is written in the latest message, where code reads no size request ("Ugh.
   What about a 30 then?" after the 34 ran out), is not blocked by an older request
   (`grounding.newer_size_in_latest_words`, v3_n23). After
   `Limits.gate_rejections` blocked handoffs only a question is allowed (`guard/handoffs_closed`).
   Protocol rejections (a handoff the turn does not allow) are not counted. Without prefetch (B),
   `find_order_lines` is offered until `find_order_lines("ALL")` has run this turn.
5. Ops loop for the handoff (≤ `Limits.ops_steps`). Duplicate (tool,args) → cached result + "already
   done; next allowed: …". The last allowed step offers only the terminal tool. A handoff that reached
   no decision (`NO_DECISION`, `NOT_EXECUTED`) does not end the agent's turn: within
   `Limits.handoffs_per_turn` Resolution may hand off again.
6. OpsReport built **by code** from tool results: `{intent, outcome, reason_codes, proposal?,
   transaction?, replay, reconciled, alternatives[]}`; the LLM's claimed outcome is compared and a
   mismatch audited. Resolution receives the report (`next` says whether it may hand off again), then
   must `reply_to_customer`.
7. Budgets live in ONE place, `desk.agents.Limits` (`LIMITS`): 12 LLM calls per turn, 5 Resolution
   steps, 4 Ops steps per handoff, 2 handoffs, 2 blocked handoffs, max_tokens 160 (tool steps) / 220
   (reply). LLMError → decision error, no mutation.

Decision table (code; strongest first): **completed** (commit verified by re-reading the store this
turn, or replay/reconcile found the request completed) > **propose** (an open proposal created or
re-presented this turn — always emitted, template reply if the reply step fails) > **deny**
(authoritative refusal for the request: Operations' create/execute/check refusals, the code-made
denies of §6.0 item 9, PROPOSAL_NOT_OPEN, REQUEST_ALREADY_COMPLETED) > **error** (unreconciled TIMEOUT,
NOT_EXECUTED, BINDING_MISMATCH, LLMError, budget exhausted, INTERNAL) > **clarify**.
Gate codes (TARGET_NOT_GROUNDED, TARGET_MISMATCH, AMBIGUOUS_TARGET, LINE_NOT_FOUND inside an owned
order, SIZE_UNSPECIFIED, REMEDY_UNCLEAR, NO_DECISION) → clarify. Errors from LLM-chosen arguments never
count as deny. Informational tools (list_exchange_options) never set deny by themselves.

Business choices (also in CLARIFICATIONS.md): "return"/"send back" alone = refund (the agent's call;
weak words never block). Exchange without a size: ≥2 valid options → clarify listing them; exactly 1 →
propose it (Operations; the clarify feedback says so once); 0 → deny. "The same size" → deny SAME_SIZE
(never the single other option). Requested size out of stock / not offered / price mismatch / same size /
other product → deny, alternatives in text only. Target uniqueness over all matching lines regardless
of eligibility. Verbal "yes" without the flag → propose (re-present the same proposal, explain explicit
confirmation is required). Customer claims never override state. Amount is always paid_cents.

Turn output: `{decision, response, proposal (object iff decision == propose else null), transaction
(object|null), confirmed_proposal_id (str|null)}`. A case that crashes after a persisted commit reports
the crashed turn as `error` naming that transaction (it is in final_state), later turns `not run`.

Response = LLM sentence(s) + code facts line (render.py). Guard rejects LLM text with: amounts that
`render.money` does not format into the facts, digits/IDs/SKUs/sizes not in the facts, success claims
when decision ≠ completed ("complete", "done", "back on your card", "you will receive" unless the
sentence is conditional, "no need to confirm"), promises to act on clarify/deny/error, failure claims
when decision == completed, channels the desk does not have (e-mail, bank statement, label, courier,
"reply yes"), a clarify without '?'. On violation → template reply + guard audit entry. Trim at sentence
boundary (no schema maxLength). A clarify with no item and no remedy in the conversation (small talk)
carries no facts line. Window wording comes from `domain.WINDOW_DAYS`.

## 7. LLM client (desk/llm.py)

`LLMClient(base_url, protocol, *, seed=42, timeout_s=20, retries=1)`;
`step(role, messages, tools: list[ToolSpec], allowed: list[str], max_tokens) -> Step{tool, args,
prompt_tokens, completion_tokens, cached_tokens, prompt_ms, predicted_ms, latency_ms,
protocol_error, dropped_calls}`. Common: `temperature 0`, `seed`, `cache_prompt true`,
`chat_template_kwargs {enable_thinking:false}`, assert no reasoning content. Retry only the HTTP request.
- `native` (baseline): OpenAI `tools` + `tool_choice:"required"` + `parallel_tool_calls:false`; take
  the first call only (extra calls → `dropped_calls`, audited); args validated against the tool schema
  in code (enums, required, no extra props) → `INVALID_ARGS` tool error; counted as protocol_errors.
- `constrained`: never send `tools` (llama-server 400s when combined with response_format); tool docs
  rendered once into the static system prompt; per step `response_format: json_schema` = `oneOf` over
  the allowed tools `{tool: const, args: schema}` with `tool` first; any step offering an ID-bearing
  action also offers `reply_to_customer` as escape. IDs are free strings validated by code (no ID
  enums — they force wrong-but-valid IDs).
`FakeLLM(script)` for tests: scripted Steps or a callable; can emit adversarial steps (two calls,
unknown tool, bad args, execute without grant, wrong IDs, invented success text).

## 8. Telemetry and adapter output

`peak_ram_mb` = Σ kernel high-water `ri_interval_max_phys_footprint` (proc_pid_rusage
RUSAGE_INFO_V4 via ctypes /usr/lib/libproc.dylib; `proc_reset_footprint_interval` at case start) over
the adapter process + llama-server; also `rss_mb` and a per-process breakdown. `peak_vram_mb` =
max over samples (5 Hz + every turn boundary) of IOAccelerator "In use system memory" minus the idle
baseline taken before server launch; `memory_architecture:"unified"`, method/scope notes (system-wide,
overlaps peak_ram_mb — never add them). Mirror as `peak_unified_gpu_mb`.

Root: `contract_version:1, input_sha256, hardware` (STATIC only: cpu "Apple M5 Pro", cpu_cores 15,
gpu "Apple M5 Pro integrated GPU (16 cores)", gpu_count 1, ram_gb 24, unified_memory true, os "macOS
26.4.1 (25E253)", driver "Metal (bundled with macOS 25E253; no separate GPU driver)", power "AC power,
pmset powermode 0 (automatic), sleep prevented"), `configuration{profile, model, model_file,
model_revision, model_sha256, quantization, engine:"llama.cpp llama-server", engine_version,
engine_argv[], protocol, seed, temperature, max_tokens, ctx_per_slot, slots, concurrency:1,
caches{cache_prompt, cache_ram_mib, ctx_checkpoints, persistence:"within a run across cases; cleared
by server restart between runs"}, timeouts{llm_step_s, case_s}, app_switches{…},
run_conditions{battery/power, therm, load avg, other GPU processes} }`, `startup{server_spawn_to_ready_ms,
first_request_ms, model_page_cache, warmup}`, `results[]`.
Each result: `case_id, turns[] (exactly len(case.turns)), action_audit[], final_state, proposals[],
external_events[], metrics{elapsed_ms (reset + all turns, excludes server start), peak_ram_mb,
peak_vram_mb, peak_unified_gpu_mb, rss_mb, memory_architecture, memory_notes, llm_calls,
prompt_tokens, cached_tokens, completion_tokens, prefill_ms, decode_ms, tool_ms, decode_tok_s,
prefill_tok_s (token-weighted), turn_latency_ms[], protocol_errors, guard_overrides, fallbacks,
first_case_after_start, status:"ok"|"failed", error?}`. A crashed case still emits every turn
(decision error, response "not run: …"); the server is restarted for the next case.

adapter.py: stdlib only at import; resolves root from `__file__`; if deps missing, `os.execv`
`<root>/.venv/bin/python` with the same argv, else prints the setup command. `--input --output
[--profile final] [--port 8401] [--llama-url URL]`; auto-starts llama-server (LLAMA_SERVER_BIN or PATH)
unless `--llama-url` is healthy; checks model sha256/engine version; stops the server only if it
started it.

## 9. Profiles (desk/profiles.py) and measurement plan (pre-registered)

Engine flags pinned in EVERY profile: `-ngl 99 -fa on -c 8192 -np 1 --jinja --reasoning off
--cache-ram <MiB> --ctx-checkpoints <n> -t <n> --host 127.0.0.1 --metrics --no-webui --seed 42`.
- **B (baseline)**: Q8_0, protocol native, no prefetch (Resolution must call find_order_lines), Ops
  reply = free-text summary field, cache-ram engine default (8192). A competent first version.
- **F (final)**: candidate set = Q4_K_M + constrained protocol + prefetch account card + structured
  ops report (code-built; LLM only emits `outcome`) + bounded cache-ram. Each change ships only if it
  passes the gate below; otherwise F keeps B's choice.
- Leave-one-out from F: F-q8, F-native, F-noprefetch, F-freetext, F-cacheram-default; plus B-q4 (same
  quant engineering-only comparison) and B-repeat (A/A noise floor). Slot pinning is NOT used: measured
  no-op on 0.7.0 (host prompt cache already restores the other role) — report as a negative result.
- Gate (pre-registered): public success (18 runs) AND variant-suite success ≥ B, zero invariant
  violations (duplicate money/stock, invented success, mutation without grant).
- Each run = fresh adapter process + fresh llama-server; randomized interleaved blocks; preflight
  refuses if foreign llama-server/GPU hogs, battery power, or Low Power Mode; caffeinate.

Reporting (bench/report.py, from raw files only): strict case success vs the independent oracle with
n/N and Wilson CI; pooled p50/p95 (nearest-rank, stated) + per-scenario medians + intersection-of-
successes latency + per-LLM-step distribution; paired F/B geometric-mean ratio with bootstrap 95% CI
(10k, seed 42); time breakdown (calls, uncached/cached prompt tokens, completion tokens, prefill/decode/
tool/overhead ms); token-weighted throughput; failure taxonomy (oracle mismatch, LLM timeout, budget,
protocol_errors, guard fallback, crash; injected faults separate); every attempted run kept; decision-
trace hash identity across the 3 runs. Files: `results/<profile>/run-<k>/{results.json, validate.txt,
server.log, metrics_before.prom, metrics_after.prom, run_conditions.json, argv.txt}`, `results/REPORT.md`.

## 10. Server API (server/app.py)

```
GET  /api/health                      {llm: ok|down, profile, model, engine}
GET  /api/scenarios                   public cases [{case_id, customer_id, title, turns:[…], faults}]
POST /api/sessions                    {case_id?|initial_state?, customer_id?, request_id?} → {session_id, state}
POST /api/sessions/{id}/turns         {message, confirm_proposal_id?, confirm_binding_hash?, generation?} → text/event-stream:
                                      `audit` events, then `turn`. With a confirm_proposal_id the other two are required
                                      (422 if missing) and must match the card and session generation (409, nothing runs)
POST /api/sessions/{id}/faults        {type:"stock_change", sku, new_stock} | {type:"timeout_after_commit"}
GET  /api/sessions/{id}/state         final_state shape + proposals + events + audit
POST /api/sessions/{id}/reset         restore the session's initial state (fresh request_id suffix optional)
GET  /api/telemetry                   last step timings, tokens/s, footprint, unified GPU memory
GET  /api/meta                        {reason_text, code_class, window_days}: the UI words codes like the desk does
```

### 10.1 Agent Mission Control (the "agentic AI" view; required, demo centrepiece)

Owner requirement (2026-10-27): the demo must look and read as an agentic AI system: many agents
with different goals working together, visualised live. The brief adds: "Show tool calls, decisions,
handoffs, and outcomes. Additional agents earn no automatic credit", and the adapter must run the same
agents as the UI. So we add **no** LLM agents. We make the real team visible: 2 LLM agents, 1 code
conductor and 6 deterministic specialists, each with one goal, every card labelled honestly.

**Placement.** The right panel has two tabs: `Team (live)` (the default) and `Audit trail` (the §10 timeline).
A `Theater` toggle (button + key `T`) widens the Team view to about 70% of the window, next to the
conversation, for presenting. Store the theater state in localStorage inside try/catch.

**Nodes** (inline SVG, no libraries, fixed layout; each node is a card showing an icon, a name, a
one-line goal, an `LLM` or `CODE` badge, its live status (idle/working/✓/✗) and per-turn counters):

| node | badge | goal line shown on the card |
|---|---|---|
| Customer | – | describes the request, confirms explicitly |
| Resolution agent | LLM · model+quant chip | understand the customer, ask when unclear, explain the outcome |
| Operations agent | LLM · model+quant chip | check the facts with business tools, prepare and carry out the approved remedy |
| Orchestrator (conductor) | CODE | runs the turn, routes handoffs, decides the outcome from tool results |
| Grounding | CODE | every order, line, product and size the model names must exist on this account |
| Policy | CODE | delivered, ≤30 days from delivery, unopened, not final sale, not already returned |
| Approval gate | CODE | no money or stock moves without the customer confirming this exact proposal |
| Transaction gateway | CODE | one atomic commit (refund/exchange/quarantine), never twice |
| Recovery | CODE | after a timeout, check what actually happened before doing anything |
| Reply guard | CODE | the customer never hears a claim the records do not support |
| Store (SQLite) | data | shows committed effects |
| Warehouse (external) | – | outside events (stock changes) |

LLM-node counters: steps, prompt/cached/completion tokens, ms and tok/s for this turn. CODE-node
counters: checks, blocks.

**Event → node mapping** (from audit entries `kind, agent, tool`; unknown kinds go to Orchestrator):
- `llm_step` + agent `resolution` → Resolution is active. The bubble shows the chosen tool plus a short
  argument summary (e.g. `handoff_to_operations · propose_refund O100/L1`). For `reply_to_customer` the
  edge animates Resolution → Customer.
- `llm_step` + agent `operations` → Operations is active, bubble `check_eligibility()` etc. Show tool
  level only, never model reasoning text.
- `grounding/analyze`, `target/set_target`, `tool/prefetch_account_card`, `gate/handoff_to_operations`,
  `guard/clarify_veto` → Grounding (a gate or veto shows ✗ plus the reason, e.g. "unknown order O999").
  A `set_target` with `grounded_by: "agent"` adds "agent grounded O100/L2 · cites “jacket”" (the
  customer's words the agent cited, §6.0) and an audit badge "grounded by agent".
- `tool/check_eligibility`, `list_exchange_options` and `tool/check_proposal` (shown as "dry run") →
  Policy (✓ eligible / ✗ with plain-language reason codes from `/api/meta`). `guard/code_deny` → Policy ✗
  ("code said no on the facts: …"), and the deny notice names the same reasons.
- `grant/set_grant` → Approval gate ✓ ("customer confirmed P-0001"). `guard/grant_withheld`,
  `confirmation/confirmation_unresolved`, and any tool error whose code is class `gate` → Approval gate ✗.
- `tool/create_proposal`, `tool/execute_proposal` → Transaction gateway. Non-empty `committed_effects`
  flash on Store as plain chips ("refund RF-0001 $30.00", "TEE-L stock 5→4", "quarantine Q-0001").
- `fault/timeout_after_commit` (when visible), `tool/get_request_status`, `route/idempotency_preamble` → Recovery
  ("reconciled: TX-0001 found, no second commit" / "replay: already processed").
- `guard/reply_guard`, `guard/report_mismatch` → Reply guard ✗ ("model said 'processed' on a proposal; template used").
- `fallback/*` (`check_proposal`, `get_request_status`) → Orchestrator ("read-only check by code: …").
  `guard/handoffs_closed` → Orchestrator ("handoffs closed: …"). Code never takes a handoff over.
- `fault/stock_change` → Warehouse → Store edge ("TEE-L stock set to 0").
- `llm_step resolution handoff_to_operations` then `ops_report` → **handoff** edges with a moving
  packet: Resolution → Operations carries the brief (intent, order/line); Operations → Resolution carries the report
  (outcome, reason codes, proposal id).

**Animation.** Feed every SSE `audit` event into a queue and play each one for a minimum dwell of
350 ms, so a 2 s turn still reads clearly. Show a "catching up" hint when the queue backs up. An edge
packet animates from the source node to the target node, and the target pulses. A one-line plain-English
**mission log** under the graph (`aria-live=polite`) narrates each event, e.g. "Operations → Policy: is
O100/L1 eligible? ✓ day 30 of 30". Honour `prefers-reduced-motion`: highlights only, no motion.

**Replay.** After each turn: ▶ Replay with speed control (0.5× / 1× / 2×), plus step back/forward with ←/→
when the Team tab has focus. A turn picker covers earlier turns. Replay uses the stored audit only and
never calls the server.

**Turn summary strip:** "Turn 2 · 7 steps · 2 LLM agents · 4 specialists consulted · 1 block · 1.9 s ·
completed".

**Legend** (always visible): "🧠 LLM agent: Qwen3.5-4B, local llama.cpp" · "⚙ Deterministic specialist:
plain code, no model" · "Agents decide; code can say no."

**Server.** Stream each audit entry as it is recorded: subscribe to the `AuditLog` listeners
(`desk/audit.py`) for the duration of the turn and push to the SSE response through a thread-safe
queue. Do not batch at the end of the turn.

**Tests (fake-policy E2E):**
- every node renders with its badge
- during a confirm turn: the Resolution→Operations edge animates (assert `data-active`/class), Approval gate ✓, and Store shows the refund chip
- expired case: Policy ✗ shows the reason
- timeout scenario: Recovery shows "reconciled"
- replay steps through the same number of events as the audit
- theater toggle works
- reduced motion still shows highlights

Screenshots: `var/screenshots/team_midturn.png`, `team_theater.png` and `team_recovery.png` (fake model,
`scripts/screenshots.sh`); real model, 1440x900 theater: `final_team_refund.png` (public_01 turn 2) and
`final_team_recovery.png` (public_06 turn 2).
