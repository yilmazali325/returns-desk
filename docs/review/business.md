# Business-rules review (25% of the grade)

Reviewer role: business correctness. Review only: no existing file was edited. Everything I wrote is
under `tests/review/business/` and this file.

## Method and commands

The FakePolicyLLM double drives the real orchestrator, gateway, store, guards and render code. It is
the same double the UI uses under `DESK_LLM=fake-policy`. No model or server was needed.

| what | command | result |
|---|---|---|
| unit suite (baseline, untouched) | `.venv/bin/pytest tests/unit` | 1348 passed |
| shipped variant suite through the fake | `.venv/bin/python tests/review/business/biz_run_suite.py evals/variants/variants_v1.json <out>` then `.venv/bin/python -m evals.score --input evals/variants/variants_v1.json --labels evals/variants/labels_v1.json --output <out>` | 241/241 strict, 0 invariant violations |
| 21 new adversarial cases (out of sample) | `.venv/bin/python tests/review/business/biz_build_cases.py`, then the same run+score with `tests/review/business/cases/adversarial_{v1,labels}.json` | **16/21 strict**, 0 invariant violations |
| repro and guard tests | `.venv/bin/pytest tests/review/business -rxX` | 21 passed, 16 xfailed (each xfail is a finding repro, `strict=True`) |
| one case, readable | `.venv/bin/python tests/review/business/biz_harness.py tests/review/business/cases/adversarial_v1.json --case adv19_return_and_send_large` | |

Note on evidence: the shipped variant suite passes 241/241 with no model at all, because the grounding
code was tuned on it (`docs/notes/tuning_log.md` says so in its "variant suite no longer measures the
model" line). The 21 new cases are the out-of-sample check. 3 of their 5 failures are real desk bugs
(B1, B2). The other 2 are desk-vs-oracle disagreements (B5, B6).

## What holds (verified, with passing tests)

- Window: 0 to 30 days inclusive, counted from `as_of`, never from the clock. A timestamped
  `delivered_on` uses its date part. A delivery after `as_of` is denied. Non-ISO dates fail closed.
  (`test_window_is_0_to_30_inclusive_from_as_of_with_timestamps`, adv01, adv02)
- Refund is always `paid_cents`: odd cents (3333), zero (0), a discount below the catalog price.
  Customer claims about the amount are ignored. (adv12, adv17)
- Exchange rules: same `product_id`, different size, same catalog price as the original SKU (XL at
  6000 vs 5500 is denied), positive stock, and numeric half sizes (9 -> 9.5). The replacement is
  decremented exactly once. The original goes to quarantine and its stock never increases.
  (adv04, adv05, adv08, adv16)
- Stale stock and faults: a stock_change applies before its turn as plain assignment, in list order
  (the last one wins), and is recorded as an external event. A change after the commit overwrites the
  stock. A restock lets a repeated request succeed. (adv06, adv14, adv15, adv18, public_05)
- timeout_after_commit fires on the first real commit, not on the first confirm, so a stale-stock
  refusal does not use it up. One execute and one commit happen, recovery reconciles with
  `get_request_status`, and the replay has no new effects. (adv04, adv07, public_06)
- Duplicates: the same line under a NEW request_id after completion is denied `ALREADY_RETURNED`
  with one refund in total. Two request_ids racing on one store commit once, because the later
  commit is refused inside the transaction. (`test_new_request_id_on_an_already_refunded_line_never_refunds_twice`,
  `test_two_request_ids_racing_on_one_store_commit_once`)
- final_state: exactly the input keys in input order, and unknown fields round-trip at every level.
  Records carry customer, order, line, request, proposal and replacement SKU.
  (`test_unknown_fields_and_key_order_round_trip`)

## Findings (ranked by grading impact)

### B1 (high): plain confirmations that contain cancel-type words throw away the confirm flag

`desk/grounding.py:368` `_CANCEL_RE` has no negation handling. It matches bare `\binstead\b`,
`\bstop\b`, `\bwait\b` (only "not wait" is excluded), `cancel\w*` and `changed? my mind`, anywhere in
the message. On a turn with `confirm_proposal: "last"`, `desk/orchestrator.py:268` (`if a.cancel_cue:`)
then withholds the grant. When the message has no remedy or size, `orchestrator.py:240` also treats it
as a *pure cancel*. A second path is `orchestrator.py:272`: a weak "send back" cue counts as "asks for
a refund instead" against an exchange proposal.

Evidence: the real orchestrator plus FakePolicyLLM ran 47 hand-written confirmations on the
public_01 and public_02 states. 12 did not execute. Examples (adv11 is from the case file):

```
"Yes please go ahead, no need to wait."                        -> ['propose','clarify']
"Yes, go ahead. Don't cancel it."                               -> ['propose','clarify']
"Yes, the proposal is right, no need to cancel anything."       -> ['propose','clarify']
"Yes, I accept, I won't change my mind."                        -> ['propose','clarify']
"Yes, I confirm. Please don't stop the exchange."               -> ['propose','propose']
"Yes, go ahead with the refund to my original payment instead." -> ['propose','propose']
"I agree. Please send the large and I'll send back the medium." -> ['propose','propose']
adv11 "Yes, please refund it to my card instead of store credit." -> ['propose','propose']
```

Scorer on adv20: `turn 2: decision expected 'completed', got 'clarify'`, plus
`state.products[POLO-S].stock: expected 4, got 5` and the missing return, exchange, quarantine and
idempotency records. The contract treats the flag as the explicit confirmation, so a hidden variant
that rewords the approval ends with no transaction.

Repro: `test_confirmed_exchange_executes_despite_negated_cancel_words` (7 params), `test_adv11_…`,
`test_adv20_…` (all xfail strict).

Fix:
1. Evaluate cancel cues per clause with the same negation check the remedy cues already use
   (`_live_cues` with `_NEGATED_BEFORE`), extended with "no need to", "don't", "won't" and "never".
2. Drop bare `instead`, `wait` and `stop` from `_CANCEL_RE`. A real change is already caught by
   `last_remedy_cue`, `last_requested_size` and the line-reference checks in `_grant_conflict`.
3. In `_grant_conflict`, block on a different remedy only when the cue is strong.

Re-run this file's tests and the `cancel_with_flag` and `change_with_flag` variant families
(`biz_run_suite.py` plus `evals.score`), which must stay green.

### B2 (high): "return … and send me a large" is processed as a refund, and code overrides a correct model

`grounding.py:848-865`: the only cue in "Return L1 of O100 and send me a large." is the weak "return".
"a large" is not read as a requested size: `analyze` gives `requested_size: None` and
`original_size_hints: ['L']`, although the item is size M. So `remedy_cue = refund`. Then:

- `check_remedy` (`grounding.py:1207`) rejects a correct `propose_exchange` handoff with `REMEDY_MISMATCH`;
- after 2 rejections, `_after_rejections` → `_veto_handoff` (`orchestrator.py:489`) hands off a **refund** itself.

Evidence (`test_gate_does_not_turn_a_correct_exchange_handoff_into_a_refund`, where the model double
picks exchange/L twice):

```
gate orchestrator handoff_to_operations {"code": "REMEDY_MISMATCH", ... "The customer asked for a refund, not that."}
gate orchestrator handoff_to_operations {"code": "REMEDY_MISMATCH", ...}
guard orchestrator gate_override {"reason": "2 rejected handoffs; code grounds the request"}
-> turn 1 proposal: ('refund', 3000)
```

Same refund proposal for "Return line L1 of O100; please ship me the L.", "I'd like to return L1 of
O100 and receive size L." and "Can I return line L1 of O100 and have size L sent?". The scorer on
adv19 (proposal plus confirm) shows a wrong-money/wrong-stock final state:
`remedy expected 'exchange', got 'refund'`, `state.refunds[Z-519]: unexpected {"amount_cents": 5500 …}`,
`state.products[POLO-L].stock: expected 0, got 1`.

Fix:
1. In `_MessageReader`, a size that differs from the grounded line's own size and is not introduced
   by a describing frame should count as a requested size and a weak exchange cue (`remedy_cue` →
   exchange, or `both` → clarify).
2. `check_remedy` should only veto the LLM's remedy on STRONG cues. With weak-only cues it should
   accept the model's choice.
3. `_veto_handoff` must not force a refund when `mentioned_sizes` holds a size other than the
   target's own.

Tests: `test_adv19_…` and the gate test above. Re-score the variant suite (0 regressions).

### B3 (medium): an ineligible line plus "exchange for a different size" can end in clarify, not deny, and the reply advertises sizes

`orchestrator.py:494`: for an exchange with no size and 2 or more valid options, `_veto_handoff`
returns None. A model clarify is accepted, and eligibility is never consulted, although code knows it.
`_clarify_details` (`orchestrator.py:624`) then lists `_valid_options` for an expired line.
The oracle expects `deny` (`build_expectation` → `['deny']`).

Evidence: a scripted Resolution reply on public_03 (31 days) with "I'd like to exchange line L1 of
order O100 for a different size." gives:

```
['clarify'] We need to know which size you would like instead. Item: TEE-M (size M), order O100, line L1.
Sizes available for an exchange: TEE-S (size S), TEE-L (size L). Which size would you like instead?
```

That text implies the item is exchangeable, which is false. The fake-policy double always hands off,
so the variant suite cannot see this. No variant family combines an ineligible line with an exchange
that has no size. A real model, which cannot see eligibility in the account card, plausibly asks.

Repro: `test_ineligible_exchange_without_size_is_denied_even_when_the_model_asks_for_a_size`.

Fix: in `_veto_handoff`, run the code eligibility check for the unique target (through the gateway's
`check_eligibility`, so it is audited). When the target is ineligible, return the exchange handoff so
Ops denies with the eligibility reason. In `_clarify_details`, never list exchange sizes for an
ineligible target.

### B4 (low, oracle shares it): after a completed refund, "exchange the same line for S" is answered `completed`

`orchestrator.py:310` → `record_matches_text` (`orchestrator.py:661`) compares only order and line.
The gateway's own rule (`tools.py:291`, `same("action", target.intent)`) says
`REQUEST_ALREADY_COMPLETED` for the same target with a different remedy. The two layers disagree.
The test shows the gateway returning that code while the turn says `completed`. Labelling an
exchange request "completed" when only a refund happened is close to "inventing a successful
transaction". CLARIFICATIONS #7 and the oracle (`_request` → `same_target` → REPLAY) encode the same
reading, so the suite cannot catch it.

Fix: reuse the gateway's `_matches_completed` logic in the preamble, treating a strong remedy cue
that differs from `record.action` as different. Reword #7.
Test: `test_adv10_…`.

### B5 (low): order IDs reused across customers break both the store and the oracle

The schema does not promise globally unique `order_id`. `store.py:46` keys `returned_lines` on
`(order_id, line_id)` only. Another customer's returned `Z-509/A1` blocks our customer's eligible
`Z-509/A1`: turn 1 proposes (policy filters by customer), and turn 2 denies with "This item has
already been returned." (false). Separately, the oracle's `find_order` (`evals/oracle.py:76`) takes
the first order with that ID regardless of customer and grades `['deny','deny']`.

Fix: add `customer_id` to the `returned_lines` key and seed. The oracle should filter by customer.
Tests: `test_adv09_…`, `test_oracle_adv09_…`.

### B6 (low): desk and oracle disagree on `status` spelling

`policy.py:171` accepts `"Delivered"` (strip + casefold), while `evals/oracle.py:93` requires exactly
`"delivered"`. adv13 → desk `propose`, oracle `deny`. The "independent oracle" therefore grades a rule
the desk does not implement. Pick one reading (strict equality is closer to the contract's literal
value) and align both. Test: `test_adv13_…`.

### B7 (low): public_05 turn 2 has no clarification question

CLARIFICATIONS #2 covers an impossible exchange *at request time*. It does not cover a confirmed
proposal that fails revalidation, which is exactly public_05 turn 2. The desk answers `deny` with
the alternatives in the text, and graders could read it as `error` or `clarify`. Add the question
to `docs/CLARIFICATIONS.md` so the answer is shared early.

## Contested defaults the oracle shares (not bugs, but say so in the deck)

The oracle encodes CLARIFICATIONS #1 (timeout turn = completed), #2 (impossible exchange = deny),
#5 (exactly one valid size gets proposed), #6 (clarify even if only one match is eligible) and #7
(B4). If an interviewer answers any of these differently, desk and oracle are both wrong together.
The scorer's `alternatives` covers only #1, LINE_NOT_FOUND and cancel.

## Files

- `tests/review/business/biz_build_cases.py`: builds `cases/adversarial_v1.json` (21 cases, contract
  schema), `cases/adversarial_intents.json` and `cases/adversarial_labels.json` (oracle labels).
- `tests/review/business/biz_harness.py`, `biz_run_suite.py`: run cases through `run_case` with
  FakePolicyLLM, and write adapter-shaped results for `evals.score`.
- `tests/review/business/test_biz_rules.py`: 21 guards and 16 strict-xfail repros.
