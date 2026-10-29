# Live reliability tuning log

Stage owner: live reliability tuning (desk/ and tests/unit/). Goal: maximise strict complete-task
success on the 6 public cases (x3 fresh runs) and on the independent variant suite
(`evals/variants/variants_v1.json` + `labels_v1.json`, 241 cases / 486 turns / 64 families) for both
profiles, F (Q4_K_M, constrained, port 8402) and B (Q8_0, native, port 8403), without special-casing
fixture IDs, phrases or counts and without weakening any safety gate.

Method per iteration: run the adapter -> `validate_submission.py` -> `python -m evals.score` -> read
the failing diffs and the action_audit -> classify each failure (LLM misread, gate too strict, gate
too loose, prompt gap, policy bug, render/guard bug, label bug) -> fix the general cause -> add a
regression unit test (FakeLLM script or grounding table row) -> rerun.

No label was changed in this stage (`docs/notes/label_changes.md` was not needed).

## Commands

```
# full variant suite, one profile (fresh adapter process + fresh llama-server each run)
.venv/bin/python adapter.py --input evals/variants/variants_v1.json --output var/tuning/<run>.json \
    --profile F --port 8402            # B: --profile B --port 8403
.venv/bin/python -m evals.score --input evals/variants/variants_v1.json \
    --labels evals/variants/labels_v1.json --output var/tuning/<run>.json --json var/tuning/<run>_score.json
# public set
.venv/bin/python adapter.py --input candidate_assets/public_cases.json --output var/tuning/<run>.json --profile F --port 8402
.venv/bin/python candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output var/tuning/<run>.json
.venv/bin/python -m evals.score --input candidate_assets/public_cases.json \
    --labels evals/expected/public_expected.json --output var/tuning/<run>.json
```

Reliability metrics (per run, summed over cases) come from each result's `metrics` and
`action_audit`: `guard_overrides` (audit kind `guard`: reply guard, grant withheld, clarify veto /
override, report mismatch, duplicate call, dropped calls), `fallbacks` (kind `fallback`: code made a
call the model did not), `protocol_errors`, `clarify_override` (code made the grounded handoff after
the model refused twice), mean LLM calls per case and median case latency (`metrics.elapsed_ms`).
The commands write to scratch (`var/tuning/`, git-ignored); the raw files of every run cited below were
copied to `results/evidence/tuning/` (tracked).

Timing caveat: during iterations 1-3 an F run and a B run sometimes shared the GPU (two
llama-servers on 8402/8403), which roughly doubles latency. Latency numbers quoted as final come
only from runs with one server alone on the machine (marked "alone").

## Iteration 0 - starting point (code from the agents-adapter stage)

F, full variant suite (alone for the first ~45 cases, then alone after B was stopped):
**237/241 strict (98.3%, Wilson 95.8-99.4%)**, 0 invariant violations, 482/486 turns.
B: the first full run was stopped after 5 cases because two servers sharing the GPU made B take
~60 s per case; B was rerun in iteration 1.

| case | expected | got | class | root cause |
|---|---|---|---|---|
| exchange_different_product_02 "for a pair of jeans in size 32" | deny | clarify | grounding gap | "pair of" broke the for-trigger scan, so JEANS was read as the item owned and 32 as its size; code saw no other-product request |
| exchange_no_size_multi_03 "Let's go with the size 30." | clarify, propose, completed | clarify x3 | grounding gap | "go with" was not a request cue and "the size 30" (determiner) was read as the size the customer has |
| change_with_flag_03 "could it be the small rather than that?" (flag set) | propose, propose, completed | propose, clarify, completed (old size executed) | grounding gap | "S rather than" was not read as the requested size, so turn 3's flag executed the old L proposal: a wrong mutation in the strict sense |
| metamorphic_rename_18 "money back for L9 in order ORD-20001" | clarify (LINE_NOT_FOUND) | deny (ORDER_NOT_FOUND) | grounding gap | an unrecognized ID with an unknown prefix defaulted to "order", even with an account order named in the same message |

Reliability (F, 241 cases): guard overrides 72 (reply_guard 24, report_mismatch 30, clarify_veto 10,
grant_withheld 8), fallbacks 59 (create_proposal 58, clarify 1), protocol_errors 0,
clarify_override 0, LLM calls 2108 (8.75 per case), median case 4.70 s, turn p50 2.35 s.
Ops step pattern: `check_eligibility` was the first Ops call on all 257 propose handoffs (85 s of the
1085 s total LLM time); 48 handoffs then reported without ever calling the deciding
`create_proposal`, and 10 more called it only with arguments code rejects as a slip, so code made
the deciding call 58 times. (The first ~70 cases of this run shared the GPU with a B run for about
5 minutes, so its latency is slightly inflated; the paired comparison in the final section uses only
cases 81-241.)

## Iteration 1 - grounding (desk/grounding.py)

Changes (general rules, each with a table test in tests/unit/test_grounding.py):
1. Quantity words: "pair"/"pairs" are fillers and "of" after them is skipped, so "for a pair of
   sneakers in size 10" reads as a request for another product in size 10 (deny DIFFERENT_PRODUCT).
2. "go with" introduces a choice ("let's go with the size 10" -> requested 10).
3. A size followed by "rather than" is the requested one, like "instead" ("the small rather than
   that" -> requested S). On a flag turn this now withholds the grant ("asks for size S") and the
   new size is proposed; the next confirm executes S (orchestrator FakeLLM test added).
4. When a message names one of the customer's orders, a second unrecognized ID-shaped token is a
   line of it (LINE_NOT_FOUND -> clarify) instead of a second order (deny). Without a named order the
   old safe default (treat as order -> ORDER_NOT_FOUND) is unchanged.
5. Typo tolerance for the two strong remedy words: a 5+ letter word one edit (insert, delete,
   substitute, adjacent swap) from "refund" or "exchange" is normalized to it ("refnd", "exchnage").
   Before, "refnd ... plz" left the remedy to the model alone (the cue was "none", so the remedy gate
   trusted the LLM and the clarify veto could not apply).

Offline check (no model; script compares `analyze()` per turn with the label intents): grounding now
agrees with the labels on every refund/exchange/clarify_needed turn of the suite (296 turns); the 14
remaining "differences" are the off-account targets (foreign / nonexistent order, unknown line) that
code denies or clarifies by design, and their not-found codes all match the labels.

F, targeted subset (the 4 failures + their whole families + typo cases, 24 cases):
`results/evidence/tuning/it1_F_sub1.json` -> **24/24 strict**, 0 violations.

## Iteration 2 - Operations step economy and two prompt wordings

Root cause (from the iteration-0 audit, class "prompt gap / gate too loose"): the Ops prompt already
said `create_proposal` checks eligibility, yet the model called `check_eligibility` first on every
propose handoff. That step never decides anything (SPEC §6: informational tools never deny), costs
one LLM call per proposal turn, and after an ineligible answer the model reported "denied" without
the deciding call, so code made it (the 58 `fallback:create_proposal`).

Changes:
1. `desk/agents/operations.py::allowed_tools`: a propose handoff offers `create_proposal` and the
   report (refund), or `list_exchange_options`, `create_proposal` and the report (exchange).
2. Resolution prompt: "Never say anything is refunded, exchanged, processed or done unless the
   report says completed. A proposal is only an offer the customer still has to confirm." (5 of the
   24 reply-guard trips were "I've processed your refund request" on a propose turn.)
3. Ops prompt: outcome definitions ("denied (a rule or the current stock refused it)", "failed (a
   technical problem only)"): 18 of the 30 report mismatches were "failed" for an OUT_OF_STOCK /
   ineligible answer.

F public (sharing the GPU with a B run, so latency is not comparable): 6/6 strict, LLM calls 54 ->
49, fallbacks 1 -> 0, guard overrides 1 (a report mismatch).

## Iteration 3 - remove the eligibility tool from the Ops agent

With iteration 2 the Ops `check_eligibility` tool was never offered at any step, i.e. dead in the
agent. It is now removed from the Ops tool list (`desk/agents/toolspecs.py`); the gateway keeps
`check_eligibility` (UI, inspection, unit tests). A model that still asks for it gets `UNKNOWN_TOOL`
and goes on to `create_proposal` (FakeLLM regression test
`test_operations_has_no_separate_eligibility_tool`). Deviation from SPEC §6's Ops tool list, recorded
below.

## Iteration 4 - not-found wording may repeat the id the customer typed (render/guard)

Class "guard too strict": 14 of F's 24 reply-guard trips were the model politely echoing the order or
line id the customer had just typed ("I can't find order O200 on your account") on a deny/clarify;
the facts line said "that order", so the id counted as invented and the template replaced the reply.
Now, when exactly one unknown order (or line) id was named, the reason wording repeats it ("We could
not find order ORD-99 on your account." / "We could not find LN-9 in that order."). The wording is
identical for an order that does not exist and another customer's order, so nothing leaks
(test_render + two orchestrator tests). With two or more unknown ids the generic wording stays (code
cannot tell which one is the order).

## Interim decision-level check of the iteration-1/2 code on the full suite

The full runs `it1_B_var` (B, iteration-1 code) and `it2_F_var` (F, iteration-2 code) were stopped at
130 and 136 of 241 cases when a foreign CPU-heavy job (another project's scheduled pipeline, load
average ~27) slowed both to ~30 s per case; their per-case decision lines were compared with the
labels: **0 decision mismatches in 130 (B) and 136 (F) cases**. Final numbers come from the final runs
below, on the final code.

## Code-only floor (no model at all)

Script: a FakeLLM that only ever asks "Could you tell me a bit more?" and whose Operations agent
reports `needs_info` at once, run through the real `run_case` (`codefloor.py`, scratchpad). Code alone
(grounding, clarify veto/override, Ops fallbacks, decision table) scores **241/241** on the variant
suite (F and B profiles), 6/6 public, 0 violations. Before this stage the same floor was 233/241.
Reading: on this suite, the model's job is the wording of replies and the cases code cannot ground;
every decision is also derivable by code. That is by design (the LLM proposes, code disposes), but it
also means the variant suite no longer measures the model much, and grounding was tuned on it.

## Fresh-phrasing holdout (written after tuning, before any run on it)

To get evidence that is not tuned-on, 46 new conversations were written (one source case per
family; same state, flags, faults and intents as the source, entirely new customer wording: 92
turns), saved as `results/evidence/tuning/holdout_v1.json` with the source labels (labels depend on intents and
state only). Results before any fix:
- grounding audit: 1 real miss, "Let me have the small please." (requested size not read: "have"
  was not a request cue and "the small" read as the owned size);
- code-only floor: 45/46 (that case);
- live F / B on the pre-fix code snapshot: see the final table.

## Iteration 5 - "have" in a request frame (grounding)

"let me have / can I have / could I have / may I have / I will have / I would have" + size now
marks the requested size; a bare "I have the medium" still describes the item (4 table tests).
After the fix the holdout code-only floor is 46/46. The holdout is therefore no longer clean for
grounding after iteration 5; the pre-fix numbers are the honest ones.

## Iteration 6 - two Resolution prompt bullets (shipped; effect on vetoes not measurable)

Class "prompt gap": in the clarify-veto cases the model asked "refund or exchange?" although the
remedy was in an earlier message, or judged availability itself ("our catalog only offers S, M, L,
would you like L instead?") instead of handing off. Added: "Read the whole conversation: an order,
line, remedy or size the customer gave in an earlier message still holds unless they change it." and
"When the line and the remedy are clear, hand off even if you think the size or product is not
available: Operations checks it and the desk explains." Result in the final F run: clarify vetoes
10 -> 11 (the Q4 model still asks; the veto costs one extra Resolution step per case and changes no
decision). Kept because it states the intended behaviour for both models and did not regress
anything; listed as an open issue below.

## Final runs (final code = iterations 1-6; `results/evidence/tuning/final/`)

Script `final_runs.sh` (scratchpad): each run is a fresh adapter process + fresh llama-server, one
server on the machine at a time, in this order: F public x3, F variants, B public x3, B variants, F
holdout, B holdout. Every output passed `validate_submission.py` ("STRUCTURE CHECK PASSED"). The
machine is shared: another project's scheduled video job (ffmpeg at ~430% CPU, load average 10-13)
overlapped the B public runs and the first part of the B variant run, so B latency is inflated
relative to F; `configuration.run_conditions` in each results file records load and heavy processes.

| run | strict | invariant violations | guard overrides | fallbacks | protocol errors | clarify_override | LLM calls / case | median case | turn p50 | load (before) |
|---|---|---|---|---|---|---|---|---|---|---|
| F public run 1 | 6/6 | 0 | 0 | 0 | 0 | 0 | 8.17 | 4.46 s | 1.87 s | 3.7 |
| F public run 2 | 6/6 | 0 | 0 | 0 | 0 | 0 | 8.17 | 4.49 s | 1.87 s | 3.3 |
| F public run 3 | 6/6 | 0 | 0 | 0 | 0 | 0 | 8.17 | 4.56 s | 1.87 s | 2.8 |
| **F variants** | **241/241** | 0 | 37 | 21 | 0 | 0 | 7.89 | 4.05 s | 2.11 s (p95 3.02 s) | 2.9 |
| F holdout (46) | 46/46 | 0 | 12 | 4 | 0 | 0 | 7.76 | 3.77 s | 2.02 s | 1.7 |
| B public run 1 | 6/6 | 0 | 0 | 0 | 1 | 0 | 9.50 | 10.95 s | 4.26 s | 11.9 |
| B public run 2 | 6/6 | 0 | 0 | 0 | 1 | 0 | 9.50 | 11.09 s | 4.28 s | 9.4 |
| B public run 3 | 6/6 | 0 | 0 | 0 | 1 | 0 | 9.50 | 10.64 s | 4.20 s | 12.7 |
| **B variants** | **241/241** | 0 | 49 | 15 | 23 | 1 | 9.19 | 9.87 s | 5.30 s | 11.7 |
| B holdout (46) | 46/46 | 0 | 8 | 3 | 3 | 0 | 8.96 | 9.42 s | 5.14 s | 1.2 |

Public: **F 18/18 and B 18/18 strict**, identical decisions in all 3 runs of each profile.
Holdout before iteration 5 (pre-fix code snapshot, the honest held-out number): F 45/46, B 45/46,
both missing the same case ("Let me have the small please."; the model read S correctly, the size
gate rejected it because grounding had not, so the gate was too strict there). After iteration 5:
46/46 for both (no longer held out for grounding).

Guard / fallback composition:
- F variants: clarify_veto 11, report_mismatch 11, grant_withheld 8, reply_guard 7;
  fallback create_proposal 21 (the model saw the option table for an impossible exchange, reported
  `needs_info` and code made the deciding call; no extra LLM step).
- B variants: reply_guard 20, report_mismatch 8, grant_withheld 8, clarify_veto 6, duplicate_call 5,
  clarify_override 1, gate_override 1; fallback create_proposal 13, get_request_status 2. Protocol
  errors 23: 18x `get_request_status` after an execute when only the report was allowed, 3x
  `create_proposal` and 2x the report outside the allowed step (native mode cannot enforce the
  per-step list in the decoder; constrained F has 0).
- `grant_withheld` (8 per profile) are the cancel/change-with-flag families doing their job.

Before / after (same suite, same profile):

| metric | F iteration 0 | F final | B (previous stage, public only) | B final |
|---|---|---|---|---|
| variant strict | 237/241 | 241/241 | not measured | 241/241 |
| public strict (per run) | 6/6 | 6/6 x3 | 6/6 | 6/6 x3 |
| guard overrides (variants) | 72 | 37 | - | 49 |
| fallbacks (variants) | 59 | 21 | - | 15 |
| protocol errors (variants) | 0 | 0 | - | 23 |
| LLM calls (variants) | 2108 (8.75/case) | 1901 (7.89/case) | - | 2214 (9.19/case) |
| LLM calls (public, one run) | 54 | 49 | 60 | 57 |
| median case (variants) | 4.70 s | 4.05 s | - | 9.87 s (loaded machine) |

Paired latency, F, cases 81-241 only (where iteration 0 ran alone): median case 4.85 s -> 4.61 s,
summed 797.0 s -> 754.5 s, paired geometric-mean ratio 0.947 (-5.3%), LLM calls 1586 -> 1417 (-10.7%).

## Which families still fail, and why

- Variant suite: none, for either profile (64/64 families strict). The code-only floor is also
  241/241, so this suite now mostly measures code: it cannot tell a strong model from a weak one.
- Fresh-phrasing holdout: pre-fix 45/46 for both profiles (exchange_no_size_multi phrasing, fixed in
  iteration 5). This is the best evidence that unseen wording is the remaining risk: grounding is
  lexicon/pattern based, and a phrasing it misses either loses a clarify-veto (the model must get it
  right alone) or, worse, makes a size/remedy gate reject a correct model reading (that is what
  happened here). Mitigation in place: after 2 gate rejections the turn falls back to a clarify,
  never to a guess, so a grounding miss costs a turn, not a wrong mutation.

## Deviations from SPEC (with reason)

1. **Ops tool list (SPEC §6)**: `check_eligibility` is no longer an Operations tool. Reason: it never
   decides (SPEC §6), the model called it first on every propose handoff (8% of LLM time) and then
   often skipped the deciding call (58 fallbacks). `create_proposal` runs the same eligibility rules.
   The gateway method stays (UI, inspection). Per-step allowed lists for propose handoffs are now
   `[create_proposal, report]` (refund) and `[list_exchange_options, create_proposal, report]`.
2. **Grounding rules (SPEC §5) extended**: one-edit typo normalization of "refund"/"exchange";
   request cues "go with", request-frame "have", "<size> rather than"; "a pair of" skipped; an
   unrecognized ID next to a named account order is a line (LINE_NOT_FOUND), not an order.
3. **Not-found wording**: ORDER_NOT_FOUND / LINE_NOT_FOUND repeat the single id the customer typed
   ("We could not find order ORD-99 on your account."). SPEC's "identical wording for foreign and
   nonexistent orders" still holds (nothing about another customer is revealed).
4. **Prompts**: three wording additions (Resolution: "processed" is a success word, read the whole
   conversation, hand off even if you think the size is unavailable; Ops: outcome definitions).
5. No label changed; `docs/notes/label_changes.md` not created.

## Open issues for the next stage

- **Clarify vetoes with Q4 (11 / 241)**: the model asks "refund or exchange?" when the remedy was in
  an earlier message. Code handles it (one extra Resolution step, ~0.6 s). Possible fix: a code-built
  "request so far: refund of order X line Y" line in the memo; not done because it hands the model
  code's reading and could hide grounding errors.
- **B protocol errors (23)**: native mode ignores the per-step list, mostly a `get_request_status`
  after an execute. Allowing the status check after an execute would remove them in B but invite an
  extra (useless) step in F; left as is, and it is a fair baseline-vs-constrained talking point.
- **Exchange denials via fallback (F 21, B 13)**: the Ops model reports `needs_info` after seeing the
  option table instead of calling `create_proposal`. Removing the report from the allowed list until
  a decisive create would make it 0 but leaves the model no choice at all on that step.
- **Reply guard on REQUEST_ALREADY_COMPLETED denies**: the model writes "refunds can only be processed
  for one item per order" (misleading; the guard swaps in the template). The desk note for this case
  itself uses "processed"; rewording it may help.
- **Held-out evidence**: the variant suite and this holdout have both been seen now. Before the final
  bench, a second fresh holdout written by someone who has not read grounding.py would be the most
  honest generalization number.
- **Latency comparison F vs B in this log is not clean** (B ran while another project's ffmpeg job
  loaded the CPU). The bench stage's preflight should refuse such runs (SPEC §9).

## Review fixes (2026-10-28): "Agents say yes; code can say no" (SPEC §6.0)

Why: the verified review (docs/review/*.md, RT-1 / code C1) showed that an agent that does nothing
(`evals.holdout.run_stub --mode always-clarify`) scored exactly like the real model on every suite,
because code re-read the language, overrode correct model choices and made handoffs/proposals itself.
All measurements below are deterministic (FakeLLM / FakePolicyLLM / the stub); no model was run.

What changed (code):
- Authority split: agents choose line, remedy, size, ask-or-act and wording; only an agent starts a
  business action. Code only BLOCKS (fact gates, one-time clarify feedback, fact-based denies), READS
  (read-only `check_proposal` / `check_eligibility`, status after a timeout) or REPLAYS.
- Removed: `guard/clarify_override`, `guard/gate_override`, the final-step forced handoff, the grant
  turn's code handoff (`fallback/handoff_to_operations`), Ops `fallback/execute_proposal` and
  `fallback/create_proposal`, `fallback/clarify` (now `guard/handoffs_closed`, no code action).
- New: citation (`evidence`) gate with `grounded_by` audit; `AMBIGUOUS_TARGET`, `NO_DECISION`,
  `NOT_EXECUTED`; `guard/code_deny`; `fallback/check_proposal`; `Limits` in one place.
- Language layer: negation-aware per-clause cancel cue; weak words never flip a remedy; hedged
  return-or-swap = both; fit words and offer/question frames for sizes; possessive size = hard filter,
  loose size that empties the list is ignored; "same size" = SAME → SAME_SIZE deny; numeric size after
  for/to/into when offered; money masked before sizes; policy/status questions marked; one size
  vocabulary/resolver in policy.py; account card shows price paid, delivery date and today's date;
  generic special-token sanitizer; stricter reply guard; one money formatter; window wording from
  WINDOW_DAYS; returned_lines keyed by customer; crash-after-commit reported honestly.

Measured (strict cases / turn accuracy / invariant violations; `evals.score`):

| run | public | variants_v1 | holdout_v2 | completed turns (pub/var/hold) |
|---|---|---|---|---|
| always-clarify stub, before | 6/6 | 241/241 | 45/57 | 5 / 156 / 37 |
| always-clarify stub, after | **1/6** (0.154) | **54/241** (0.187) | **11/57** (0.178) | **0 / 0 / 0**, 0 refund/exchange records |
| FakePolicyLLM, before | 6/6 | 241/241 | 45/57 | |
| FakePolicyLLM, after | **6/6** | **241/241** | **50/57** (0.896) | 0 invariant violations in all three |

- What the stub still gets right is only what code may say on facts: denies of a line the customer
  named by its ids that policy refuses (code_deny: pub 1, var 36, hold 7), foreign/unknown orders,
  and clarifies. It never proposes, so it never completes.
- FakePolicyLLM's 7 holdout misses are all reading tasks its regex brain cannot do (recency, "the large
  back" among two tees, "mediun", "30s for 32s", "chaqueta", "talla L", a 3-product ramble). With the
  real agent's reading, the gates now accept 6 of them (evals/holdout/test_redteam_findings.py: 15/16
  pass; hx12 stays open because code cannot verify the Spanish "la chaqueta" as a citation).
- Post-hoc probes (evals/holdout/probes): FakePolicyLLM 7/7 (review baseline: code-only 1/7).
- Adversarial business cases (tests/review/business): 18/21 (review baseline 16/21). The 3 misses are
  oracle-side: adv09 (the oracle's own B5 bug, it also triggers its 2 "invariant violations"), adv10
  (the oracle still encodes the old replay reading of B4), adv13 (B6, not in this change).
- Reply guard on stored live replies (out_F/out_B holdout, F/B variants, F holdout, F public): 1,301
  accepted replies re-checked, 2 now rejected, both intended (RT-8: "check your email or bank
  statement", "reply 'yes' to proceed").
- Removed fallbacks vs the stored real runs: F variants had 21 `create_proposal` fallbacks, all
  refusals (OUT_OF_STOCK 13, SAME_SIZE 3, NO_EXCHANGE_AVAILABLE 3, SKU_NOT_FOUND 2); the read-only check
  reports the same denies. Grant/execute fallbacks: 0 in every stored real run (public F/B x3,
  variants F/B, holdouts) and a correct scripted agent never needs one, so they were removed.

Risks for the live run (not measured here; the next stage runs the model):
- The model must now hand off `execute_confirmed` itself on confirm turns (historically always did).
  If it ever does not, the turn is `error` NOT_EXECUTED (no wrong state) and the next confirmation works.
- On multi-line accounts without typed ids the model should send `evidence`; when it does not, the
  handoff is still accepted if code's own reading agrees, else it gets AMBIGUOUS_TARGET feedback once
  per attempt. Confidence medium that variants stay at 241/241 live (variants were tuned with
  code-forced handoffs that no longer exist); high that public stays 6/6 (all ids typed).

## Live re-tuning after the authority rebalance (2026-10-28, `var/tuning2/`)

Evidence: the final runs (`final3/`), the code-only floors (`floor_final/`) and the run and scoring
scripts are tracked in `results/evidence/tuning2/`; intermediate runs stay in `var/tuning2/` (git-ignored).

Stage owner: live re-tuning. Ports 8440 (F) and 8441 (B), one llama-server on the machine at a time.
Diagnostic runs (`d1`, `i1`-`i3`) attach to one persistent F server (`adapter.py --llama-url`, or
`var/tuning2/live.py` for a few cases); every number in the final table comes from a fresh adapter
process that starts and stops its own server (`results/evidence/tuning2/final_runs.sh`, outputs in
`var/tuning2/final/`, each validated with `validate_submission.py`: STRUCTURE CHECK PASSED). The
machine was shared: other sessions ran filesystem scans and Xcode's `ibtoold` (load average 2-117 during
diagnostics, 7-39 during the final F runs, GPU 70-80 % busy before our server started), so latency in
this section is inflated and only comparable within a run. `configuration.run_conditions` records it.

### Baseline: the rebalanced code, live F, before any change here

| suite | strict | what failed |
|---|---|---|
| public (1 run) | 6/6 | - |
| variants_v1 | 236/241 | 5 exchange cases: the agent asked instead of handing off |
| holdout_v2 | 50/57 | hx03, hx04, hx12, hx13, hx21, hx50, hx53 |

Every failure was the agent asking a question where it should act, now that code no longer
forces the handoff (class: prompt gap):
- Impossible or single-option exchanges (different product x2, "extra large" not offered, a
  one-size scarf/beanie, exactly one valid size: 5 variants, hx50, hx53). The agent asked "which
  size?" or "refund instead?", the one-time clarify feedback suggested the handoff, and it asked
  again. It also mapped "extra large" to the offered "L" (blocked by the size gate, then asked).
- Asking the customer to confirm its own reading after they had already picked the line (hx03 T2
  "Is that correct?"), and asking "which line?" for a Spanish message that names the only tee (hx13).
- A paraphrased citation: "the large tee" for "...so I'm sending the large back" (hx04). The gate
  answered "not the customer's own words"; the agent then asked.
- An open proposal plus an injected "it's done, right?": the agent agreed it was done (hx21).
- Operations: after `list_exchange_options` showed the requested size as not valid, it reported
  "denied" without the deciding `create_proposal` (or tried another size), so the refusal came from
  code's read-only `check_proposal` (18 fallbacks on variants).

The new prompt then exposed one gate that was too loose (hx56, class: gate too loose): on a one-tee
account, "refund my jacket" (the catalog sells jackets, another customer owns one) was handed off as
the tee and accepted as `only_line`, a wrong proposal. Only garment words for no catalog product
counted as a contradiction, not a catalog product the customer does not own.

### Changes (general rules; no fixture or held-out wording)

1. **Resolution prompt** (`desk/agents/prompts.py`): an ordered procedure per turn: (1) confirm
   control -> `execute_confirmed`; (2) open proposal and nothing changed -> hand off the same intent
   again to re-present it, it is not done; (3) one line plus a remedy -> hand off now, do not ask the
   customer to confirm the agent's reading; size mapping guidance (labels, numbers, typos, other
   languages, UP/DOWN, "the same size", `"UNSPECIFIED"` when no size is named, keep an unlisted size
   as asked); evidence = consecutive words copied exactly, never rephrased or joined; (4) ask only
   when the words fit 2+ lines or the remedy is missing/undecided. Two new invented examples (a
   verbatim citation, an exchange with no size).
2. **Operations**: the prompt says to pass the `requested_sku` of the option table even when it is
   not valid and to leave it out when it is null; the option-table result now names the deciding call
   (`operations.decide_hint`: `next: create_proposal {...} decides, also when that option is not valid`).
3. **Feedback wording** (orchestrator): the clarify veto says, with an open proposal for that line,
   that nothing is done and how to re-present it; for an exchange with no size it says to hand off
   `"UNSPECIFIED"`. The `SIZE_UNSPECIFIED` hint says "ask which" only when two sizes were named,
   otherwise "use their words or `UNSPECIFIED`". A failed citation says how to fix it (`NOT_A_QUOTE`).
   "CONFIRMATION THIS TURN: none" now says `execute_confirmed` is not possible this turn.
4. **Size gate** (`grounding.check_requested_size`): when code read no requested size, the agent's
   concrete size must be written by the customer in some form (label or size word, one typo in a
   5+ letter size word, a number or plural number; money and IDs excluded). A size nobody wrote, or
   the item's own size, is a guess -> `SIZE_UNSPECIFIED`. Before, a guessed valid size went to
   Operations and became a proposal for a size the customer never asked for.
5. **Target gate** (`grounding._description_contradiction`): on a one-line account, a named catalog
   product the line is not is a contradiction (`TARGET_MISMATCH`), like an unknown garment word.
6. **B6** (`policy._is_delivered_status`): the status must be exactly `"delivered"` (the contract's
   literal value; the oracle reads it the same way). adv13 now agrees.
7. After the first full final run (`var/tuning2/final/`) four more changes, then everything was
   re-measured on the frozen code (`results/evidence/tuning2/final3/`; `final2/` is an aborted intermediate):
   - **Other product** (`grounding.check_requested_size(..., requested_product)`, class: gate too
     strict). B handed off "for a pair of jeans in size 32" on a hoodie as requested_product "jeans",
     size "UNSPECIFIED"; the size gate answered REPLACEMENT_NOT_REQUESTED twice (32 is no hoodie size)
     and the turn ended clarify instead of deny DIFFERENT_PRODUCT. When the agent's product is another
     catalog product the customer asked for, no size can make the exchange valid, so the size check
     is skipped. Replaying every recorded exchange handoff of the first final run through the old and
     new gate (`results/evidence/tuning2/replay_size_gate.py`, 342 handoffs, F and B, all suites) changes exactly
     the 2 verdicts of that case.
   - **Reply rule for a proposal** (class: prompt gap, B only). With the restructured prompt the Q8
     native model wrote "Your refund ... has been processed" on 27 propose turns of the variant suite
     (previous stage: 1); the reply guard replaced each with the template, so no decision changed.
     Step 5 now says what to write per outcome (proposed: offered, must be confirmed, nothing is
     processed yet). On 25 propose-heavy variant cases B's propose-turn guard trips went 7 -> 0 with
     the same decisions (`var/tuning2/i4_B_replies.json`).
   - **What check_status is for** (class: tool-doc gap). In `final2`, adv14 turn 2 ("Can you check
     size 10 again for line A1 of Z-514?" after a restock) became a `check_status` handoff, so no new
     proposal was made. The handoff tool description and the prompt now say check_status is only for
     "did it go through?" about a confirmed request, and a stock re-check is the exchange itself. On 23
     status / restock / stale-stock cases (adv, holdout, variants) F then matched every label
     (`var/tuning2/i5_F_status.json`).
   - **No lookup loop in the baseline** (`orchestrator._allowed`, class: protocol gap, B only). In
     final3's first B variant run the native Q8 model called `find_order_lines("ALL")` five times in a
     row (the duplicate guard answered `already_done` each time) until the step budget ended the turn
     as an error. Once every line was listed, `find_order_lines` is no longer offered that turn. F has
     the account card prefetched and never offers the lookup, so F's code path is unchanged and only
     B was re-run (`results/evidence/tuning2/b_runs.sh`).

Unit tests: new rows in `test_grounding.py` (size-written table, catalog-product contradiction),
`test_orchestrator.py` (veto wording with an open proposal / no size, size hint branches),
`test_agents.py` (`decide_hint`), `test_policy.py` (strict status). 1592 passed, 100 % line and
branch coverage of `desk/` (4716 statements, 1492 branches).

### The evidence gate held

In the final F variant run (`final3`) the agent handed off a guess on 2+-line accounts 27 times in
26 cases ("my medium tee" with two medium tees, "something in order X", "L1" when two orders have an
L1, "the one I got recently" inside "I'm not sure..."). The citation gate rejected every one
(`AMBIGUOUS_TARGET`: fits more than one line / hedged / names nothing distinguishing); every one of
those turns is labelled clarify: 0 wrong proposals. (The diagnostic run `i2`: 26 in 25 cases, same
result.) The accept-any-candidate regression
(223/241) does not reappear.

### Final results (frozen code, `results/evidence/tuning2/final3/`, fresh adapter + fresh llama-server per run)

`python -m evals.score` per suite (live2/live3 have only decision lists: strict = decisions, plus the
replacement SKU for live3). Floors: `evals/holdout/run_stub.py --mode fake-policy | always-clarify`
on the same code (`results/evidence/tuning2/floor_final/`). "Blocks" = audit kinds guard + gate; "records" =
refund + exchange records in the final states. All 12 live outputs: STRUCTURE CHECK PASSED. Load
average 1.2-2.6 during the F runs, 1.5-2.0 at the B starts; the GPU was 63-83 % busy with another
session's work before every server start (recorded in run_conditions), so latency is indicative only.

| suite | run | strict | turn acc | inv. viol. | median case ms | LLM calls/case | blocks | fallbacks | records |
|---|---|---|---|---|---|---|---|---|---|
| public | F x3 | 6/6, 6/6, 6/6 (18/18) | 1.000 | 0 | 5713 / 5711 / 5652 | 8.33 | 2 per run | 0 | 4 |
| public | B | 6/6 | 1.000 | 0 | 11164 | 9.33 | 0 | 0 | 4 |
| public | fake-policy | 6/6 | 1.000 | 0 | 4 | 8.17 | 0 | 0 | 4 |
| public | always-clarify | 1/6 | 0.154 | 0 | 3 | 4.17 | 13 | 0 | 0 |
| variants_v1 | F | 241/241 | 1.000 | 0 | 4948 | 7.94 | 82 | 2 | 132 |
| variants_v1 | B | 241/241 | 1.000 | 0 | 11064 | 9.20 | 108 | 3 | 132 |
| variants_v1 | fake-policy | 241/241 | 1.000 | 0 | 3 | 7.68 | 8 | 0 | 132 |
| variants_v1 | always-clarify | 54/241 | 0.187 | 0 | 2 | 3.79 | 463 | 0 | 0 |
| holdout_v2 | F | 56/57 | 0.985 | 0 | 5384 | 9.00 | 30 | 0 | 38 |
| holdout_v2 | B | 56/57 | 0.985 | 0 | 11685 | 10.40 | 34 | 0 | 38 |
| holdout_v2 | fake-policy | 50/57 | 0.896 | 0 | 5 | 8.18 | 2 | 0 | 32 |
| holdout_v2 | always-clarify | 11/57 | 0.178 | 0 | 3 | 4.18 | 110 | 0 | 0 |
| adversarial_v1 | F | 19/21 | 0.939 | 2 (oracle, adv09) | 5313 | 9.43 | 4 | 0 | 18 |
| adversarial_v1 | fake-policy | 19/21 | 0.939 | 2 (oracle, adv09) | 4 | 9.38 | 0 | 0 | 18 |
| adversarial_v1 | always-clarify | 2/21 | 0.061 | 0 | 2 | 4.57 | 49 | 0 | 0 |
| probes | F / fake / stub | 7/7 / 7/7 / 2/7 | 1.000 / 1.000 / 0.154 | 0 | 4661 | 7.00 | 3 | 0 | 5 |
| redteam live2 | F / fake / stub | 12/12 / 12/12 / 0/12 | 1.000 / 1.000 / 0.053 | 0 | 8550 | 13.50 | 3 | 0 | 12 |
| redteam live3 | F / fake / stub | 3/3 / 3/3 / 0/3 | 1.000 / 1.000 / 0.000 | 0 | 5669 | 9.00 | 0 | 0 | 3 |

Agent-attributable successes (F strict minus the always-clarify floor): public 5 per run (15 of 18),
variants 187, holdout 45, adversarial 17, probes 5, live2 12, live3 3. The do-nothing floor completes
no refund or exchange anywhere (0 completed turns, 0 records). F beats the regex test double on the
held-out set (56 vs 50); on the tuned suites they tie.

Baseline -> final for F on the same suites: variants 236 -> 241, holdout 50 -> 56, public 6/6 ->
18/18; variant fallbacks 18 -> 2, clarify vetoes 27 -> 5; LLM calls per case 7.84 -> 7.94.
B, measured only on the final code: protocol errors 25 on variants (native mode, a tool outside the
step's allowed list), reply-guard trips 40 on variants (deny replies with dates or day counts 16,
completed replies repeating record ids 10, the rest mixed; the guard swaps in the template, no
decision changes). F: 8 reply-guard trips on variants.

### Remaining failures, with root cause

- **hx12** (F and B, holdout): "quiero devolver la chaqueta ... Refund por favor" on a jacket + tee
  account. Both models hand off the jacket with evidence "(quiero) devolver la chaqueta"; code
  cannot verify a Spanish garment word, so the citation gate answers "names nothing that tells the
  lines apart" and the agent asks which item (safe clarify, no wrong action). Fixing it needs a
  multilingual garment lexicon in grounding; adding words because one held-out case uses them would
  be a phrase patch, so it is left open.
- **adv09** (oracle side, review B5): another customer's order shares the id Z-509. The desk
  correctly refunds this customer's own Z-509/A1; the oracle's `find_order` takes the first order
  with that id regardless of customer, grades deny/deny and reports the refund as 2 invariant
  violations (`ineligible_return`, `unexpected_mutation`). These are the only invariant violations
  in any run. Fix belongs in `evals/oracle.py` / `evals/invariants.py`.
- **adv10** (oracle side, review B4): after a completed refund, "Actually I'd rather exchange line
  A1 of Z-510 for size S." The desk denies REQUEST_ALREADY_COMPLETED (strong wording for the other
  remedy is a different request); the oracle still grades the old replay reading.
- **Update (integration stage):** both oracle bugs are fixed. `find_order` and the invariants look up
  the authenticated customer's own order, and the oracle replays only the same line AND the same
  remedy (CLARIFICATIONS #7 reworded). The adversarial labels were rebuilt
  (`tests/review/business/biz_build_cases.py`; only adv09 and adv10 changed). Re-scoring the stored
  runs above, without a model: F 21/21 strict, 0 invariant violations; fake-policy 21/21;
  always-clarify 2/21. The variant labels and the holdout_v2 decisions are unchanged.

### Deviations from SPEC / previous stage

- The size gate is stricter than "the agent's reading stands": a concrete size must come from the
  customer's words (§5). The agent still reads numerics, typos and other languages.
- Operations' option table carries a `next` hint naming the deciding call (§6.0 item 8).
- Order status must be exactly `"delivered"` (B6); `test_status_comparison_ignores_case_and_spaces`
  was replaced by two parametrized rows in `test_only_delivered_orders_qualify`.
- The first final run (`var/tuning2/final/`) and `final2/` are superseded; their numbers are not used.

### Open issues

- `CONFIRMATION_REQUIRED` protocol rejections (F 15 on variants): the agent still tries
  `execute_confirmed` on a typed "yes" before re-presenting. Costs one LLM step, never an action.
- B's free-text Operations summary carries record ids and dates that Resolution then repeats; the
  reply guard catches them. A shorter B summary prompt would help, but B is the baseline profile.
- The clarify facts line says "Matching item on your account" when it lists the only line for a
  product the customer does not own (hx56); wording only.

## Blind held-out v3 found a wrong-mutation bug (2026-10-28)

The first and only blind run of holdout_v3 (pre-fix bench, kept in
`results/evidence/bench_2026-10-28_prefix/`) scored F 56/60, B 56/60, 0 invariant violations. Two
of the 4 misses executed the WRONG business action: a confirmed turn whose text changed the
request ("Yes, refund the t-shirt." on a jeans proposal, v3_n22; "Yes, but make it an 11 instead",
v3_n28). The agent correctly proposed the change; code allowed only `execute_confirmed` and the
old proposal ran. The invariants did not catch it (nothing was duplicated or invented).

Fixes (general rules, `desk/orchestrator.py`, `desk/grounding.py`):
- On a confirmed turn, an agent handoff that proposes another line, remedy or size withdraws the
  grant (`guard/grant_withheld`, reason "the agent read a changed request") and the turn runs as
  unconfirmed (CLARIFICATIONS #9). Stored runs: the agent did this on 0 of ~300 other confirmed
  turns, so the rule fires only on real changes.
- The open exchange offer's own size does not compete with a new size written in the customer's
  latest message (`changes_the_open_offer`), so "make it an 11 instead, the 10 might still be
  tight" proposes the 11.

Live re-check on just those two cases: v3_n28 passes; v3_n22 executes nothing and proposes the
t-shirt one turn late (turn 2's typo "t-shrit" cannot be verified as a citation, so it asks).
From here on holdout_v3 is NOT blind; the post-fix bench reports it as a regression suite.

## Post-fix bench found two more code bugs (2026-10-29)

The first post-fix bench (kept in `results/evidence/bench_2026-10-29_fix1/`): F 57/60, B 56/60 on
v3, 0 invariant violations. Reading every miss:
- **B still refunded the wrong item on v3_n22.** Turn 3, "Yes, refund the t-shirt." with the flag on
  the jeans proposal: B's agent called `execute_confirmed`, and code did not read the product word
  as a change (it checked only IDs). The first fix covered only an agent that reads the change. Fix:
  `_grant_conflict` withholds the grant when the message names a product on the account other
  than the bound line's ("names a different item").
- **v3_n23 was a code block on a correct agent.** After the 34 ran out, "Ugh. What about a 30
  then?": the agent proposed the 30, the size gate read the earlier 34 as the request and refused,
  then the clarify veto pushed the agent back to the out-of-stock 34. Fix:
  `grounding.newer_size_in_latest_words`: a size written in the latest message, where code reads
  no size request, is not blocked by an older one.
- Not fixed: v3_n22 for F still proposes the t-shirt one turn late (the typo "t-shrit" can't be
  verified), and v3_n30 ("too tight" proposed instead of asking refund-or-exchange; the label is
  debatable).

Live check on the four v3 misses, both profiles: no wrong mutation, n23 passes, n28 passes on F
(lenient on B). No-model floors unchanged on all suites. Full bench re-run on this code.
