# Architecture and code review (20% criterion)

Reviewer role: a senior interviewer who will ask the candidate to explain and change the code live.
Review only. Nothing outside this file was edited. All experiments ran in a scratch copy
(`/private/tmp/claude-501/-Users-ali-Desktop-Projects/57fddab1-8df1-44b8-bb9b-749f533b4387/scratchpad/rehearsal/`).
Side effect to know about: running the documented default `.venv/bin/pytest` once rewrote 20 fake-model
screenshots in `var/screenshots/` (timestamps 08:28). The `live_*` real-model shots were not touched.

## What is good (say it in the interview)

- The shape is clear. The gateway (`desk/tools.py`) is the only path to state. Policy is pure. The
  decision comes from one table (`orchestrator.decide` with `domain.CODE_CLASS`). evals/ imports nothing
  from desk/, so the oracle really is independent.
- Functions are small. 769 functions, only 4 are longer than 80 lines (3 are in the variant generator).
  The longest product function is `server/app.py:558 create_app` at 97 lines.
- Adding a reason code is safe. `test_every_reason_code_has_a_decision_class` (tests/unit/test_tools.py:71)
  and `test_every_domain_code_has_wording` (tests/unit/test_render.py:92) fail if a new code has no class
  or no wording.
- Setup from a clean copy works. The untracked file set, `uv sync --frozen --offline` and then
  `tests/unit` gave 1347 passed and 1 skipped in 40.8 s. `python3 adapter.py` without a venv prints the
  setup hint and exits 2.

## Measurements

| module | lines | | module | lines |
|---|---:|---|---|---:|
| server/static/app.js | 1709 | | desk/orchestrator.py | 715 |
| desk/grounding.py | 1261 | | desk/llm.py | 668 (FakeLLM test double inside) |
| bench/report.py | 1244 | | desk/tools.py | 593 |
| evals/variants/generate.py | 977 | | bench/run.py | 588 |
| server/app.py | 741 | | desk/render.py | 523 |

Cyclomatic hot spots (AST count): bench/report.py:1028 section_failures cc 29,
desk/grounding.py:893 _trigger_scan cc 23, desk/render.py:454 guard_reply cc 21.
app.js has 12 sections in one IIFE. Its longest function is describeEntry (81 lines).
Default `pytest` runs 1422 tests in 112 s. `tests/unit` alone runs 1348 tests in 37 s, so the 74
Playwright tests take about 75 s.

## Findings, ranked by grading impact

### 1. HIGH: the LLM has no authority over any decision, and a stub model scores 100%
`desk/grounding.py:1202` accepts a propose handoff only when `analysis.unique_target == target`.
So the LLM can hand off only a target that the regex grounding has already found by itself. When
grounding finds that target, the clarify veto and override (`orchestrator.py:479`, `_reply`) plus the Ops
fallbacks do the work without the model. SPEC §5 says "equals a candidate". The code is stricter than that.
Evidence: `scratchpad/code_floor.py` uses a model that only ever replies "Could you tell me a bit more?"
and reports `needs_info`. It scores **241/241 strict on variants_v1 and 6/6 on public**, with
0 violations (`python -m evals.score ...`). The public run needed clarify_veto 6, clarify_override 6 and
fallback create_proposal 6 / execute_proposal 5. In the other direction, a model that picks the right line for "Please refund
the cheap one I paid 15 dollars for." is rejected twice with TARGET_NOT_GROUNDED and the turn ends as
clarify (`scratchpad/llm_cannot_ground.py`). Hidden variants change the wording, so grounding's coverage
of phrasing caps business correctness, and the brief says "meaningful agent use" is mandatory.
tuning_log.md:140 already notes the code floor, but nothing in results/ or the slides shows it.
Fix (small): (a) make the code floor a measured, reported row in bench/report.py, next to B and F;
(b) accept an LLM target that grounding cannot see only when it is in `candidate_keys`, no named
ID contradicts it, and the LLM gives an `evidence` quote that appears verbatim in the customer text.
The result is still only a proposal that needs explicit confirmation. Audit it as `grounded_by: llm`.
Test it on a paraphrase holdout where the floor is below 100%, and keep public_04 as clarify.

### 2. HIGH: any other llama.cpp build gives exit 3, and every turn is `error`
`desk/llm_server.py:308` refuses unless the version matches exactly. `allow_engine_mismatch` is not
reachable from `adapter.py` or `server.app`, yet the error message tells the user to "pass
allow_engine_mismatch". I faked a `llama-server` that prints 0.7.1 (`LLAMA_SERVER_BIN=...`). The
adapter exited 3, all 13 turns came back `error`, and each response read "not run: ... engine version ... != pinned".
`scripts/setup.sh` installs whatever Homebrew has today.
Fix: add `--allow-engine-mismatch` (and `DESK_ALLOW_ENGINE_MISMATCH=1`) to adapter_main and server.app.
In the adapter, default to warn and record `engine_version_verified:false`. Keep bench/ strict.
Test: the fake-binary unit test in tests/unit/test_llm_server.py plus one adapter test.

### 3. HIGH: the submission is not packaged; the evidence and raw logs would not ship
- There is no root README. Nothing covers setup/run/reset in one place, known limitations, or the
  AI-tool disclosure the brief requires.
- `git log` says "does not have any commits yet". So far there is no repository to hand over.
- `.gitignore:9 *.log` drops `results/**/server.log` and `adapter.log`. `git check-ignore -v` confirms
  it for results/smoke/public/F/run-1/server.log and results/offline/adapter.log. The brief says
  "Preserve raw logs".
- `.gitignore:6 var/` hides evidence that docs cite: var/tuning/final/ (the 241/241 F and B runs),
  var/tuning/holdout_v1.json (the only set not tuned on) and var/screenshots/ (named in SPEC §10.1).
- `.coverage` (88 KB) is not ignored and would be committed.
Fix: write README.md. Change `*.log` to `/logs/` and `var/**/*.log`. Move holdout_v1 and its labels to
evals/holdout/, the cited final runs to results/tuning/ and the presentation screenshots to
docs/screenshots/. Ignore `.coverage`. Make the first commit.

### 4. MEDIUM: the preflight saves the command line of every process on the machine into results/
`bench/preflight.sh:59-60` saves `ps -Ao pid=,%cpu=,rss=,command=` as `preflight/ps.txt` (801 lines).
These files are committed. They contain `...api_key=AIza...` strings (Google Drive crashpad args) and
paths of unrelated private projects. The full matrix writes about 30 of these files. The brief says
"Exclude secrets", and secret scanners may block a push.
Fix: save `comm=` (basename) for the llama and hog checks only, or redact everything after the
executable. Regenerate or scrub the ps.txt files already written.

### 5. MEDIUM: the same rules are written twice, and the two copies already disagree
`policy.resolve_size` (used by the gateway, `tools.py:311`, and by `_grant_conflict`, `orchestrator.py:275`)
and `grounding.resolve_size` (used by the gate, `check_requested_size`) disagree on 7 of 10 direct inputs:
`'lrg'`, `'bigger'`, `'size up'`, `'teel'` (policy says SIZE_NOT_OFFERED, grounding gives TEE-L),
`'M'` on TEE-M (policy gives TEE-M, grounding gives SAME_SIZE), UP from PANT-XL (policy gives PANT-32,
grounding gives SIZE_NOT_OFFERED), `''` (policy gives (None, None), grounding gives SIZE_UNSPECIFIED).
Today grounding canonicalises sizes before the gateway sees them, so this does not show up live yet. The
same split exists for the size ladder and size words (policy.py:41 and grounding.py:83), for 12 reason codes
redefined in grounding.py:45-56, for decision constants in render.py:25, for SIZE_CODES (operations.py:58 and
render.py:94) and for three ID normalisers (tools.normalize_ref, grounding._norm_id, render._id_tokens).
No test compares the two resolvers.
Fix: create desk/sizes.py with one ladder, one word table, one resolver. grounding.check_requested_size
then calls policy's resolver, and the codes are imported from domain. Add a Hypothesis test that the gate
and the gateway agree.

### 6. MEDIUM (live change: "return window = 14 days"): the rule is one place, the wording is four
`domain.WINDOW_DAYS` drives the logic, which is good. The wording still says 30 in render.py:27
`DEFAULT_WINDOW_DAYS = 30`, render.py:54 `"It is outside the 30-day return window."`,
app.js:512 `card.window_days || 30` and app.js:632 Policy goal "≤30 days". After I set 14 in the scratch
copy, `REASON_TEXT` still read "30-day". It is served to the UI through /api/meta and shown on the Policy node.
46 unit tests failed, mostly tests tied to the public fixtures. Nothing tells you which failures are
intended. evals/oracle.py:27 also has its own `WINDOW_DAYS = 30`, which is fine by design but must change too.
Fix: build the text from `WINDOW_DAYS`, put `window_days` in /api/meta for app.js, and pin the
public-fixture tests to an explicit 30-day policy so a window change fails in a known set of tests.

### 7. MEDIUM (live change: "different currency format"): the guard re-parses its own text
Money is formatted in three places: render.money (render.py:33), the guard's parser `_MONEY_RE` (render.py:384)
and app.js money() (app.js:64, used by chips and tables). With `money()` changed to "USD 81.00", the guard
rejected a faithful "$81.00" reply (`money:USD 81.00`). The proposal card then showed "USD" while the
effect chips and tables still showed "$". 24 tests failed on literal "$" strings. `state.currency` is ignored.
Fix: `facts_line` should return the allowed cents set (structured facts) and the guard should compare
against that set, not against re-parsed facts text. The server should send formatted amounts for every
UI amount (or put the format in /api/meta).

### 8. MEDIUM: a cosmetic prompt edit breaks the fake-model test suite
fake_policy_llm.py:109-113 finds the sections of the prompt with literal strings: `"ACCOUNT CARD ("`,
`"CONVERSATION SO FAR:"`, `"DESK: "`. Renaming one header in prompts.py, with no change in meaning, failed
19 test_server tests with an unclear message (`assert not [... guard/fallback ...]`).
"Tweak the prompt" is a likely live request.
Fix: export the header constants from prompts.py and import them in the fake.

### 9. MEDIUM-LOW: dead code and drift between SPEC and code
The agents no longer have check_eligibility (toolspecs.py:116-120), yet `BusinessTools.check_eligibility`,
SPEC §4/§6/§10.1 and app.js:870 still describe it. The comment "keeps check_eligibility for the UI" is
wrong: nothing in server/ calls it. Also unused outside tests: `policy.exchange_options`,
`tools.clear_target`, `ToolResult.is_replay`. An interviewer who reads SPEC will ask where Ops checks
eligibility. The answer is `create_proposal`.
Fix: delete the unused code or mark it, and update SPEC §4/§6/§10.1.

### 10. LOW-MEDIUM: default `pytest` is slow and writes over evidence
`addopts = "-q -m 'not llm'"` includes 74 browser tests, so a run takes 112 s instead of 37 s. Those tests
write `var/screenshots/*.png`, the folder SPEC names as evidence.
Fix: use `-m 'not llm and not e2e'` and document `pytest -m e2e`. Write screenshots to tmp unless
`E2E_SCREENSHOTS=1`.

### 11. LOW (live change: "per-turn tool budget"): the limits are spread over 4 modules
The limits are MAX_LLM_CALLS_PER_TURN and the token limits (agents/__init__.py:24-26),
MAX_RESOLUTION_STEPS (resolution.py:22), MAX_OPS_STEPS (operations.py:54) and MAX_HANDOFFS/MAX_REJECTIONS
(orchestrator.py:103-104). The "2 steps" fallback is a bare literal at orchestrator.py:361 and operations.py:249.
None of them is on `Profile`, so bench cannot ablate a budget. Changing MAX_OPS_STEPS from 4 to 3 still
passes all 1347 unit tests, so no test pins behaviour at the limit.
Fix: a `Limits` dataclass on Profile, threaded into Desk, OperationsAgent and StepRunner, plus one test
per limit.

### 12. LOW (live change: "clearance items cannot be exchanged"): easy if the flag is on the product
A product-level flag took 3 files and about 6 lines: a domain code plus CODE_CLASS, a REASON_TEXT entry
and one check in `policy.validate_replacement`. All 1348 unit tests passed. End to end, a named size gives
deny "Clearance items can be refunded but not exchanged". Two friction points. Without a size, the customer
gets the generic NO_EXCHANGE_AVAILABLE instead of the clearance reason. A line-level flag (like final_sale)
needs a `remedy` parameter on `policy.eligibility`, which is remedy-blind today.
Fix: `eligibility(state, customer, order, line, remedy=None)`, and keep the first specific reason when
`valid_options` is empty.

### 13. LOW (live changes: "colour exchange" / "third agent"): out of scope; push back, but know the files
Colour: "different size" is built into both resolvers, `options_for_sku` (which drops same-size siblings),
the `requested_size` tool argument, GroundedTarget, render wording and the oracle. That is about 10 files,
and finding 5 doubles part of it. While probing I found that "Exchange my red tee on order Q-77 for the
blue one, same size" proposes TEE-L-RED (size L): the one-valid-option rule ignores "same size". It is only
a proposal, but a confusing one.
Third agent: both Team-view grids are full (12 cells for 12 nodes, app.js:645-650), app.js has 22 role
literals, and domain.ACTORS, toolspecs, prompts, the fake and the orchestrator would all need edits. The
brief says "additional agents earn no automatic credit", so say that, and point to where it would go.

## Live-change cheat sheet

| request | edit | safety net | friction |
|---|---|---|---|
| window 14 days | domain.WINDOW_DAYS (+ render.py:27/54, app.js:512/632, evals/oracle.py:27) | 46 tests go red, mostly fixture-bound | wording stays "30-day" (finding 6) |
| clearance no exchange | domain code + CODE_CLASS, REASON_TEXT, policy.validate_replacement | table tests force class + wording | line-level flag needs remedy param (12) |
| colour exchange | both resolvers, options_for_sku, grounding lexicon, toolspecs, render, oracle | partial | out of scope; duplicated resolver (5, 13) |
| tool budget | constant in agents/ or orchestrator | none at the limit | 4 modules + 2 literals (11) |
| currency format | render.money + render._MONEY_RE + app.js money | 24 literal-string tests | guard re-parses text (7) |
| third agent | ACTORS, toolspecs, prompts, new loop, orchestrator, fake, app.js grids | fake + e2e | grids full (13) |
