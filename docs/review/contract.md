# Review: contract compliance, evidence and UI/UX

Reviewer role: contract, benchmark evidence, UI against the brief. Read-only review. No llama-server was
started. UI checks used the fake-policy server (`--port 8431 --fake`) and a real-mode UI with a dead
model URL (`--port 8432 --llama-url http://127.0.0.1:8499`) for the model-down banner.

Tests written: `tests/review/contract/` (25 tests: 19 pass, 6 xfail mark the defects below).
Run `.venv/bin/pytest tests/review/contract -q`, and add `--runxfail` to see the real failures.
Screenshots: `docs/review/shots/` (1440x900, 1280x720, 1920x1080, 390x844, dark mode, model down).

## What holds (verified)

- The validator passes on `var/verify/F.json`, the smoke B and F runs and `results/offline/results.json`,
  including `--baseline var/smoke_B.json`.
- Failed runs stay contract-valid with every case present. This covers a server that cannot start
  (exit 3, all turns `error`, state untouched), a server that dies mid-run (restarted 6 times, all
  turns `error`, only the external stock_change applied) and a refused LLM mid-case. No invented
  effect appeared in any of them.
- Proposal objects appear only on `propose` turns and carry the full binding. `final_state` keys equal
  the input keys in order. Records carry customer, order, line, request, proposal and replacement SKU.
- Audit entries carry agent, tool, arguments, result, error and committed_effects. `llm_step` entries
  hold only the chosen tool call and timings. No reasoning fields or think tags appear.
- The UI and the adapter build the same `CaseSession`, with the same LLMClient and profile F.
  Production code does not import `evals`. The fixture-leak test exists.
- UI: timeout recovery, stale-stock revalidation, replay and the model-down banner are all clear
  and actionable. In dark mode the smallest text contrast is 6.9:1. There is no horizontal overflow
  at 390px. Keyboard confirmation works and shows a 2px focus ring.
- REPORT numbers recompute from the raw files: F p50 4,207 and p95 4,848 ms, decode 75.2 tok/s.

## Findings (ranked by grading impact)

| # | sev | finding | evidence | fix |
|---|---|---|---|---|
| 1 | high | No root README. Setup and run steps, dependency and model versions, the reset command, known limitations and the AI-assistant disclosure are all required by the brief, and none are findable. | `find . -maxdepth 3 -iname "readme*"` finds only `candidate_assets/README.md`. A grep of docs for Claude, Copilot or "coding assistant" hits only `spec_critique.json`. | Write `README.md`: quickstart, versions, hardware table, reset, tests, bench, offline proof, limitations, AI tools used. |
| 2 | high | REPORT.md never discloses hardware or configuration, and never defines profiles B and F, so the "F 58.7% faster" claim cannot be attributed to any change. | 0 hits for M5, Q4_K_M, Q8_0, llama.cpp, macOS, concurrency, cache-ram, temperature and revision. `test_report_disclosure.py` (3 xfail). | Add `section_setup()` to `bench/report.py`. It should render the hardware block, a per-profile switch table with the argv diff, seed and temperature, concurrency, caches, timeouts, model revision and sha, engine version, run conditions, and the commands from `argv.txt` and `schedule.json`. |
| 3 | medium | `.gitignore` `*.log` drops every raw `server.log` and `adapter.log` under `results/`, yet REPORT "Files" lists them as raw per-run files. | `git check-ignore -v results/smoke/public/F/run-1/server.log` prints `.gitignore:9:*.log`. | Add `!results/**/*.log` after line 9. |
| 4 | medium | When the adapter's port is busy, every case ends in `error` and the adapter exits 3. There is no fallback. An orphaned llama-server on 8402 would give the grader 0/N. | `test_busy_default_port_does_not_fail_every_case` returns `(3, 'port 65001 is already in use (pid 8510)')`. See `llm_server.py:322`. | In `ServerHandle.start`, if the port is busy: attach when the server is healthy and serves the profile's model file, otherwise bind a free port. |
| 5 | medium | The engine version must match exactly, and nothing on the command line can relax that, while `setup.sh` installs an unpinned `brew install llama.cpp`. A different bottle makes every case `error`. | The check is at `llm_server.py:308`. Its message says "pass allow_engine_mismatch", but grep finds no CLI or env path to that argument. | Add `--allow-engine-mismatch` / `DESK_ALLOW_ENGINE_MISMATCH=1` to the adapter and server, and record it in the configuration. |
| 6 | medium | Team view text is too small and truncated. At 1440x900 the on-screen SVG text is 8.1 to 13.6 px (goal line 9.9 px, counters 8.7 px). At 1280x720 it is 7.1 to 12 px, and at 390px it is 4.5 to 7.6 px. 9 of 12 goal lines end in "...". At 1280x720, theater mode makes the graph 824 px tall inside a 532 px panel, so the bottom two rows are hidden. | Shots `17_*`, `18_team_theater_1280x720.png`, `14_phone_proposal_full.png`. The truncation comes from `app.js:722` `wrapText(n.goal, 31, 2)`. | Wrap by measured width. Keep on-screen text at 11 px or more. In theater mode, size the SVG by panel height or use a 4x3 layout. On narrow screens, use a card list. |
| 7 | low | A crash after a persisted commit makes the turn read "not run: RuntimeError..." even though the refund is in `final_state`. | `test_crash_after_commit_response_does_not_claim_not_run` (xfail) | In `session.run_case`'s except block, check `tools.completed_record()` and name the committed transaction in the error turn. |
| 8 | low | `hardware.power` is a constant string ("AC power ... sleep prevented") even on battery, and even in plain adapter runs where no caffeinate is running. | `telemetry.py:334`; `test_hardware_power_reflects_the_machine` (xfail) | Derive the value from `_power_conditions()` once per process, or rename it to a declared bench condition. |
| 9 | low | The TIMEOUT tool error tells the LLM what to do ("reconcile with get_request_status"). That is borderline against "not as instructions telling the LLM". | `tools.py:535` | Use a neutral message. The orchestrator already forces the status check. |
| 10 | low | The composer offers to confirm P-0001 after it was invalidated (public_05) or executed. A deny lists alternatives but gives no action for them. A "Not now" disable is lost on re-render. | `19_after_deny_confirm_offer.png`; `app.js:203` `lastPresentedProposalId` ignores the proposal's status. | Label dead proposals. Add a "propose this size" button that fills the composer. |
| 11 | low | `model_sha256_verified: true` comes from a cached digest keyed on path, size and mtime (`models/.sha256-cache.json`), not from a fresh hash. | `llm_server.py:76-84` | Add `model_sha256_source: cache\|computed`. |

Context, not a finding: `results/smoke` has 1 run per profile. The variant gate reads INCOMPLETE until
the full matrix (`scripts/bench.sh`) runs. Note that this review read the Q4_K_M model file once (a
page-cache probe check), so the next run's `model_page_cache` may read "warm".
