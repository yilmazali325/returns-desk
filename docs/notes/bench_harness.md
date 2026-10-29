# Benchmark harness, statistics and offline proof

Owner files: `bench/__init__.py`, `bench/run.py`, `bench/report.py`, `bench/preflight.sh`,
`scripts/bench.sh`, `scripts/offline_check.sh`, `scripts/offline.sb`, `scripts/reset.sh`,
`tests/unit/test_report.py`, `tests/unit/test_bench_run.py`.
Measured 2026-10-28 on the bench machine (Apple M5 Pro, 24 GB, macOS 26.4.1 25E253, llama.cpp 0.7.0
build 11669), AC power, powermode 0, no thermal warning, load average about 1, no other llama-server.

## Commands

```
# full matrix (about 80 minutes; needs a quiet machine: stop the UI server and its llama-server first)
scripts/bench.sh --port 8405                 # --resume continues an interrupted matrix
#   = bench/preflight.sh --out results/preflight/public-start --port 8405 -- \
#       .venv/bin/python -m bench.run --suite public --all-profiles --runs 3 --port 8405 --preflight-wait-min 30
#     bench/preflight.sh --out results/preflight/variants-start --port 8405 -- \
#       .venv/bin/python -m bench.run --suite variants --profiles B F --runs 1 --port 8405 --preflight-wait-min 30
#     the same for --suite holdout_v2 and holdout_v3 (B F x 1) when their files are on disk
#     .venv/bin/python -m bench.report --results results          # -> results/REPORT.md + results/csv/
# every bench.run first writes the code-only floors (<suite>/floors/<mode>/, no model, seconds) and
# hashes the model files in full once (<suite>/model_sha256*.json); every adapter run gets --strict-engine

# what was run for this stage
.venv/bin/python -m bench.run --suite public --profiles B F --runs 1 --results results/smoke   # 1 min 35 s
.venv/bin/python -m bench.report --results results/smoke        # -> results/smoke/REPORT.md
scripts/offline_check.sh --port 8406                             # -> results/offline/OFFLINE_PROOF.md, ~30 s

# code-only floors for the smoke report (no model; re-run 2026-10-28 12:40 EDT on the rewritten desk,
# code_sha256 e5265954dee8...; the smoke model runs are older and carry no fingerprint, so the report
# prints "n/a: other code" for agent-attributable instead of mixing two versions of the desk)
for s in public variants holdout_v2 holdout_v3; do .venv/bin/python -m bench.run --suite $s --floors-only --results results/smoke; done

# other entry points
.venv/bin/python -m bench.run --all-profiles --runs 3 --dry-run  # print the pre-registered schedule
bench/preflight.sh --out DIR --port 8405                          # check only; exit 1 = refused
scripts/reset.sh [--dry-run] [--force]                           # remove var/ui (UI sessions) and var/adapter

# tests: 73 tests, 100 % line and branch coverage of bench/
.venv/bin/pytest tests/unit/test_report.py tests/unit/test_bench_run.py --cov=bench --cov-branch --cov-report=term-missing
```

## How a run works (bench/run.py)

1. **Schedule.** Block k runs every profile once, in an order shuffled by `random.Random(42)`, so run k
   of every profile shares a block (the report pairs on it). `schedule.json` is written before the
   first run. For the full matrix, block 1 is F-native, F-cacheram-default, B-q4, F-noprefetch,
   B-repeat, F-q8, F-freetext, B, F.
2. **Preflight before every run** (`bench/preflight.sh --out <run>/preflight --port N`). It refuses
   when any of these is true:
   - a llama-server is running;
   - a process is at 50 % CPU or more, or the 1-minute load is above 5;
   - median GPU utilization is above 10 %, or IOAccelerator in-use memory is above 2048 MiB;
   - the machine is on battery, powermode is not 0, or a thermal or CPU speed limit is set;
   - the port is in use.

   It prints the offending processes and saves the raw pmset, therm, ps, ioreg and lsof readings.
   bench.run waits up to `--preflight-wait-min` for the machine to become clean, then aborts (exit 3).
   `--allow-dirty` runs anyway, records the override, and the report flags those runs as "dirty".
3. **The adapter.** `python3 adapter.py --profile P --port N --input <cases> --output
   results/<suite>/<P>/run-<k>/results.json` runs as a subprocess in its own process group. A
   timeout kills the group, adapter and server together. The adapter starts and stops its own
   llama-server, so cold start, sha256 and engine checks and the GPU baseline are measured the same
   way the grader's command measures them.
4. **/metrics scrapes.** The runner scrapes /metrics twice and never polls during cases:
   - `metrics_before.prom`: when /health first answers;
   - `metrics_after.prom`: when the adapter logs its last case. The adapter then still spends a few
     hundred ms collecting run conditions, so the server is still up.

   Each scrape has a `# moment:` header. A missed scrape is recorded as "not captured: <why>".
5. **Afterwards.** The runner copies the server log, then runs `validate_submission.py` on the run
   and again with `--baseline` against B run-1 (re-checked at the end of the matrix for runs that
   finished before B run-1). It scores the run with `python -m evals.score`, then writes
   `run_conditions.json`, `argv.txt` and the `run.json` manifest.
6. **Nothing is overwritten.** An incomplete run directory becomes `run-<k>.attempt-<n>`, every
   attempt is appended to `attempts.jsonl`, a complete run is skipped with `--resume`, and without
   `--resume` the runner refuses to start.

## Statistics (bench/report.py, from raw files only)

- The report re-scores every `results.json` with the oracle and warns when `score.json` disagrees.
- **Success** is strict case success with a Wilson 95% interval, pooled over runs. The report says
  that pooled case-runs are not independent.
- **Percentiles** are nearest-rank, `sorted[ceil(p/100*n)-1]`. With n < 20 the report marks p95 as
  the maximum.
- **Latency** covers:
  - successful cases;
  - all attempts;
  - turns;
  - per scenario: case id for public, label family for variants;
  - the intersection of cases solved in every run by every profile;
  - the first case after a start vs warm cases;
  - every LLM step, overall, per agent and per tool.
- **Paired comparison.** The report computes F/B and every profile vs B, plus F vs each
  leave-one-out profile and F vs B-q4. The ratio is the geometric mean over (case, run k) pairs
  that succeeded in both profiles. The 95% CI comes from a percentile bootstrap with 10,000
  resamples and seed 42. **It resamples whole cases**, because the 3 runs of a case are
  near-copies under greedy decoding. Resampling 18 pairs as if independent gives a much narrower
  interval; this is covered by a unit test.
- **Noise floor.** B-repeat/B is the A/A noise floor. An effect is named only when its CI excludes
  1 and its estimate lies outside the A/A interval.
- **Throughput** is token-weighted (sum of tokens / sum of time). It is cross-checked against the
  server's own /metrics counters. A run with a server restart is left out of that check, because
  the counters reset.
- **Leave-one-out** profiles are detected from the five switches in each `results.json`
  configuration (quantization, protocol, prefetch, structured report, cache-ram). A profile that
  differs from F in exactly one switch, set to B's value, is a leave-one-out profile.
- **Gate.** The SPEC gate is public ≥ B, variants ≥ B and zero invariant violations. Per change it
  also requires F ≥ F-without-the-change on public. Missing data gives INCOMPLETE; fewer than 3
  public runs gives PROVISIONAL.
- **Taxonomy.** These categories are always listed, with zero counts:
  - oracle mismatch (decision, or binding/state/events);
  - invariant violation;
  - real LLM timeout;
  - other LLM error;
  - step/call cap (inferred);
  - case time budget;
  - crash.

  Below them come protocol errors, fallbacks and guard overrides by name. Injected faults are
  listed separately: stock_change applied or not; timeout_after_commit armed, fired, and
  reconciled to completed.
- **Decision-trace hash.** It covers per-turn decisions, proposal bindings, transaction ids and the
  ordered audit of tool choices, with non-text arguments and error codes. It excludes timings and
  wording. A separate text hash covers responses and drafted replies.
- **Unavailable** metrics are listed with the reason, for example:
  - per-process GPU memory;
  - energy;
  - discrete VRAM;
  - the A/A floor, when B-repeat did not run;
  - a missed /metrics scrape;
  - null `peak_vram_mb`;
  - a true SSD cold start, when every start found the model in the page cache.

## Measured in this stage

**Bench smoke, public cases, B and F, 1 run each, preflight passed** (`results/smoke/REPORT.md`):

| | B | F |
|---|---|---|
| strict | 6/6 | 6/6 |
| case p50 / p95 (= max, n=6) | 10,173 / 12,638 ms | 4,207 / 4,848 ms |
| turn p50 / p95 (= max, n=13) | 4,074 / 6,937 ms | 1,827 / 3,200 ms |
| LLM step p50 / p95 | 947 / 1,792 ms (57 steps) | 426 / 826 ms (49 steps) |
| LLM calls per case | 9.50 | 8.17 |
| decode tok/s, client = server /metrics | 52.0 = 52.0 | 75.2 = 75.2 |
| prefill tok/s, uncached (client / server) | 1,013 / 1,054 | 1,047 / 1,056 |
| cache hit (cached / all prompt tokens) | 88.2 % | 85.9 % |
| peak RAM max (phys_footprint) | 1,940 MiB | 2,070 MiB |
| peak unified GPU max | 5,131 MiB | 3,330 MiB |
| spawn → ready / first request (page cache cold, 0 % resident) | 2,268 / 59 ms | 1,751 / 59 ms |
| protocol errors | 1 (TOOL_NOT_ALLOWED) | 0 |

- **F vs B.** The paired geometric-mean ratio F/B is **0.413, 95% CI [0.375, 0.440]**. That is 6
  pairs from 1 run, with no A/A floor measured, so it is a pipeline proof, not the headline number.
- **Bench overhead.** It is about 2 s per run: 94.9 s total against 90.4 s of adapter wall time.
  Most of it is the preflight's 2 s of GPU sampling.
- **Load during runs.** The 1-minute load stayed at about 1.1 before and after each run, so the
  load limit of 5 does not refuse the bench's own back-to-back runs.

**Offline proof** (`results/offline/OFFLINE_PROOF.md`): PASSED.

- The machine was online. Outside the sandbox, a connect to 1.1.1.1:443 succeeded and
  huggingface.co returned HTTP 200.
- Inside the sandbox, all four negative controls failed:
  - nc to 1.1.1.1:443 exited 1;
  - the project's Python got `PermissionError: [Errno 1] Operation not permitted`;
  - curl by name exited 6 (DNS blocked);
  - curl with DNS pre-resolved outside (`--resolve`) exited 7 (TCP blocked).
- The full public run inside the sandbox took 28 s; the adapter and its llama-server were both
  sandboxed.
- Six lsof snapshots taken from outside showed only 127.0.0.1 sockets.
- The validator passed and the oracle scored 6/6 strict, also inside the sandbox.

**Determinism.** The smoke F run and the sandboxed F run are two fresh adapters and servers, 6
minutes apart. They have identical decision traces and identical text traces in 6/6 cases.

## Estimated wall time of the full matrix

**About 80 minutes** (range 70–95), plus about 2 s of bench overhead per run.

| part | estimate | basis |
|---|---|---|
| public matrix, 9 profiles x 3 runs | ~20 min | 3 blocks of ~6.8 min |
| variants, F x 1 | ~17 min | tuning stage: 1,018 s of case time |
| variants, B x 1 | ~39 min | tuning stage: 2,347 s, under load 10–12, so an upper bound |

Per run on the public set:
- measured: B 63 s, F 27 s;
- estimated from Q4/Q8 decode rates and extra calls: B-repeat ~63 s, B-q4 ~48 s, F-q8 ~34 s,
  F-native ~35 s, F-freetext ~35 s, F-noprefetch ~32 s, F-cacheram-default ~27 s.

## Review fixes (2026-10-28)

- **Setup disclosure (contract C2).** REPORT.md opens with "Setup": the hardware block (with a
  warning when runs differ), every profile of `desk/profiles.py` defined in one table (B, F, each
  leave-one-out F-*, B-q4, B-repeat), per-profile configuration (model file, quantization, sha256 and
  how it was obtained, engine version and whether it matched the pin, protocol, seed, temperature,
  max tokens, context, concurrency, caches, timeouts), the engine flags, the measured power state of
  every run (before/after) and the exact commands (each profile's argv.txt plus the bench commands).
- **Code-only floors (redteam RT-1, code C1).** Each suite's quality table adds two rows run through
  `evals/holdout/run_stub.py` with no model (`always-clarify`, `fake-policy`) and an
  "agent-attributable" column: profile strict rate minus the always-clarify floor. With the reviewed
  desk the always-clarify floor equalled F everywhere (6/6, 241/241, 45/57, 56/60: +0.0 pp). With the
  rewritten desk it is 1/6 public, 54/241 variants, 11/57 holdout_v2, 9/60 holdout_v3 (fake-policy
  double: 6/6, 241/241, 50/57, 56/60), so code alone no longer reaches what a working agent reaches.
  The model runs have to be redone on that desk before the column means anything.
- **Same code or no comparison.** `bench.code_fingerprint()` (sha256 over desk/**/*.py and
  adapter.py) is written as `code_sha256` into every run.json and floor.json; the report computes
  agent-attributable only when a profile's runs and the floor carry the same value, and otherwise
  prints `n/a: other code` with both fingerprints under the table.
- **Held-out suites.** `bench.SUITES` has `holdout_v2` (evals/holdout/) and `holdout_v3`
  (evals/holdout_v3/); `Suite.available()` lets the CLI and `scripts/bench.sh` skip one that is not on
  disk yet.
- **Strict engine, fresh hashes (code C2, contract C11).** Adapter runs get `--strict-engine` and
  never inherit `DESK_ALLOW_ENGINE_MISMATCH`; `verify_models_fresh` hashes every byte once per
  invocation and refuses (exit 3) on a mismatch, so the per-run `model_sha256_source = cache(...)`
  reads a digest computed minutes earlier.
- **Preflight privacy (code C4).** `ps.txt` holds `pid %cpu %mem name` (executable basename via
  `ps -o comm`), never command lines; the two committed smoke `ps.txt` files were scrubbed the same
  way (pid, %cpu, rss, name). `telemetry.run_conditions()` follows the same rule, and the process
  rows inside the copied tuning evidence (`results/evidence/tuning/**/*.json`, run_conditions) were
  reduced from `command` to `name` too; nothing else in those files changed.
- **Raw logs tracked (contract C3/NEW-2).** `.gitignore` has no global `*.log` rule any more
  (`results/**/*.log` and `evals/**/*.log` are evidence); `var/*` stays scratch, and the evidence the
  notes cite was copied to `results/evidence/tuning/` and `results/evidence/llm_infra/`. The one
  exception inside var/ is `var/screenshots/` (the presentation shots SPEC §10.1 cites), which is
  tracked; E2E shots in `var/e2e/` are not.
- **No orphaned server after a timeout (contract NEW-1).** llama-server now runs in its own session,
  so killing the adapter's process group no longer reaches it; after a timed-out run the bench kills
  whatever still listens on the run's port (`stop_orphan_server`, recorded as
  `adapter.orphan_server_killed`). Preflight proved that port free before the run.
- **Fast default tests (code C10).** `pytest` = unit + review (`-m 'not llm and not e2e'`);
  `scripts/test.sh` runs everything but the real model; E2E screenshots go to `var/e2e/shots/`, and
  only `scripts/screenshots.sh` writes the presentation shots in `var/screenshots/`.

## Deviations from SPEC (and why)

1. **Results layout.** It is `results/<suite>/<profile>/run-<k>/`, where SPEC has
   `results/<profile>/run-<k>/`. The task asks for public and variant suites side by side.
2. **Extra files.** Each run also gets `adapter.log`, `score.txt`, `score.json`, `preflight.txt`,
   `preflight/` and `run.json`. Each suite gets `schedule.json` and `attempts.jsonl`. They make every
   number traceable, and they let the bench resume without overwriting anything.
3. **`--baseline` scope.** The check also runs for B run-2..k against B run-1, a superset of "every
   non-B run".
4. **/metrics scraped twice per run, not per case.** A scrape mid-run competes with decoding.
   Per-case token and time figures come from the per-step audit telemetry, which matches the
   server counters (52.0 = 52.0 and 75.2 = 75.2 tok/s above).
5. **Case-level (cluster) bootstrap** instead of resampling independent pairs. The task says
   "bootstrap 95% CI (10k, seed 42)" without a unit. Cases are the independent units here.
6. **Stricter per-change gate.** It adds "F ≥ F-without-the-change (public)" to SPEC's "≥ B", so a
   change cannot ride on the others. Both columns are printed, so the SPEC-only verdict stays
   readable.
7. **Budget exhaustion is inferred.** The orchestrator does not audit it, so "step/call cap
   reached" is inferred from the audit: a turn with 12 LLM calls or 5 Resolution steps.
8. **`scripts/offline.sb` rewritten.** It now denies all outbound IP except loopback and closes the
   mDNSResponder socket. Before, it blanket-denied network-outbound, which also blocks local unix
   sockets. This is the recipe the review verified. The rewrite was checked: loopback HTTP works
   inside it, and nc, the project's Python and curl are blocked.
9. **No `--offline` flag for llama-server.** Profiles pin the flags and `desk/llm_server.py` strips
   `LLAMA_ARG_*`. The server only gets a local `-m` path, and the sandbox blocks it anyway.
10. **The sample REPORT.md is at `results/smoke/REPORT.md`.** `results/REPORT.md` is kept for the
    real matrix, so that `--resume` can never pick up smoke runs.

## Open issues for the next stage

- **The full matrix has not been run.** At the end of this stage the UI server
  (`python -m server.app --port 8410`) and another agent's llama-server (port 8404, Q4_K_M) were
  running. Preflight names that server and refuses, which is the intended behaviour. Stop both
  before `scripts/bench.sh`.
- **The variant suite and the holdout are not clean for grounding.** Both were seen in tuning. The
  report's variant numbers measure regression, not generalization; a fresh held-out set written by
  someone who has not read `desk/grounding.py` would be the honest number.
- **`desk/orchestrator.py` could audit budget exhaustion explicitly** (for example `kind="fallback",
  tool="budget_exhausted"`). Then the report would not need to infer it.
- **`desk/adapter_main.py` could take `--log-dir`.** Today the bench copies
  `var/adapter/server-<P>-<port>*.log` by name and mtime.
- **The bench depends on the adapter's log format.** It detects the end of the cases from the
  adapter's `[adapter] <case_id>: [...]` lines. If that format changes, metrics_after becomes "not
  captured" (recorded, not silent).
- **Small-n caveat.** The public suite has 6 cases. The interval on 18 pooled case-runs is
  optimistic, and its p95 is the maximum; the report states both.
- **`results/` is committed as evidence** (no ignore rule): about 150 KB per public run, about 6 MB per
  variant run, 22 MB for results/evidence/tuning/.
- **A moved port loses the /metrics scrapes.** The adapter now moves llama-server to the next free
  port when `--port` is busy (recorded in `startup.port_note`); the bench watches only its own port,
  so such a run would record "not captured". Preflight refuses a busy port first, so this needs a race.
- **Smoke model runs predate the core rewrite.** The floors were re-run on the rewritten desk; the
  B/F smoke runs were not (they need the model and a clean machine). Re-run them, e.g.
  `python -m bench.run --suite public --profiles B F --runs 1 --results results/smoke-v2`, before quoting
  an agent-attributable number.
