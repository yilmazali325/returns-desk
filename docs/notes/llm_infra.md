# LLM infrastructure and telemetry: design notes and measurements

Owner modules: `desk/llm.py`, `desk/llm_server.py`, `desk/profiles.py`, `desk/telemetry.py`.
Tests: `tests/unit/test_{llm,llm_server,telemetry,profiles}.py` (no server needed),
`tests/llm/test_llm_live.py` (`-m llm`, real llama-server on port 8401).
Measured 2026-10-27 on the bench machine (Apple M5 Pro, 15 CPU cores, 16-core GPU, 24 GB, macOS 26.4.1
25E253), llama.cpp `llama-server` 0.7.0 (build 11669, commit 620d2225c), Qwen3.5-4B Q4_K_M / Q8_0 with
the pinned sha256 values. Run conditions recorded by the live suite before and after the run: AC power,
`pmset powermode 0`, sleep prevented, no thermal warning, load average 1.5–2.4, **no other llama-server
and no process above 50 % CPU**.

## How to reproduce every number

```
# default suite (no server): 164 tests for these modules, 100 % line coverage
.venv/bin/pytest tests/unit/test_llm.py tests/unit/test_llm_server.py tests/unit/test_telemetry.py \
  tests/unit/test_profiles.py --cov=desk.llm --cov=desk.llm_server --cov=desk.profiles \
  --cov=desk.telemetry --cov-report=term-missing

# live suite (19 tests, ~6.5 min; starts/stops its own servers on :8401 only). The numbers below come
# from one full run (18 tests, 387.5 s) plus `-k engine_facts` run right after it, same conditions.
.venv/bin/python -m pytest -m llm tests/llm/test_llm_live.py -s -p no:cacheprovider
#   → prints every number as "[live] <section> <key>: {...}" and merges them into
#     results/evidence/llm_infra/live_results.json (a tracked copy of var/llm_infra/, which is git-ignored) with the run conditions of each run under "runs".
#   Server logs: results/evidence/llm_infra/server-<quant>-cram<MiB>.log
```

Checkpoint and prompt-cache entry sizes come from the engine's debug log (not printed at the
default log level), measured once with the F flags plus `-lv 4`:

```
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --port 8401 -ngl 99 -fa on -c 8192 -np 1 --jinja \
  --reasoning off --cache-ram 1024 --ctx-checkpoints 32 -t 5 --host 127.0.0.1 --metrics --no-webui \
  --seed 42 -lv 4 > var/llm_infra/debug_cache_q4.log 2>&1 &   # copy kept in results/evidence/llm_infra/
.venv/bin/python -c "import sys; sys.path.insert(0,'tests/llm'); from test_llm_live import agent_workload; \
  from desk.llm import LLMClient; agent_workload(LLMClient('http://127.0.0.1:8401','constrained'), cases=3)"
grep -E "cache state|created context checkpoint|total state size" results/evidence/llm_infra/debug_cache_q4.log
pkill -f "llama-server.*--port 8401"
```

## Review fixes (2026-10-28): engine pin, busy port, signals, hash provenance, power, privacy

What changed after the adversarial review (docs/review/code.md C2/C4, contract.md C4/C5/C8/C11/NEW-1),
and how to check it:

| Area | Behaviour now | Where / test |
|---|---|---|
| Engine pin | `LlamaServer.verify()` always compares `llama-server --version` with `ENGINE_VERSION` and returns `engine_version_actual`, `engine_version_expected`, `engine_version_match` (in `startup`). A different build is refused unless the caller allows it; the error names `--allow-engine-mismatch`, `DESK_ALLOW_ENGINE_MISMATCH=1` and `--llama-url`. | `test_verify_engine_version_mismatch` |
| Adapter policy | default **warn** (runs another build, logs a WARNING, `configuration.engine_check = "warn"`, `engine_version_verified = false`); `--strict-engine` refuses (exit 3, every turn `error`); `--allow-engine-mismatch` / `DESK_ALLOW_ENGINE_MISMATCH=1` accepts quietly. The bench always passes `--strict-engine` and strips the env var. | `test_adapter.py::test_another_engine_build_runs_with_a_warning_by_default` and the two tests after it |
| UI policy | `python -m server.app --start-llama` stays strict; `--allow-engine-mismatch` (also through `scripts/run.sh`) or the env var allows another build with a WARNING line. | `test_ensure_llama_passes_the_mismatch_override_and_warns` |
| Which build | `scripts/check_engine.sh` (run by `setup.sh`) prints found vs pinned, records both in `var/setup/engine.txt`, and gives the exact build-from-source steps for commit `620d2225c`. | `test_run_script.py::test_check_engine_*` |
| Busy port | `ServerHandle` moves llama-server to the next free port (then an OS-chosen one) and records `startup.port_requested`, `startup.port`, `startup.port_note`; it never fails every case for an occupied port. | `test_a_busy_port_moves_the_server_to_the_next_free_port`, `test_next_free_port_*` |
| Signals | llama-server runs in its own session (`start_new_session=True`); `stop()` sends SIGTERM then SIGKILL to its process group. The adapter turns SIGTERM, SIGHUP and SIGINT into `SystemExit(128+n)`, so its `finally` stops the server; a real subprocess test checks the port is free after each signal. Not covered: SIGKILL of the adapter itself (nothing can run then). | `test_a_signalled_adapter_stops_the_llama_server_it_started[*]` |
| Hash provenance | `file_sha256()` returns `Sha256Reading(value, source)`, source `fresh` or `cache(path,size,mtime)`; `configuration.model_sha256_source` records it. The bench hashes every model file in full once per invocation (`results/<suite>/model_sha256*.json`, `bench.run.verify_models_fresh`). | `test_file_sha256_caches_by_path_size_and_mtime`, `test_verify_models_fresh_*` |
| Power | `hardware.power` is now a static pointer ("measured per run ... see configuration.run_conditions"), so it can never contradict the reading; `run_conditions()` adds `power_summary` (e.g. "AC Power, powermode 0, no thermal warning, sleep prevented"). | `test_hardware_power_is_a_pointer_and_the_reading_lives_in_run_conditions` |
| Privacy | `run_conditions()` lists other processes as `{pid, name, cpu_pct, mem_pct}` (`ps -o comm`, basename only), never full command lines. | `test_run_conditions_on_ac_with_competitors` |

## Findings that change how the next stages should build things

| # | Finding (measured) | Consequence | Confidence |
|---|---|---|---|
| 1 | `tools` + `response_format` in one request → **HTTP 400** "failed to parse grammar" | constrained mode never sends `tools`; docs go into the system prompt | high (reproduced by `test_engine_facts_behind_the_protocol_design`) |
| 2 | Native `tools` with a top-level `oneOf` schema → the model emits `handoff_to_operations` with **`{}`** (no parameters); the same schema flattened → `{"intent":"propose_refund","order_id":"ORD-1Q","line_id":"LN-1"}` | `ToolSpec.openai_tool()` sends tagged unions flattened (`flatten_one_of`); code still validates the strict union. Without this, B could never hand off. | high |
| 3 | Native mode cannot restrict tools per step: with only `report_to_resolution` allowed, Q4 called `execute_proposal` | native relies on code (`TOOL_NOT_ALLOWED` protocol error); constrained enforces `allowed` in the decoder | high (1 of 1 in the sequence; consistent with design) |
| 4 | Constrained mode: **60/60** steps (30 per quant) parsed, tool always in `allowed`, args always schema-valid (enums, oneOf, required, no extras) | constrained steps never produce protocol errors in practice; validation stays as a guard | high for these prompts |
| 5 | Off-account request ("LN-7 of ORD-999") in constrained mode with `handoff` + `reply` allowed → both quants chose `reply_to_customer` asking to confirm; no ID was forced onto the account | free-string IDs + reply escape branch work as SPEC §7 intends | medium (1 prompt per quant) |
| 6 | Within a turn, append-only steps re-prefill only the new call + tool result (29–40 tokens, 84–94 ms); a role switch R→O→R restores R from the host cache (34 tokens re-prefilled) | one slot serves both agents; per-step prefill is not the bottleneck | high |
| 7 | Qwen3.5 is a hybrid recurrent model: the server cannot truncate to the point of divergence. It keeps **context checkpoints**, one at the start of the **last user message**, and falls back to the nearest one. A replayed call whose bytes differ from what was generated costs 338 (constrained) / 497 (native) re-prefilled tokens instead of ~35 | history must be byte-identical: replay with `assistant_message(step)` / `tool_message()`, deterministic tool results (no timestamps) | high |
| 8 | Changing the tool list → `cached = 0` in both protocols (637 / 687 ms re-prefill on Q4); changing the system prompt → 0 (constrained) / 383 of 1081 (native, the tools block before it survives) | tool lists per role constant; no dates/IDs in system prompts; restrictions only via `allowed` | high |
| 9 | Putting the account card in its **own user message** before the memo/current-message user message: new-turn re-prefill **159 → 69 tokens** (156 → 95 ms) | recommended prompt layout for the orchestrator (not mine to change) | high for the mechanism, 1 measurement |
| 10 | Decode is 78 % of wall time in the cached agent workload; Q4 decodes at **70.5 tok/s**, Q8 at **50.8 tok/s**; prefill is faster on Q8 (1693 vs 1523 tok/s) | output tokens are the latency lever (short args, structured report); Q4 wins on agent latency | high |
| 11 | `--cache-ram` bounded (1024/2048) vs default 8192: identical cached tokens, identical decision trace, wall time 38.2 / 38.4 / 39.1 s over 80 calls (spread 2.4 %); the cache never held more than **one** entry | the bound is a worst-case cap, not a measured saving. The SPEC critique's projection that the default "keeps growing across cases" did not happen in this pattern | high for this workload |
| 12 | Host-memory lever is the slot's context checkpoints (50.25 MiB each, up to 32 per slot), not `--cache-ram` | proposal only: a `--ctx-checkpoints` ablation (not measured, not in F) | medium |

## Interfaces for the next stages

- `ToolSpec(name, description, parameters, terminal=False)`. `parameters` must be `type: object`;
  construction fails on schema keywords the validator does not enforce (e.g. `pattern`, `maxLength`)
  and on objects without `additionalProperties: false`. Define tool specs as module constants.
- `LLMClient(base_url, protocol, *, seed=42, timeout_s=20, retries=1, retry_backoff_s=0.25, transport=None)`;
  `step(role, messages, tools, allowed, max_tokens) -> Step`. Pass the **full** role tool list every step
  and restrict with `allowed`. Roles are free strings (used for telemetry and FakeLLM scripts).
- Messages: `{role: system|user|assistant|tool, content, tool_name?, tool_args?}`; append
  `assistant_message(step)` then `tool_message(step.tool, result)` after every step (also after a
  protocol error, with the error as the result).
- `Step`: `tool, args, protocol_error, dropped_calls, role, prompt_tokens (cached + new), cached_tokens,
  completion_tokens, prompt_ms, predicted_ms, latency_ms, finish_reason, text, http_attempts`,
  `.ok`, `.prefill_tokens`, `.telemetry()` (numbers only, no model text).
  `prompt_tokens = timings.prompt_n + timings.cache_n` (= `usage.prompt_tokens`, checked: 26 = 4 + 22).
- Protocol errors: `NO_TOOL_CALL`, `INVALID_JSON`, `TRUNCATED`, `UNKNOWN_TOOL`, `TOOL_NOT_ALLOWED`,
  `INVALID_ARGS: args.remedy: 'gift' is not one of ['refund', 'exchange']; …`. Count them in metrics.
- `LLMError(kind, …)`: `connection | timeout | http_status | bad_response | reasoning`, raised after one
  retry of the HTTP request (connection, timeout, 5xx). 4xx is not retried. Reasoning content in the
  response is a misconfiguration and raises (`kind="reasoning"`).
- `FakeLLM(script | {role: script} | callable, usage=None)`; items: `("tool", {args})`,
  `[(…), (…)]` (two calls), `"text"` (no call), `Step(...)`, an exception, or a callable. Scripted calls
  pass through the same validation, so adversarial outputs produce the real protocol errors.
  `fake.requests` records every request (deep copies). Exhaustion raises `FakeLLMExhausted`
  (an `AssertionError`, deliberately not an `LLMError`).
- `LlamaServer(profile, port, log_path, …).start()` → `{server_spawn_to_ready_ms, first_request_ms, warmup,
  model_page_cache, model_page_cache_resident_pct, engine_version, engine_version_mismatch_allowed,
  model_sha256, server_log_facts{n_slots, n_ctx_slot, kv_unified, n_threads}}`; `stop()`, `pid`, `argv`,
  `health()`, `metrics()`, `metrics_text()`, context manager. Refuses: wrong sha256, wrong engine version
  (unless `allow_engine_mismatch=True`, then recorded), port in use. Strips `LLAMA_ARG_*` from the child env.
  Helpers: `server_healthy(url)`, `pid_listening_on(port)` (for `--llama-url` attach), `parse_prometheus`,
  `evict_page_cache(path)` (true cold start without sudo), `page_cache_residency(path)`.
- `telemetry.measure_gpu_baseline()` **before** launching the server; then per case
  `MemoryProbe({"adapter": os.getpid(), "llama-server": server.pid}, gpu_baseline_bytes=…)` →
  `start()` (resets the kernel high-water), `mark(label)` at turn boundaries, `stop()`, `report()` →
  `peak_ram_mb, peak_vram_mb, peak_unified_gpu_mb, rss_mb, memory_architecture, memory_notes (list),
  per_process, gpu_in_use_peak_mb_absolute, gpu_baseline_mb, memory_samples, memory_marks`.
- `telemetry.static_hardware()` returns exactly the SPEC §8 object on this machine (checked: equal);
  `telemetry.run_conditions(own_pids)` for `configuration.run_conditions` / preflight.
- `profiles.get_profile("final"|"baseline"|"B"|"F"|…)`, `Profile.engine_argv(binary, model, port)`,
  `.app_switches()`, `.caches()`, `.model_sha256`; module constants `MODEL_REVISION`, `ENGINE_VERSION`.

## Profiles (SPEC §9)

Every profile pins `-ngl 99 -fa on -c 8192 -np 1 --jinja --reasoning off --cache-ram <MiB>
--ctx-checkpoints 32 -t 5 --host 127.0.0.1 --metrics --no-webui --seed 42` (plus `-m`, `--port`).
`-t 5` equals llama.cpp's own choice here (`n_threads = 5` in the log; 5 "Super" cores);
`--ctx-checkpoints 32` is the engine default, pinned. The server log confirms the effective values
(`n_slots = 1, n_ctx_slot = 8192, kv_unified = false, n_threads = 5`) and `LlamaServer` records them.

| profile | model | protocol | prefetch | structured report | cache-ram MiB |
|---|---|---|---|---|---|
| B | Q8_0 | native | no | no (free text) | 8192 |
| F | Q4_K_M | constrained | yes | yes | **2048** |
| F-q8 / F-native / F-noprefetch / F-freetext / F-cacheram-default | F with exactly one switch reverted to B | | | | |
| B-q4 | Q4_K_M, otherwise B | | | | 8192 |
| B-repeat | identical to B | | | | 8192 |

`tests/unit/test_profiles.py` asserts that each leave-one-out profile differs from F in exactly one
switch and that switch equals B's value.

**Why 2048 MiB.** A prompt-cache entry is the slot's prompt state (68–80 MiB) plus its context
checkpoints (50.25 MiB each); entries in the agent workload were 228–371 MiB (3–6 checkpoints) and the
cache never held more than one, because a restored entry leaves the cache. The worst single entry the
pinned flags allow is 32 × 50.25 + 80 ≈ 1688 MiB, so 2048 always fits one (a role switch can always be
restored) while capping the never-restored garbage at 2 GiB instead of 8. 1024 would be enough for
SPEC's step caps (≤ ~13 checkpoints ≈ 730 MiB) but has no margin. Measured cost of the bound: none (below).

## Measurements

### Cold start (`test_cold_start_and_integrity`, `test_true_cold_start_after_page_cache_eviction`)

`server_spawn_to_ready_ms` = `Popen` → first `/health` 200; `first_request_ms` = a 1-token request;
the agent step is a cold 823-token Resolution prefill + decode. `ssd_cold` = model file evicted from
the page cache with `evict_page_cache` (verified 0.0 % resident by mincore before spawn).

| quant | page cache at spawn | spawn → ready ms | first request ms | first agent step ms | second agent step ms |
|---|---|---|---|---|---|
| Q4_K_M | 0.0 % (SSD cold) | 1558.5 | 56.2 | 1127.7 | 583.0 |
| Q4_K_M | 99.6 % (warm) | 717.2 | 60.7 | 1137.2 | 590.5 |
| Q8_0 | 0.0 % (SSD cold) | 2086.4 | 63.0 | 980.0 | 442.5 |
| Q8_0 | 99.8 % (warm) | 791.8 | 57.6 | 982.4 | 444.8 |

The weights are fully read during load (100 % resident right after the first step either way), so a
cold page cache costs +0.8 s (Q4) / +1.3 s (Q8) before `/health` and nothing afterwards. The first
start of the same suite found Q4 52.4 % resident (spawn → ready 1107.9 ms) and Q8 0 % (2197.6 ms):
residency between runs depends on memory pressure, which is why `startup.model_page_cache` is recorded
for every start. Idle server after load: phys_footprint 540 MiB (Q4) / 564 MiB (Q8), RSS 3150 / 4127 MiB,
AGX "In use system memory" +2901 / +4733 MiB over the idle baseline.

### Protocols (`test_native_protocol_parses_without_http_400`, `test_constrained_protocol_always_parses_and_respects_schema`)

Eight Resolution messages (refund, exchange with size, ambiguous order, off-account IDs, "send back",
size-down without IDs, verbal yes, an `<|im_end|>` injection) plus two Ops briefs.

- Native: 0 HTTP 400s, 8/8 Resolution answers parsed per quant. Q4 called `find_order_lines` 8/8 times
  (the baseline has no prefetch, so this is expected); Q8 6 × `find_order_lines`, 2 × `handoff`.
- Constrained: 30 steps per quant with `allowed` ∈ {all, handoff+reply, reply only} for Resolution and
  {all, create_proposal only, report only} for Ops: 30/30 parsed, valid and inside `allowed` for both
  quants. The decoder's bytes are exactly `json.dumps({"tool": …, "args": …})`, which is what
  `encode_constrained_call` replays (cache stays aligned: R2 cached 864 of 904).

### Prompt cache across a realistic turn (`test_prompt_cache_reuse_and_breakers`, Q4_K_M)

| protocol | step | prompt | cached | re-prefilled | prefill ms | generated | decode ms | tool |
|---|---|---|---|---|---|---|---|---|
| constrained | R1 cold | 823 | 0 | 823 | 561 | 42 | 572 | handoff_to_operations |
| constrained | R2 append-only, allowed narrowed | 904 | 864 | 40 | 94 | 59 | 809 | reply_to_customer |
| constrained | O1 other role | 512 | 0 | 512 | 364 | 15 | 194 | check_eligibility |
| constrained | O2 append-only | 555 | 526 | 29 | 85 | 19 | 250 | create_proposal |
| constrained | O3 append-only, terminal only | 608 | 573 | 35 | 86 | 19 | 266 | report_to_resolution |
| constrained | R3 back to R (host-cache restore) | 996 | 962 | 34 | 86 | 59 | 813 | reply_to_customer |
| constrained | R turn 2 (new user block) | 832 | 665 | 167 | 162 | 42 | 572 | handoff_to_operations |
| constrained | BREAK tool list changed | 931 | 0 | 931 | 637 | 59 | 824 | reply_to_customer |
| constrained | BREAK dynamic text in system prompt | 1011 | 0 | 1011 | 664 | 59 | 813 | reply_to_customer |
| constrained | BREAK replayed call differs from generated | 1003 | 665 | 338 | 236 | 77 | 1076 | reply_to_customer |
| native | R1 cold | 899 | 0 | 899 | 599 | 32 | 442 | find_order_lines |
| native | R2 append-only, allowed narrowed | 970 | 930 | 40 | 88 | 63 | 868 | handoff_to_operations |
| native | O1 other role | 700 | 0 | 700 | 525 | 17 | 224 | check_eligibility |
| native | O2 append-only | 745 | 716 | 29 | 84 | 28 | 378 | create_proposal |
| native | O3 append-only, terminal only | 807 | 772 | 35 | 92 | 16 | 210 | execute_proposal (TOOL_NOT_ALLOWED) |
| native | R3 back to R (host-cache restore) | 1066 | 1032 | 34 | 86 | 53 | 732 | reply_to_customer |
| native | R turn 2 (new user block) | 908 | 741 | 167 | 162 | 32 | 436 | find_order_lines |
| native | BREAK tool list changed | 992 | 0 | 992 | 687 | 52 | 716 | reply_to_customer |
| native | BREAK dynamic text in system prompt | 1081 | 383 | 698 | 522 | 53 | 730 | reply_to_customer |
| native | BREAK replayed call differs from generated | 1062 | 565 | 497 | 381 | 53 | 730 | reply_to_customer |

Mechanism (from the `-lv 4` log): checkpoints are created at the start of the last user message
(token 665 = everything before `<|im_start|>user`, verified with `/apply-template` + `/tokenize`), at
token 304, and near the end of each processed prompt (up to 7 per slot in the 3-case run); a new task that diverges inside the cached history restores the
newest checkpoint before the divergence and erases the later ones. Native mode puts the tools block at
the very top, so a system-prompt change keeps a little (383) while constrained (docs after the role
prompt, inside the same system message) keeps nothing; both lose everything when the tool list changes.

Card layout (`test_new_turn_reuses_more_when_the_card_is_its_own_user_message`, constrained Q4, turn 2
after a role switch): card inside the last user message → cached 665 / re-prefilled 159 (156 ms);
card as its own earlier user message → cached 759 / re-prefilled 69 (95 ms). Same tool chosen.

### Throughput (`test_decode_and_prefill_throughput`)

Decode: 128 tokens with `ignore_eos`, median of 5. Prefill: `cache_prompt: false`, median of 3.

| quant | decode tok/s (range) | prefill tok/s ~1000 tokens | prefill tok/s ~2000 tokens |
|---|---|---|---|
| Q4_K_M | 70.5 (69.8–70.9) | 1522.6 | 1529.0 |
| Q8_0 | 50.8 (50.3–51.1) | 1693.3 | 1709.5 |

Decode is memory-bandwidth bound (Q4 moves fewer bytes per token: +39 %); prefill is compute bound and
Q8 avoids the 4-bit dequantisation work (−10 % for Q4). The agent workload is decode-dominated (78 % of
wall time), so Q4 is faster end to end.

### Memory (`test_memory_footprint_and_unified_gpu_memory`, 3 cases = 30 constrained calls)

| quant | peak_ram_mb (Σ high-water) | llama-server high-water | adapter (pytest) high-water | rss_mb (Σ peak RSS) | llama-server peak RSS | peak_vram_mb (AGX − baseline) | AGX idle baseline | samples |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 1728.9 | 1675.0 | 53.9 | 4116.2 | 4067.0 | 3214.8 | 814.9 | 69 |
| Q8_0 | 1743.6 | 1688.5 | 55.1 | 5576.0 | 5525.9 | 4910.8 | 661.3 | 94 |

Reading it: the weights are clean file-backed mmap pages, so they are in RSS and in the GPU figure but
not in phys_footprint. The server's footprint (~1.7 GiB for both quants) is KV/compute buffers plus
context checkpoints plus at most one prompt-cache entry; it does not depend on the quantisation. The
GPU figure tracks the weights (Q8 +1.7 GB over Q4). RSS overcounts by ~2.4 GB (Q4) / ~3.8 GB (Q8),
which is why it is reported only for comparison. The kernel high-water caught a 256 MiB spike that was
freed before any sampler could see it (`test_kernel_high_water_catches_a_spike_that_sampling_misses`).

### Host prompt cache size (`test_cache_ram_bounded_vs_default`)

Fresh Q4 server per setting, one warm-up case, then 8 distinct cases × 10 calls (turn 1: R, R, O, O, O, R;
turn 2: R, O, O, R), constrained. Order of settings: 2048, 8192, 1024, 0 (one block, not interleaved).

| --cache-ram MiB | wall s | cached / prompt tokens | prefill s | decode s | step p50 ms | step p95 ms | server high-water MiB | peak unified GPU MiB | decision trace |
|---|---|---|---|---|---|---|---|---|---|
| 2048 (F) | 38.2 | 52209 / 56597 (92.2 %) | 7.8 | 29.8 | 442.1 | 869.6 | 1927.0 | 3300.4 | 49d2cfef87c76f62 |
| 8192 (default) | 38.4 | 52209 / 56597 (92.2 %) | 7.9 | 30.0 | 443.6 | 890.6 | 1925.7 | 3284.6 | 49d2cfef87c76f62 |
| 1024 | 39.1 | 52209 / 56597 (92.2 %) | 8.0 | 30.5 | 443.3 | 915.8 | 1927.8 | 3268.8 | 49d2cfef87c76f62 |
| 0 (off) | 52.7 | 30422 / 56597 (53.8 %) | 21.3 | 31.2 | 583.5 | 1270.8 | 884.8 | 3471.7 | a0e6e1275e855eb4 |

The bound changes nothing measurable in this pattern (an earlier identical run gave 1024: 38.5 s /
1707 MiB, 8192: 38.6 s / 1693 MiB). Switching the host cache off saves ~1 GiB of host memory and costs
+38 % wall time (re-prefill of the other role after every switch). Per-call overhead outside the
engine's prefill/decode timers (HTTP, template, grammar setup): 7.1–7.8 ms. p95 here is nearest-rank
over 80 steps.

## Deviations from SPEC (and why)

1. **Native tool schemas are flattened** (`flatten_one_of`): a top-level `oneOf` yields `{}` arguments on
   this engine (finding 2). The strict union is still what code validates. SPEC §7 did not say how
   tagged unions go over the native protocol.
2. **F's cache-ram = 2048 MiB**, justified above. The measurement says the bounded-cache change is a
   no-op for latency and memory in the agent pattern; it remains in F as a worst-case cap, and
   F-cacheram-default should be read as a null-effect ablation.
3. **`page_cache` state is measured at spawn**, after the hash check (a first-time hash reads the file
   and would make "cold" meaningless). `evict_page_cache` added for true cold-start runs (not in SPEC).
4. Additive API beyond SPEC: `Step.role/finish_reason/text/http_attempts`, `LLMClient(retry_backoff_s,
   transport)`, `LLMError.kind`, extra keys in `start()` and `MemoryProbe.report()`, `memory_notes` is a
   list of strings, `profiles.variant()` and aliases `final`/`baseline`.
5. Reasoning content in a response raises `LLMError(kind="reasoning")` (SPEC: "assert no reasoning
   content"); it never happened with `--reasoning off` + `enable_thinking: false`.

## Open issues for the next stage

- `.gitignore` should list `models/.sha256-cache.json` (I do not own `.gitignore`).
- Orchestrator/prompts: put the account card in its own user message (finding 9); keep tool lists
  constant per role and history byte-identical (findings 7–8); feed every protocol error back as the
  tool result and count it; native will produce `TOOL_NOT_ALLOWED` and extra-ID `INVALID_ARGS`
  (seen in exploration: `execute_confirmed` with `order_id`/`line_id`), which the B profile must absorb.
- Adapter: call `measure_gpu_baseline()` before `LlamaServer.start()`; put `startup` (incl.
  `model_page_cache`) at the root; use `pid_listening_on(port)` when attaching via `--llama-url`
  (then `peak_vram_mb` is null with a note, since no idle baseline exists).
- `hardware.power` is a declared string; the bench preflight must enforce it using `run_conditions()`
  (`on_ac_power`, `powermode == 0`, `sleep_prevented`, no `other_llama_servers`, no CPU hogs).
- Not measured, proposal only: `--ctx-checkpoints 8` (caps checkpoint memory at ~400 MiB per slot, may
  drop the start-of-user checkpoint in long turns), speculative decoding with the 0.8B draft model.
- The cache-ram comparison ran one block in a fixed order; the bench's interleaved F vs
  F-cacheram-default runs are the authoritative comparison.
