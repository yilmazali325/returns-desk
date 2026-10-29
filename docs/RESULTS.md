# Results: baseline B vs final F

Everything in sections 1–8 is **measured**: recomputed by `bench.report` from the raw per-run records
in `results/` (full tables in [results/REPORT.md](../results/REPORT.md), CSVs in `results/csv/`).
Section 9 is **proposed** and not measured. Known gaps are in [LIMITATIONS.md](LIMITATIONS.md).

- **Machine:** Apple M5 Pro, 24 GB unified memory, macOS 26.4.1, on AC power.
- **Engine and model:** llama.cpp `llama-server` 0.7.0 (build 11669, commit 620d2225c, Metal),
  Qwen3.5-4B. F uses Q4_K_M weights, B uses Q8_0. Model files are hashed fresh for every suite
  (`results/<suite>/model_sha256.json`).
- **Profiles:** B is the baseline. F is the final profile: the same desk code with five changes
  (section 4). Both run the same orchestrator, gates and store.
- **Method:**
  - **Schedule:** interleaved, seed 42, with a fresh `llama-server` for every run.
  - **Clean-machine check:** before every run, a preflight checks CPU, GPU, power and thermals.
  - **Success:** strict success scored by an independent oracle (`evals/`), with Wilson 95%
    intervals.
  - **Percentiles:** nearest-rank. With fewer than 20 samples, p95 is the maximum.
  - **Latency comparisons:** paired, with a 10,000-resample case-level bootstrap. B run twice
    (`B-repeat`) gives the A/A noise floor.

## 1. Quality

Strict means every turn's decision, the proposal binding and the final business state all match the
labels, with no invariant violation. Two "floors" run the same desk with no model:
- **always-clarify:** an agent that only ever asks a question. Whatever it passes, code alone solved.
- **fake-policy:** our scripted test double.

| suite | cases × runs | B | F | always-clarify floor | fake-policy double |
|---|---|---|---|---|---|
| public | 6 × 3 | 18/18 | 18/18 | 1/6 | 6/6 |
| variants | 241 × 1 | 241/241 | 241/241 | 54/241 | 241/241 |
| holdout_v2 | 57 × 1 | 56/57 | 56/57 | 11/57 | 50/57 |
| holdout_v3 | 60 × 1 | 57/60 (lenient 58/60) | 58/60 | 9/60 | 56/60 |

- **Invariant violations:** 0 in every run of every profile. The invariants cover duplicate money
  or stock, invented success, and a mutation without a grant.
- **Code alone completes nothing.** The always-clarify floor made 0 refunds or exchanges on every
  suite. Its decisions were only `clarify` and `deny`.
- **The model's share** is F's strict rate minus the floor's (agent-attributable): +83.3 pp public,
  +77.6 pp variants, +78.9 pp holdout_v2, +81.7 pp holdout_v3.
- **Repeatability:** on the public suite, every profile made identical decisions and wrote identical
  text in all 3 runs. The held-out sets ran once per profile, so they have no repeatability figure.

Every failure in the final runs:

| case | profile | what happened | safe? |
|---|---|---|---|
| hx12_spanish_mix_refund_jacket | B, F | "la chaqueta" can't be matched as a citation on a multi-item account, so the desk asks which item | yes: asks, changes nothing |
| v3_n22 | B, F | The confirm turn says "the t-shirt" (with typos) on a jeans proposal. Nothing executes; the desk asks, then proposes the t-shirt one turn later than the label | yes: no wrong refund |
| v3_n30 | B, F | "Too tight" with no remedy named: the agents propose the next size up; the label expects a refund-or-exchange question. The label is debatable | yes: a proposal changes nothing until confirmed |
| v3_n28 | B | "Is the exchange done yet?" with an open proposal: B asks instead of re-presenting the proposal. Accepted by the lenient score | yes |

Other counts:
- **LLM step timeouts:** 0 in every run, including the held-out runs on a busy GPU.
- **Hitting the step or call cap:** B 10 times (4 on variants, 2 on v2, 4 on v3); F never.
- **Protocol errors:** F 0 on every suite. B 37 (25 on variants, 12 on the held-out sets). On public, F-native has 12 and
  B-q4 has 9, all handled by guards.

## 2. Latency (public suite, clean machine)

| profile | case p50 | case p95 (= max, n=18) | turn p50 | turn p95 (n=39) | case mean |
|---|---|---|---|---|---|
| B | 11,434 ms | 13,205 ms | 4,784 ms | 8,249 ms | 11,128 ms |
| **F** | **5,044 ms** | **5,894 ms** | **1,876 ms** | **3,910 ms** | **4,796 ms** |
| B-repeat (A/A) | 11,095 ms | 12,451 ms | 4,584 ms | 8,350 ms | 10,920 ms |

- **Paired F/B ratio:** 0.425, 95% CI [0.398, 0.449], so F is **57.5% faster**. The A/A ratio
  (B-repeat/B) is 0.982 [0.968, 0.996], so the effect is far outside the noise.
- **Variants suite, F (clean):** p50 5,150 ms, p95 9,411 ms over 241 cases.
- **Held-out suites:** F/B 0.423 on holdout_v2 and 0.431 on holdout_v3. These four runs, and
  variants-B, started after the preflight refused the machine (GPU busy from another app, `--allow-dirty`). The report
  marks their timings as not clean, and no latency claim here rests on them.

## 3. Throughput and where the time goes (public, means per case)

| profile | LLM calls | completion tokens | decode tok/s | uncached prefill tok/s | prompt cache hit | decode share of case time | cases/min (serial, derived) |
|---|---|---|---|---|---|---|---|
| B | 9.33 | 475 | 51.0 | 1,079 | 89.5% | 83.6% | 5.4 |
| F | 8.33 | 243 | 73.7 | 1,110 | 87.6% | 68.8% | 12.5 |

- Token rates are token-weighted (total tokens / total time). They match the server's own
  `/metrics` (F 73.8, B 51.0).
- Cases per minute is 60 s / mean case time, one case at a time. Concurrent throughput was not
  measured.
- Code is cheap: tools take under 1 ms per case and orchestration about 0.1 s. Decoding is most of
  the time.

## 4. What each change bought (leave-one-out, public)

Each row is F with that one change put back to B's value, 3 runs each. Every row kept 18/18 strict.

| F's change (B's value) | F faster by | 95% CI | other measured effect per case |
|---|---|---|---|
| tool calls by JSON-schema-constrained decoding (B: native tool calling) | 36.8% | 30.5–44.0% | −183 completion tokens, −0.8 LLM calls |
| Q4_K_M weights (B: Q8_0) | 21.2% | 17.8–23.6% | −1,696 MiB peak GPU memory |
| code-built Operations report (B: model free text) | 18.7% | 16.6–20.6% | −72 completion tokens |
| prefetched account card (B: off) | 10.3% | 7.4–13.2% | −1.2 LLM calls |
| `--cache-ram 2048` MiB (B: 8192) | **no effect** | −1.5% to +0.7% | −104 MiB peak GPU memory |

The effects overlap, so they don't add up to the 57.5% total. The prompt-cache change is a measured
null result for speed; it stays in F because it uses less memory.

Pre-registered gate (SPEC §9): a change ships only if F ≥ B on public and on variants, with zero
invariant violations, and F ≥ F-without-that-change on public. **Result: F as a whole and each of
the five changes pass.**

## 5. Memory (public, MiB; unified memory, so never add RAM and GPU)

| profile | peak RAM median (max) | peak unified GPU median (max) |
|---|---|---|
| B | 1,801 (1,932) | 4,800 (5,080) |
| F | 2,107 (2,220) | 3,096 (3,235) |

- **Peak RAM** is the sum of the kernel phys_footprint high-water marks of the adapter and
  `llama-server`.
- **Peak GPU** is IOAccelerator in-use memory minus the idle baseline. It is system-wide and overlaps
  RAM.
- F uses 1.7 GiB less GPU memory and about 0.3 GiB more RAM. The RAM increase comes with the Q4_K_M
  weights: F-q8 is 1,842.
- **Not measured:** per-process GPU memory and energy both need root, which the bench doesn't use.
  Apple Silicon has no separate VRAM.

## 6. Cold start vs warm

| | F | B |
|---|---|---|
| `llama-server` spawn → ready, model file in page cache | 678–730 ms | — (every B run started cold) |
| spawn → ready, model read from SSD | 1,725–1,926 ms | 2,150–2,506 ms |
| first request (Metal pipeline compile) | 56–80 ms | 59–64 ms |
| first case after start vs warm median, public runs | 5.5–5.7 s vs 5.0–5.1 s | 12.3–12.4 s vs 10.9–11.4 s |

The first case pays about 0.5 s (F) or 1 s (B) for a cold prompt cache. The server start is excluded
from case latency.

## 7. The held-out story: blind run, two fixes, final run

| holdout_v3 run | F | B | what it showed |
|---|---|---|---|
| 1, blind (2026-10-28) | 56/60 | 56/60 | On a confirm turn whose text changed the item ("Yes, refund the t-shirt" on a jeans proposal), code forced the old proposal to execute and refunded the jeans. **The invariant checks reported 0 violations**; only the end-state comparison with the labels caught it. |
| 2, after fix 1 (2026-10-29) | 57/60 | 56/60 | The fix withdrew the confirmation only when the agent noticed the change. B's agent didn't, and code compared only IDs, so B still refunded the jeans. Code also blocked a correct agent: "What about a 30 then?" was refused because of an older request for 34. |
| 3, after fix 2 (2026-10-29) | 58/60 | 57/60 | Code now reads product names on confirm turns, and a size in the latest message beats an older one. No wrong refund or exchange on any suite. |

Runs 1 and 2 are archived in `results/evidence/bench_2026-10-28_prefix/` and
`results/evidence/bench_2026-10-29_fix1/`. Every test set, holdout_v3 included, has now been used
to fix bugs, so these are regression scores, not generalization scores.

## 8. How sure we are

- **High:** F is about 2.3× faster than B on this machine at equal quality. The interval is tight
  and the A/A noise is small.
- **High:** no invariant violation and no wrong mutation in the final runs.
- **Medium:** the quality numbers. The samples are small (6 public cases), every set has been used
  for tuning, and the held-out sets ran once per profile. The grader's hidden variants are the real
  test.
- **Medium:** the held-out quality is unaffected by the busy GPU. There were no step timeouts, and
  decoding is greedy and seeded, but those runs were not repeated on a clean machine.

## 9. Proposed improvements (not measured)

- **Typo-tolerant, multilingual product lexicon for the citation gate.** This would fix hx12,
  v3_n22 and the remaining wrong-execution path in LIMITATIONS.
- **Speculative decoding with a small draft model.** Decoding is 69% of F's case time. The draft
  model file is in `models/`, but nothing has been measured.
- **A blind test set written by people**, plus 3 runs per held-out set to measure run-to-run
  variation.
- **Held-out and variants-B runs on a clean machine**, to replace the five runs marked not clean.
