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
  text in all 3 runs. The held-out sets and variants-B ran twice, three days apart (2026-10-29 on a
  busy GPU, 2026-11-01 on a clean machine): all 475 case-runs had identical decisions, text and
  final state. The first runs are archived in `results/evidence/bench_2026-10-29_busygpu/`.
- **After the CLARIFICATIONS #18 change** (a confirm flag with nothing to confirm and no item named
  now asks which item), the whole matrix ran again on a clean machine on 2026-11-01. All 878
  case-runs made the same decisions and reached the same final state as before. The text differed
  in 2: hx12 turn 2 in B and F ("gracias, sí confirmo"), the one stored turn where the new rule
  fires, which was `clarify` before and still is. The previous runs are archived in
  `results/evidence/bench_2026-11-01_pre_confirm_rule/`.

Every failure in the final runs:

| case | profile | what happened | safe? |
|---|---|---|---|
| hx12_spanish_mix_refund_jacket | B, F | "la chaqueta" can't be matched as a citation on a multi-item account, so the desk asks which item | yes: asks, changes nothing |
| v3_n22 | B, F | The confirm turn says "the t-shirt" (with typos) on a jeans proposal. Nothing executes; the desk asks, then proposes the t-shirt one turn later than the label | yes: no wrong refund |
| v3_n30 | B, F | "Too tight" with no remedy named: the agents propose the next size up; the label expects a refund-or-exchange question. The label is debatable | yes: a proposal changes nothing until confirmed |
| v3_n28 | B | "Is the exchange done yet?" with an open proposal: B asks instead of re-presenting the proposal. Accepted by the lenient score | yes |

Other counts:
- **LLM step timeouts:** 0 in every run.
- **Hitting the step or call cap:** B 10 times (4 on variants, 2 on v2, 4 on v3); F never.
- **Protocol errors:** F 0 on every suite. B 37 (25 on variants, 12 on the held-out sets). On public, F-native has 12 and
  B-q4 has 9, all handled by guards.

## 2. Latency (public suite, clean machine)

| profile | case p50 | case p95 (= max, n=18) | turn p50 | turn p95 (n=39) | case mean |
|---|---|---|---|---|---|
| B | 11,519 ms | 13,162 ms | 4,597 ms | 8,149 ms | 10,934 ms |
| **F** | **5,034 ms** | **5,742 ms** | **1,878 ms** | **3,869 ms** | **4,767 ms** |
| B-repeat (A/A) | 11,776 ms | 12,778 ms | 4,572 ms | 8,226 ms | 10,991 ms |

- **Paired F/B ratio:** 0.429, 95% CI [0.397, 0.456], so F is **57.1% faster**. The A/A ratio
  (B-repeat/B) is 1.005 [0.998, 1.014], so the effect is far outside the noise.
- **Variants suite, F (clean):** p50 4,456 ms, p95 7,936 ms over 241 cases.
- **Variants and held-out suites (one run each, clean machine):** paired F/B 0.409
  [0.404, 0.415] on variants, 0.411 [0.400, 0.422] on holdout_v2 and 0.424 [0.414, 0.434] on
  holdout_v3. These suites have no A/A run, so their noise floor is not measured.

## 3. Throughput and where the time goes (public, means per case)

| profile | LLM calls | completion tokens | decode tok/s | uncached prefill tok/s | prompt cache hit | decode share of case time | cases/min (serial, derived) |
|---|---|---|---|---|---|---|---|
| B | 9.33 | 475 | 51.6 | 1,101 | 89.5% | 84.1% | 5.5 |
| F | 8.33 | 243 | 74.0 | 1,120 | 87.6% | 68.9% | 12.6 |

- Token rates are token-weighted (total tokens / total time). They match the server's own
  `/metrics` (F 74.0, B 51.7).
- Cases per minute is 60 s / mean case time, one case at a time. Concurrent throughput was not
  measured.
- Code is cheap: tools take under 1 ms per case and orchestration about 0.1 s. Decoding is most of
  the time.

## 4. What each change bought (leave-one-out, public)

Each row is F with that one change put back to B's value, 3 runs each. Every row kept 18/18 strict.

| F's change (B's value) | F faster by | 95% CI | other measured effect per case |
|---|---|---|---|
| tool calls by JSON-schema-constrained decoding (B: native tool calling) | 35.8% | 30.0–42.3% | −183 completion tokens, −0.8 LLM calls |
| Q4_K_M weights (B: Q8_0) | 22.3% | 19.4–25.2% | −1,687 MiB peak GPU memory |
| code-built Operations report (B: model free text) | 17.6% | 15.1–19.6% | −72 completion tokens |
| prefetched account card (B: off) | 9.7% | 6.5–12.9% | −1.2 LLM calls |
| `--cache-ram 2048` MiB (B: 8192) | **no effect** | −0.3% to +1.0% | none measurable (+33 MiB peak GPU, −15 MiB peak RAM) |

The effects overlap, so they don't add up to the 57.1% total. The prompt-cache change is a measured
null result for speed, and its memory effect is within run-to-run noise (−104 MiB peak GPU in the
previous bench, +33 MiB in this one). It stays in F because it passes the gate and caps the host
prompt cache at 2 GiB of RAM instead of 8.

Pre-registered gate (SPEC §9): a change ships only if F ≥ B on public and on variants, with zero
invariant violations, and F ≥ F-without-that-change on public. **Result: F as a whole and each of
the five changes pass.**

## 5. Memory (public, MiB; unified memory, so never add RAM and GPU)

| profile | peak RAM median (max) | peak unified GPU median (max) |
|---|---|---|
| B | 1,864 (2,080) | 4,792 (5,010) |
| F | 1,970 (2,218) | 3,146 (3,278) |

- **Peak RAM** is the sum of the kernel phys_footprint high-water marks of the adapter and
  `llama-server`.
- **Peak GPU** is IOAccelerator in-use memory minus the idle baseline. It is system-wide and overlaps
  RAM.
- F uses 1.6 GiB less GPU memory and about 0.1 GiB more RAM (medians). The RAM increase comes with
  the Q4_K_M weights: F-q8 is 1,786.
- **Not measured:** per-process GPU memory and energy both need root, which the bench doesn't use.
  Apple Silicon has no separate VRAM.

## 6. Cold start vs warm

| | F | B |
|---|---|---|
| `llama-server` spawn → ready, model file in page cache | 682–708 ms | — (every B run started cold) |
| spawn → ready, model read from SSD (fully or partly) | 1,540–1,969 ms | 2,089–2,302 ms |
| first request (Metal pipeline compile) | 56–60 ms | 59–63 ms |
| first case after start vs warm median, public runs | 5.5–5.6 s vs 5.0–5.1 s | 12.1–12.6 s vs 10.8–11.5 s |

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
- **High:** the desk is repeatable on these sets. Every held-out and variants-B case made the same
  decisions, wrote the same text and reached the same final state in two runs three days apart.
- **Medium:** the quality numbers. The samples are small (6 public cases) and every set has been
  used for tuning. The grader's hidden variants are the real test.

## 9. Proposed improvements (not measured)

- **Typo-tolerant, multilingual product lexicon for the citation gate.** This would fix hx12,
  v3_n22 and the remaining wrong-execution path in LIMITATIONS.
- **Speculative decoding with a small draft model.** Decoding is 69% of F's case time. The draft
  model file is in `models/`, but nothing has been measured.
- **A blind test set written by people**, plus 3 runs per held-out set to measure run-to-run
  variation.
