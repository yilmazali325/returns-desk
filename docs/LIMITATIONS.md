# Known limitations

These are known and not fixed. Measured numbers are in [RESULTS.md](RESULTS.md); proposed
improvements below are not measured unless stated.

## Evaluation honesty

- **Most of our test sets are no longer unseen.**
  - The public cases, the 241 variants and `holdout_v2` were all used while tuning, so their scores
    measure regression, not generalization.
  - `holdout_v3` was written blind and run blind exactly once (results archived in
    `results/evidence/bench_2026-10-28_prefix/`). That run found the bug fixed below, so v3 is a
    regression suite now too.
  - The grader's hidden variants are the real unseen test. The biggest remaining risk is wording
    we have not seen.
- **The test sets were written by the same AI tooling that built the desk.** `holdout_v3` came from
  a separate agent that had not read the grounding code, but it is still the same model family. A
  human-written set would be a stronger test.
- **Small samples.** The public suite has 6 cases. 18 pooled case-runs give a wide Wilson interval,
  and with fewer than 20 samples p95 is the maximum. Pooled runs of the same cases are not
  independent. The report states both.
- **The held-out sets ran once per profile, on a busy GPU.** Another app kept the GPU about 37%
  busy for over 2 hours, so the four held-out runs (and variants-B) used `--allow-dirty`. The
  report marks their timings as not clean, and no latency claim rests on them. There were no step
  timeouts, so quality is very likely unaffected, but those runs have no repeat.
- **The scripted fake-policy agent scores close to the real model** (`holdout_v3` 56/60, against
  58/60 for F and 57/60 for B in the final run). It is our own test double, written for our vocabulary. Its score shows that the
  suites stay within that vocabulary; it does not show the LLM is unnecessary. The always-clarify
  floor (code with no agent) completes no transaction at all.

## Behaviour

- **Wrong-refund bugs that were fixed.** The blind v3 run found that on a confirmation turn whose
  text changed the request ("Yes, refund the t-shirt" on a jeans proposal), code forced the old
  proposal to execute. The first fix covered only an agent that reads the change; the next bench
  showed B still refunding the jeans, because B's agent executed and code checked only IDs. Code
  now also reads product words. Both are fixed, but no fresh blind set confirms the fixes generalize.
- **A change nobody can read still executes.** If the confirmation message changes the item with a
  misspelling ("Yes, refund the t-shrit") and the agent misses it too, the bound proposal executes.
  The confirm control was still pressed for that proposal, so this is the customer's explicit
  confirmation, but it is not what they meant. A typo-tolerant product lexicon would close it
  (proposed).
- **Customer typos in product words can't be verified.** "t-shrit" or Spanish "la chaqueta" cannot
  be matched as a citation, so on multi-item accounts the desk asks which item. It fails safe (asks)
  but costs a turn. A multilingual or typo-tolerant product lexicon would fix it (proposed).
- **English only.** Prompts, grounding and the reply guard assume English.
- **Wasted steps.** On a typed "yes" without the confirm control, F first tries to execute (refused,
  about one LLM step) before re-presenting the proposal. It never causes an action.
- **The reply guard can make replies blunt.** When the model writes an unsupported number, ID or
  claim, the guard swaps in a template reply. The decision is unaffected, but the wording is generic.
  In the final v3 run it fired 5 times across the 60 cases for F, and 16 times for B.
- **Run-to-run variation.** Decoding is greedy and seeded, but small timing-dependent differences in
  the prompt cache can still flip a borderline case between runs. The public suite was identical
  across 3 runs; unseen wording may vary more.

## Scope (by design, from the brief)

- Refunds and same-product, same-price size exchanges only. There are no partial quantities,
  price-difference collection, tax, shipping or real payment services.
- One customer per session; the customer identity comes from the fixture or UI, not from
  authentication.
- The business tools and store are simulated (SQLite). There is no real order system,
  warehouse or payment provider.

## Engineering

- **SIGKILL of the adapter leaves its llama-server running.** Nothing runs on SIGKILL. The next run
  moves to the next free port, and the benchmark preflight finds and refuses any leftover
  llama-server.
- **Metrics on a moved port.** When the adapter moves llama-server to another port because the
  requested one is busy, the bench's `/metrics` scrape of the original port misses it. This is
  recorded, not silent.
- **The bench relies on the adapter's log format** to find case boundaries. If that format changes,
  the affected metrics are marked "not captured".
- **Latency is for one machine.** Apple M5 Pro, unified memory, Metal. Numbers on other machines will
  differ; the comparisons (F vs B paired on the same machine) are what carry over.
- **Unified memory.** On Apple Silicon RAM and GPU memory are one pool. We report the process
  footprint and the GPU allocation separately and never add them; "VRAM" here means the GPU's
  unified-memory allocation.
- **The confirm control trusts the browser session.** Confirmation is bound to the proposal's binding
  hash and generation, and the server answers only to localhost, but there is no user
  authentication.
