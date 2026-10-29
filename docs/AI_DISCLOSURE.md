# AI tools used

The brief encourages AI coding assistants and asks for disclosure. This project was built with:

- **Claude Code** (Anthropic), in the Claude desktop app, running **Claude Opus 5.5**
  (`claude-opus-5-5`).

## What it was used for

- **Design and spec.** Drafting `docs/SPEC.md`, the clarification questions and the benchmark plan.
  It also ran an adversarial critique of the spec (`docs/notes/spec_critique.json`).
- **Code.** Most of the code in `desk/`, `server/`, `bench/`, `evals/` and `scripts/` was generated
  by Claude Code, then iterated with tests.
- **Tests.** Unit, review and Playwright tests, including the repro tests for review findings in
  `tests/review/`.
- **Reviews.** Multi-agent adversarial reviews (business rules, red team, code, security,
  contract), with every finding independently verified before it was fixed (`docs/review/`).
- **Evaluation data.** The variant suite, `holdout_v2` (red-team) and `holdout_v3`. v3 was written by
  a separate agent that was not shown the grounding code, then frozen by checksum
  (`evals/holdout_v3/FROZEN.md`).
- **Docs.** README, architecture, results, limitations and the slide deck.

## What was not AI

- **The local agents at runtime are not Claude.** The desk's two agents run Qwen3.5-4B locally
  through llama.cpp. No cloud model or API is used at runtime, and the demo works offline.
- **The numbers come from local runs, not from a model's claims.** Every number in the results was
  produced by local runs and recomputed from the per-run records by `bench/report.py`.

## Responsibility

The candidate directed the work, made the product and design decisions, and is responsible for
all of the code, including the AI-generated parts. The code is meant to be explained and changed
live; `docs/STUDY_GUIDE.md` maps where each rule lives and walks through likely live changes.
