# Study guide: explain it, change it live

The interview has 15 minutes of questions and a small live change on this code. This guide maps
where every rule lives and rehearses the likely changes. Run `.venv/bin/pytest` (about 50 s) after
any change. The failing tests tell you what else depends on the rule.

## The 60-second tour

| If they ask about… | Open | One line to say |
|---|---|---|
| the agents | `desk/agents/resolution.py`, `desk/agents/operations.py`, `desk/agents/prompts.py` | Two loops over one LLM client. Resolution talks to the customer; Operations only gets a structured brief, never customer text. |
| what each agent may call | `desk/agents/toolspecs.py` | Tool schemas per role, byte-stable for the prompt cache; per-step limits are enforced by code. |
| one turn end to end | `desk/orchestrator.py` (`Desk.handle_turn` → `_start_turn` → `_route` → `_handoff` → `_check_propose` → `decide`) | Code grants, gates, budgets and the decision label; the agents choose. |
| "agents say yes; code can say no" | `docs/SPEC.md` §6.0, `desk/orchestrator.py` `_check_propose` | Code blocks with a reason; it never makes a proposal or a commit. |
| reading the customer's words | `desk/grounding.py` (`analyze`, `check_target`, `check_remedy`, `check_requested_size`) | A deterministic reader used only to check the agent's choice. |
| the business rules | `desk/policy.py` (`eligibility`, `refund_amount`, `resolve_size`, `validate_replacement`) | Pure functions; the window constant is `desk/domain.py` `WINDOW_DAYS`. |
| money | `desk/policy.py` `refund_amount` (net paid, integer cents), `desk/render.py` `money` | Money is never a float; one formatter, also used by the reply guard. |
| commits and idempotency | `desk/tools.py` (`_create_proposal`, `_execute_proposal`, `_short_circuit`), `desk/store.py` | One SQLite transaction per commit; the proposal binding hash; replay by `request_id`. |
| faults | `desk/faults.py`, `desk/tools.py` `begin_turn` | `stock_change` before turn N; `timeout_after_commit` fires once after a real commit. |
| what the customer sees | `desk/render.py` (`compose_response`, reply guard, templates) | LLM sentence plus a code facts line; unsupported claims are replaced by a template. |
| budgets | `desk/agents/__init__.py` `Limits` | All per-turn limits in one dataclass. |
| F vs B | `desk/profiles.py` | Named profiles: engine flags plus the app switches measured one at a time. |
| the UI | `server/app.py` (API, SSE), `server/static/app.js` (Team view) | Audit-log listeners stream events; the cards are driven by the audit kind, agent and tool. |
| the scores | `evals/oracle.py`, `evals/score.py` | An independent re-implementation of the contract rules, used only for scoring. |
| the benchmark | `bench/run.py`, `bench/report.py`, `bench/preflight.sh` | Interleaved runs, a fresh server per run, the report recomputed from raw records. |

## Live change 1: shorter return window (30 → 14 days)

1. `desk/domain.py:35`: set `WINDOW_DAYS = 14`. Policy (`desk/policy.py` `eligibility`), the reason
   text (`desk/render.py`) and the UI (`/api/meta` sends `window_days`) all read it.
2. The oracle is deliberately independent. Make the same change in `evals/oracle.py:27`, or the
   scores will disagree.
3. Run `.venv/bin/pytest`. The tests that pin 30 days fail; that is the dependency list. Update them.
4. Show it: `public_01_refund_boundary` (delivered exactly 30 days earlier) is now **deny**,
   "It is outside the 14-day return window." The agents' prompts need no change; they never
   state the window.

## Live change 2: another currency

The contract says USD. If asked for EUR, it is a formatting change, because amounts are already
integer cents.

1. `desk/render.py` `money(cents)` is the one formatter. Give it a currency (for example
   `money(cents, currency="USD")` with a symbol map) and pass `state["currency"]` from the callers.
   `desk/tools.py:526` already reads `currency` from the state.
2. The reply guard checks amounts by formatting them with `money`, so it stays consistent.
3. Tests in `tests/unit/test_render.py` pin "$". Update them, and add a EUR case.

## Live change 3: a new business rule (example: no refunds above $200; exchange instead)

1. `desk/domain.py`:
   - add `REFUND_ABOVE_LIMIT = "REFUND_ABOVE_LIMIT"`;
   - add it to `CODE_CLASS` as `REFUSAL`, which makes it a **deny**.
2. `desk/policy.py`: add a pure check, for example
   `def refund_allowed(line): return refund_amount(line) <= 20000`.
3. `desk/tools.py` `_create_proposal` (line 421): when `remedy == "refund"` and the check fails,
   return `ToolResult.failure(REFUND_ABOVE_LIMIT, …)`. Execute re-runs the same checks, so a
   confirmation can never bypass it.
4. `desk/render.py` `REASON_TEXT`: add the customer wording, for example "Refunds above $200 are not
   available; an exchange is."
5. Mirror the rule in `evals/oracle.py` and add a unit test in `tests/unit/test_policy.py`, plus one
   in `tests/unit/test_tools.py`.

What to say: the agents need no prompt change. Operations calls `create_proposal`, the gateway
refuses with a code, and the report and reply follow from that code. Rules live in code; the
agents pick and talk.

## Live change 4: a budget or an agent limit

`desk/agents/__init__.py` `Limits`: for example `ops_steps` from 4 to 3. The tests in
`tests/unit/test_agents.py` show which flows need the extra step.

## Questions to expect, with honest answers

- **"Isn't code doing all the work?"** No, and it is measured. A stub that only asks questions runs
  the same code with no model. It completes zero refunds or exchanges on every suite (public 1/6,
  variants 54/241, all asks or denies). Every completed transaction was started by an agent.
- **"Why two agents?"**
  - Separation of duties: the agent that reads customer text cannot move money or stock, and the
    agent that can never sees customer text, so injected instructions can't reach it.
  - They are distinct roles with different tools and different context.
- **"How do you stop invented success?"**
  - The decision comes from facts: a commit re-read from the store, or a replay.
  - Operations' claims are compared to the code-built report.
  - The reply guard rejects success words on a non-completed turn.
- **"What went wrong?"** The blind held-out set caught a real bug. On a confirmation turn whose text
  changed the request, code forced the old proposal to execute, which refunded the wrong item. The
  agent had read the change correctly. Fixed: a different proposal from the agent now withdraws
  the confirmation. In about 300 other stored confirmation turns this never fired. The next bench
  showed the fix was incomplete: B's agent did not read the change and executed, and code checked
  only IDs, so code now reads product names too. The same bench showed code blocking a correct
  agent ("What about a 30 then?" refused because of an older 34), also fixed. Now v3 is no longer
  blind, and we say so.
- **"Is the baseline weak on purpose?"** No. B's quality is within one case of F's on every suite. F is faster
  because of five measured changes, each tested by putting it back alone. The prompt-cache change
  showed no measurable effect (inside the A/A noise), and we say that too.
- **"Why Qwen3.5-4B and llama.cpp?"** Fast enough on a laptop GPU, reliable structured output with
  JSON-schema-constrained decoding, fully offline, one pinned build. Q4_K_M kept quality and saved
  about 1.7 GB of GPU memory against Q8_0.
- **"How do you know the oracle is right?"** It was written separately from `desk/`, and the labels
  were checked by hand. Its own bugs were found by review and fixed (orders not filtered by customer,
  replay reading). The desk and the oracle agreeing is evidence, not proof.
- **"What if the model is wrong?"** A block becomes a question; an error changes nothing. A wrong
  commit is only possible if the agent's wrong reading also passes every fact check. That is why
  the citation gate exists.
