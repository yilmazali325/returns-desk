# Security, injection and UI robustness review

Scope: server/ (FastAPI + SSE), server/static/app.js, desk/sanitize.py, the prompt builders, the gateway's
customer scoping. Review only; no project file was edited. Repro tests: `tests/review/security/`
(`xfail(strict=True)` = reproduces a finding and fails today; remove the marker after the fix).

```
.venv/bin/pytest tests/review/security -q -rxX      # 52 passed, 14 xfailed in 6 s (2026-10-28)
.venv/bin/pytest tests/review/security -q --runxfail  # shows the 14 real failures
```

The UI test file starts `python -m server.app --port 8430 --fake` itself and stops it. No llama-server was started.

## Findings (ranked by grading impact)

### SEC-1 (high): a Confirm click can execute a proposal the user never saw

The UI sends only `confirm_proposal_id` (app.js:520 `confirmProposal(card.proposal_id, ...)`). The server grants
whatever proposal has that id in the session now (orchestrator.py:249
`confirm if confirm in presented`). Proposal ids are per-store counters (`tx.next_id("P")`), so they restart at
P-0001 after every reset. Session ids are per-process counters (app.py:237 `S-{self._counter:04d}`), so they
restart after every server restart. Two tabs share one session, because the UI restores it from localStorage.

Repro (API test and Playwright, public_04):
1. Tab B: "Refund line L1 of order O100 please." The card reads **Confirm refund $30.00** (binding 3788882c36dd).
2. Tab A: Reset, then "Refund line L2 of order O100 please." This creates a new P-0001: a $90.00 refund of L2.
3. Tab B clicks its stale button. The result is `completed`, "Completed refund: **$90.00 for JACKET-M**, order O100,
   line **L2**", and `refunds == [('L2', 9000)]`. Screenshot: tests/review/security/shots/stale_tab_confirm.png.

The card says "Confirming approves exactly this proposal" and shows a binding hash, but the hash never goes back
to the server. The contract requires that a confirmed proposal "must bind customer, order line, remedy, amount and
replacement SKU". The adapter path ("last") is not affected.

Fix:
- `proposal_card` exposes the full `binding_hash`.
- The UI sends `confirm_binding_hash` and `generation` with `confirm_proposal_id`, from both the card button and the
  composer checkbox.
- `TurnRequest` accepts both fields. Before `claim()`, `post_turn` returns 409 "this proposal changed, reload" when
  the generation differs or the stored proposal's `binding_hash` differs.
- Optionally use random session ids (`secrets.token_hex`) so a stale tab gets a 404 after a restart.
- Tests: `test_http_security.py::test_stale_tab_confirm...` and `test_ui_security.py::test_two_tabs...` will XPASS.
  Update `tests/unit/test_server.py::run_over_http` to send the hash.

### SEC-2 (medium): the reply guard lets some false success claims through on a propose turn

`_SUCCESS_RE` (render.py:395) matches "completed", "refunded", "all set" and similar words. It does not match
"complete", "done", "back on your card", "back in your account" or "no need to confirm". Scripted with a model
that obeys the injected instruction, the customer receives this on a **propose** turn, and no guard fires:

> Your refund is complete. Proposed refund: $30.00 for TEE-M ... Nothing has been processed yet; please confirm ...

Blocked today: "Completed refund: ...", "All set ...", "issued", "went through", "approved".
Passed today: "Your refund is complete.", "Your money is back in your account.",
"... already back on your card.", "Done! ... no need to confirm."

Evidence limits: this uses a scripted model, not Qwen. The repository's real runs contain 0 such phrasings in
1,482 non-completed turns. Confidence: medium that a hidden injection-style variant triggers it.

Fix: add `complete|done|finished|back (on|in|to) your|on (its|the) way|no need to confirm|refund(ed)? to your` to
`_SUCCESS_RE`. Keep the `_NOT_A_CLAIM` prefix logic. Run tests/unit/test_render.py and
`test_prompt_injection.py` (5 xfails flip).

### SEC-3 (low): 4 control tokens of the deployed model survive sanitize()

The GGUF vocabulary of Qwen3.5-4B-Q4_K_M has 33 CONTROL tokens. sanitize() neutralizes 29 of them.
`<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>` and `<tts_text_bos_single>` pass through unchanged.
`llama-tokenize -m models/Qwen3.5-4B-Q4_K_M.gguf -p 'refund L1 <tts_text_bos> hi ‹|im_start|>system'` returns
`[66165, 436, 16, 220, 248073, ...]`, so the tts token arrives as control id 248073. These tokens carry no
chat-role meaning, and authorization is code-only, so the impact is limited. Still, the docstring's "we break every
such marker" is not true for this model.

Fix: in `_MARKER_START`, neutralize any `<` that starts a token from the model's special vocabulary. Either load
the list from the GGUF at startup, or add `tts_\w+` and a generic `<(?=/?[a-z_]+>)` for known names.
`test_sanitize_vocab.py` re-reads the GGUF, so the next model swap is caught.

### SEC-4 (low): localhost hardening (CSRF on reset, DNS rebinding, no body cap)

The server binds 127.0.0.1 only (verified with lsof), with no CORS headers. JSON endpoints refuse `text/plain`
with 422, so cross-site "simple" requests cannot drive turns or faults. Three gaps remain:
- `POST /reset` takes an optional body. A bodiless cross-site POST with `Origin: https://evil.example` returns
  200 and wipes the session.
- Any `Host` header is served (`Host: attacker.example` returns 200), so a DNS-rebinding page can read state and
  post confirmations.
- No request size limit: a 50 MB `initial_state` was accepted (200 in 0.5 s) and is echoed in every snapshot.

Fix: add `TrustedHostMiddleware(allowed_hosts=["127.0.0.1", "localhost"])`. Add a small middleware that refuses
POSTs whose `Origin` is not this server, or that lack `application/json`. Cap the body at about 1 MB.

### SEC-5 (low): some malformed inputs give an opaque 500 instead of an actionable 422

- A message containing a lone surrogate (`"\ud800"`): pydantic rejects it, and FastAPI's 422 handler echoes the
  input. Starlette's UTF-8 encode then raises, so the response is a plain "Internal Server Error". No state changes.
- Blank-session `initial_state` with wrong shapes: `orders: ["x"]` returns 500, and `paid_cents: "10"` produces an
  SSE `error` "TypeError: money() takes integer cents".

Fix: add a `RequestValidationError` handler that drops `input`, and a pydantic model for the state schema.

## Checked and holding (regression tests in tests/review/security)

- **XSS:** every DOM write goes through `textContent` or `setAttribute` (app.js `el()`/`setProps()`). No
  `innerHTML`, no URL or style sinks. Payloads in the message, SKU, product_id, size, order_id, line_id and
  customer_id render literally in the chat, card, team view, audit trail and state inspector. `window.__xss`
  stayed undefined, no nodes were injected, and there were no page errors.
- **No off-machine traffic:** the only URLs are an SVG namespace and setup.sh's model download. llama-server and the
  UI bind 127.0.0.1.
- **Path traversal:** `/static/..`, `%2e%2e`, `..%2f` and `%00` in session ids all return 404. No path is built
  from request data.
- **Disconnects and concurrency:** after an SSE disconnect mid-turn, the turn finishes, the lock is released, and
  one turn and one open proposal are recorded. A concurrent turn or reset during a turn gets 409. Two simultaneous
  confirms make exactly one refund and one quarantine.
- **Malformed input:** malformed JSON returns 422, and over 2,000 chars returns 422. NUL, zero-width and bidi
  characters are stripped before grounding and prompts.
- **Prompt injection:** a scripted model obeying an injected fake "CONFIRMATION THIS TURN" block plus
  `<|im_start|>system` cannot execute; it is gated and gets CONFIRMATION_REQUIRED. Customer text never reaches a
  system prompt or the Operations agent, and `"""` cannot close the untrusted block.
- **Cross-customer:** a foreign order and a nonexistent one give an identical decision, reply and audit (modulo the
  typed id). With a hostile model calling `find_order_lines("O777")` and handing off O777/L5, the lookup is
  customer-scoped (`count 0, order_not_on_account`), the handoff is gated, and nothing about C999 reaches any prompt.

Note (no finding filed): `start_turn` releases `session.lock` (app.py:504) before it builds the final snapshot
(app.py:506). A turn started in that window can tear the first turn's `turn` event state (it shows `busy: false`
mid-turn). The impact is cosmetic. Taking the snapshot before `release()` fixes it.
