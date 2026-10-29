# 5-minute live demo script

The goal is to show a successful transaction and a failure recovery, with the two agents and the
code checks visible the whole time.

## Before the session (10 minutes earlier)

1. Close Chrome tabs with video or graphics. They slow the GPU, so first answers get slower.
2. Start the desk and wait for "Returns Desk is ready":

   ```bash
   scripts/run.sh
   ```

3. Open http://127.0.0.1:8000 and switch to **Team (live)**.
4. Send one throwaway turn (any scenario, then **Reset state**) so Metal is compiled and the prompt
   cache is warm.
5. Keep a terminal ready in the repo for the adapter at the end.
6. Plan B, if the model misbehaves:
   - restart the UI on the deterministic double with `scripts/run.sh --fake`, and say so;
   - or show the shots in `var/screenshots/` (`final_team_refund.png`, `final_team_recovery.png`).

## 0:00–0:30 — the idea

> "Two local agents run this desk. Resolution talks to the customer; Operations runs the back
> office. Code checks every choice they make. Agents decide; code can only say no. The model is
> Qwen3.5-4B on llama.cpp on this laptop, and the network is off."

Point at the legend and the LLM / CODE labels on the cards.

## 0:30–2:15 — a successful exchange (public_02)

1. **Scenario** → `public_02_exchange` → **Reset state**. In **Script**, press **Fill** on the first
   message, then **Send**.
2. Narrate the Team view as it moves:
   - Resolution reads the request and hands off an exchange with the size the customer asked for.
   - Grounding checks that the line is on this customer's account and that the size matches their
     words.
   - Operations lists the options and creates the proposal.
   - Policy checks eligibility (delivered, unopened, not final sale, inside 30 days of the fixture
     date), same product, same price and stock.
3. Show the **proposal card**: original and replacement SKU, amount $0.00, and **Confirm**.
   > "Nothing has changed yet. Saying 'yes' in chat would not execute it either; only the explicit
   > confirm control can, and only for this exact proposal."
4. Tick **Confirm**, press **Send**:
   - Operations executes;
   - the gateway re-checks eligibility and stock, then commits in one transaction (return record,
     quarantine, stock decrement, idempotency entry);
   - the decision is **completed**.
5. Open **Audit trail** for a few seconds: every tool call, its arguments and results, who called it.

## 2:15–4:00 — failure recovery: a timeout after the commit (public_06)

1. **Scenario** → `public_06_timeout_replay` → **Reset state** → **Fill** → **Send**. A refund
   proposal appears.
2. The scenario has the timeout armed: the commit succeeds, but the response is lost. You can arm it
   yourself with **Failure lab** → **Arm timeout after next commit**. Confirm and **Send**.
3. Narrate:
   - the gateway card turns red with the timeout;
   - Operations is now only allowed to ask for the request status;
   - the Recovery card reads the store, finds the committed refund, and reports **completed**.
   > "It never claims success it can't prove, and it never retries the money movement blindly."
4. Send the third scripted message (the customer confirms again). The decision is **completed** as a
   **replay**: the same transaction id, zero new effects. Point at the unchanged refund list.

Optional, if there is time: in `public_05_stale_stock` the scenario drops the replacement's stock
to 0 just before the confirmation. The re-check at execute refuses, the decision is **deny**, and
nothing changes. You can do the same live in any exchange with **Failure lab** → **Set stock now**.

## 4:00–5:00 — the same desk, headless

```bash
python3 adapter.py --input candidate_assets/public_cases.json --output results.json
```

It takes about 35 s for all six cases, with a fresh llama-server.

```bash
python3 candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output results.json
```

> "The adapter is the same code path the UI uses. Every result carries hardware, model hash,
> engine version, per-case latency, memory and token metrics."

Hand over to the slides.
