/* Returns Desk UI: vanilla JS, no libraries, no network beyond this server (works offline).
 *
 * Data flow, one direction only:
 *   GET /state or the SSE `turn` event   -> app.session (the server's snapshot)
 *   SSE `audit` events during a turn      -> app.live.entries (appended as they are recorded)
 *   app.session + app.live                -> render*() functions, which rebuild their view from data
 *
 * The UI never decides anything. Decisions, amounts, proposal facts and confirm labels come from
 * the server (deterministic desk code); this file only formats and arranges them. Success is shown
 * only when the server's decision is "completed".
 *
 * Sections: 1 helpers · 2 API · 3 state · 4 wording · 5 header · 6 rail · 7 conversation
 *           8 team view (graph, event mapping, player) · 9 audit trail · 10 state inspector
 *           11 actions · 12 start-up
 */
(() => {
  "use strict";

  // ============================================================================================
  // 1. Small helpers

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  /** el("div", {class: "x", text: "hi", "data-a": 1, onclick: fn}, child, ...) */
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    setProps(node, props);
    append(node, children);
    return node;
  }

  function svgEl(tag, props = {}, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    setProps(node, props);
    append(node, children);
    return node;
  }

  function setProps(node, props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null) continue;
      const stateAttr = key.startsWith("data-") || key.startsWith("aria-");
      if (key === "text") node.textContent = String(value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (typeof value === "boolean" && stateAttr) node.setAttribute(key, String(value)); // data-x="true"/"false"
      else node.setAttribute(key === "className" ? "class" : key, String(value));
    }
  }

  function append(node, children) {
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  /** Integer cents -> "$12.50". Money is never a float anywhere in this app. */
  function money(cents) {
    if (!Number.isInteger(cents)) return "–";
    const sign = cents < 0 ? "-" : "";
    const abs = Math.abs(cents);
    return `${sign}$${Math.floor(abs / 100).toLocaleString("en-US")}.${String(abs % 100).padStart(2, "0")}`;
  }

  const fmtMs = (ms) => (ms === null || ms === undefined ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
  const fmtMb = (mb) => (mb === null || mb === undefined ? "n/a" : mb >= 1024 ? `${(mb / 1024).toFixed(2)} GiB` : `${Math.round(mb)} MiB`);
  const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;
  const lineRef = (order, line) => `${order}/${line}`;
  /** At most `max` characters, cut at a word boundary when there is one, with an ellipsis. */
  function short(text, max) {
    if (text.length <= max) return text;
    const cut = text.slice(0, max - 1);
    const space = cut.lastIndexOf(" ");
    return `${(space > max / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
  }
  const json = (value) => JSON.stringify(value, null, 2);

  /** localStorage can throw (private mode, blocked storage): it only holds conveniences. */
  const prefs = {
    get(key, fallback) {
      try {
        const raw = window.localStorage.getItem(`returns-desk.${key}`);
        return raw === null ? fallback : JSON.parse(raw);
      } catch (_) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(`returns-desk.${key}`, JSON.stringify(value));
      } catch (_) {
        /* not essential */
      }
    },
  };

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // ============================================================================================
  // 2. API

  class HttpError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function errorFrom(response) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    } catch (_) {
      /* keep the status text */
    }
    return new HttpError(response.status, detail);
  }

  async function request(method, path, body) {
    const response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) throw await errorFrom(response);
    return response.json();
  }

  /** POST a turn and call onEvent(type, data) for every Server-Sent Event as it arrives. */
  async function streamTurn(sessionId, body, onEvent) {
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw await errorFrom(response);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const event = parseSse(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        if (event) onEvent(event.type, event.data);
      }
    }
  }

  function parseSse(block) {
    let type = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // keep-alive comment
      if (line.startsWith("event:")) type = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    return data.length ? { type, data: JSON.parse(data.join("\n")) } : null;
  }

  // ============================================================================================
  // 3. Application state

  const app = {
    meta: { reason_text: {}, code_class: {}, window_days: null },
    health: null,
    telemetry: null,
    scenarios: [],
    session: null, // the server snapshot (GET /state)
    live: null, // {turn, message, confirmId, entries: [], finished, output}
    busy: false,
    notice: null, // {kind, title, text} shown above the conversation (HTTP or network problems)
    llmFailedAt: null, // a turn in this page saw LLM_ERROR
    // The composer's confirmation box: the proposal it was ticked for (a tick never moves to a newer
    // proposal) and the proposal a scripted confirm turn was filled for (script fidelity: the
    // contract's "last" may name a proposal that is no longer open, e.g. a replay after completion).
    confirm: { tickedFor: null, scripted: null },
  };

  const scenarioById = (id) => app.scenarios.find((s) => s.case_id === id) || null;

  /** All audit entries known now: the snapshot plus anything streamed since. */
  function allEntries() {
    const bySeq = new Map();
    for (const entry of app.session ? app.session.action_audit : []) bySeq.set(entry.seq, entry);
    for (const entry of app.live ? app.live.entries : []) bySeq.set(entry.seq, entry);
    return [...bySeq.values()].sort((a, b) => a.seq - b.seq);
  }

  const entriesOfTurn = (turn) => allEntries().filter((e) => e.turn === turn);

  /** Turn records (message, confirm id, output) plus the running one, if any. */
  function turnRecords() {
    const records = app.session ? [...app.session.turns] : [];
    if (app.live && !app.live.finished) {
      records.push({ turn: app.live.turn, message: app.live.message, confirm_proposal_id: app.live.confirmId, output: null, live: true });
    }
    return records;
  }

  /** The proposal the contract's confirm_proposal "last" would resolve to now (whatever its status). */
  function lastPresentedProposalId() {
    const turns = app.session ? app.session.turns : [];
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      if (turns[i].output && turns[i].output.proposal) return turns[i].output.proposal.proposal_id;
    }
    return null;
  }

  /** What the composer could attach: the last presented proposal, its current card and status. */
  function confirmOffer() {
    const pid = lastPresentedProposalId();
    if (!pid) return null;
    const card = app.session.cards[pid] || null;
    return { pid, card, status: card ? card.status : "unknown" };
  }

  /** The fields that bind an approval to what this page displayed (the server answers 409 otherwise). */
  function confirmationFields(card) {
    return { confirm_proposal_id: card.proposal_id, confirm_binding_hash: card.binding_hash || null, generation: app.session.generation };
  }

  /** The newest turn with audit entries (setup is turn 0), or null before anything happened. */
  function latestTurn() {
    const turns = allEntries().map((e) => e.turn);
    return turns.length ? Math.max(...turns) : null;
  }

  // ============================================================================================
  // 4. Wording (codes in plain language, compact tool summaries)

  const reasonText = (code) => app.meta.reason_text[code] || code;
  const codeClass = (code) => app.meta.code_class[code] || "failure";

  const AGENT_LABEL = { resolution: "Resolution", operations: "Operations", orchestrator: "Orchestrator", external: "External" };

  function toolSummary(tool, args = {}) {
    if (tool === "handoff_to_operations") {
      const where = args.order_id ? ` ${lineRef(args.order_id, args.line_id)}` : "";
      const size = args.intent === "propose_exchange" ? ` → ${args.requested_size || "?"}` : "";
      return `handoff_to_operations · ${args.intent}${where}${size}`;
    }
    if (tool === "reply_to_customer") return "reply_to_customer";
    if (tool === "find_order_lines") return `find_order_lines(${args.order_ref || "ALL"})`;
    if (tool === "create_proposal") return `create_proposal(${[args.remedy, args.replacement_sku].filter(Boolean).join(", ")})`;
    if (tool === "report_to_resolution") return `report_to_resolution(${args.outcome || ""})`;
    return `${tool}()`;
  }

  function effectChips(effects) {
    const chips = [];
    for (const fx of effects || []) {
      if (fx.type === "record" && fx.collection === "refunds") chips.push({ kind: "money", text: `refund ${fx.id} ${money(fx.amount_cents)}` });
      else if (fx.type === "record" && fx.collection === "quarantine") chips.push({ kind: "quarantine", text: `quarantine ${fx.id}` });
      else if (fx.type === "record") chips.push({ kind: "record", text: `${fx.collection.replace(/s$/, "")} ${fx.id}` });
      else if (fx.type === "stock") chips.push({ kind: "stock", text: `${fx.sku} stock ${fx.old_stock}→${fx.new_stock}` });
      else if (fx.type === "line_returned") chips.push({ kind: "record", text: `${lineRef(fx.order_id, fx.line_id)} returned` });
      else if (fx.type === "idempotency") chips.push({ kind: "record", text: "idempotency record" });
    }
    return chips;
  }

  // ============================================================================================
  // 5. Header: model chip, live telemetry, LLM status, banners

  function renderHealth() {
    const h = app.health;
    const dot = $("llm-dot");
    if (!h) {
      dot.dataset.state = "down";
      dot.setAttribute("aria-label", "Desk server not reachable");
      $("model-text").textContent = "desk server not reachable";
      return;
    }
    dot.dataset.state = h.llm === "ok" ? "ok" : "down";
    dot.setAttribute("aria-label", h.llm === "ok" ? "Model answering" : "Model not reachable");
    // The chip stays one short line (the header must not wrap at 1280 px); the full engine string is its tooltip.
    $("model-text").textContent = [h.model, h.quantization, h.engine_short || h.engine, `profile ${h.profile}`].filter(Boolean).join(" · ");
    $("model-chip").title = [h.model, h.quantization, h.engine, `profile ${h.profile}`].filter(Boolean).join(" · ");
    $("legend-llm").textContent = h.mode === "fake-policy"
      ? "LLM agent: fake-policy test double (no model loaded)"
      : `LLM agent: ${h.model} ${h.quantization}, local llama.cpp`;
    renderBanners();
  }

  function renderTelemetry() {
    const t = app.telemetry;
    if (!t) return;
    const step = t.last_step;
    const who = step ? AGENT_LABEL[step.agent] || step.agent : "";
    $("tm-step").textContent = !step ? "–" : step.error ? `failed: ${step.error} (${who})` : `${fmtMs(step.latency_ms)} (${who})`;
    $("tm-toks").textContent = step && step.decode_tok_s ? `${step.decode_tok_s} tok/s` : t.last_turn && t.last_turn.decode_tok_s ? `${t.last_turn.decode_tok_s} tok/s` : "–";
    $("tm-mem").textContent = fmtMb(t.memory && t.memory.footprint_mb);
    $("tm-gpu").textContent = fmtMb(t.memory && t.memory.gpu_unified_mb);
    $("tm-mem").title = t.memory ? `server ${fmtMb(t.memory.server_footprint_mb)} + llama-server ${fmtMb(t.memory.llama_footprint_mb)} (phys_footprint)` : "";
    $("tm-gpu").title = t.memory ? t.memory.notes : "";
  }

  function commandRow(label, command) {
    const button = el("button", { type: "button", class: "btn btn-small", text: "Copy" });
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(command);
        button.textContent = "Copied";
      } catch (_) {
        button.textContent = "Select and copy";
      }
    });
    return el("div", { class: "cmd" }, el("span", { text: label }), el("code", { text: command, "data-testid": "start-command" }), button);
  }

  function renderBanners() {
    const area = clear($("banner-area"));
    const h = app.health;
    const down = h && h.mode !== "fake-policy" && h.llm !== "ok";
    if (h && (down || app.llmFailedAt !== null)) {
      const check = el("button", { type: "button", class: "btn btn-small", text: "Check again", onclick: refreshHealth });
      area.append(el("section", { class: "banner", role: "alert", "data-testid": "model-down" },
        el("h2", { text: "The local model is not answering" }),
        el("p", { text: `No reply from llama-server at ${h.llama_url}. Nothing was changed: without the model the desk cannot act. Start it from the project folder:` }),
        el("div", { class: "banner-cmds" }, commandRow("everything:", h.start.run_script), check),
        commandRow("or only the model:", h.start.llama_server)));
    }
    if (app.notice) {
      area.append(el("section", { class: `banner ${app.notice.kind || ""}`, role: "alert", "data-testid": "notice" },
        el("h2", { text: app.notice.title }), el("p", { text: app.notice.text }),
        el("div", {}, el("button", { type: "button", class: "btn btn-small", text: "Dismiss", onclick: () => { app.notice = null; renderBanners(); } }))));
    }
  }

  function showNotice(title, text, kind = "") {
    app.notice = { title, text, kind };
    renderBanners();
  }

  async function refreshHealth() {
    try {
      app.health = await request("GET", "/api/health");
      if (app.health.llm === "ok") app.llmFailedAt = null;
    } catch (_) {
      app.health = null;
    }
    renderHealth();
  }

  async function refreshTelemetry() {
    try {
      app.telemetry = await request("GET", "/api/telemetry");
      renderTelemetry();
    } catch (_) {
      /* the health check reports an unreachable server */
    }
  }

  // ============================================================================================
  // 6. Left rail: scenario picker, script, failure lab

  const BLANK = "__blank__";

  function renderScenarioPicker() {
    const select = clear($("scenario-select"));
    for (const s of app.scenarios) select.append(el("option", { value: s.case_id, text: s.title }));
    select.append(el("option", { value: BLANK, text: "Blank conversation (choose state and customer)" }));
    const stateSelect = clear($("blank-state-select"));
    for (const s of app.scenarios) stateSelect.append(el("option", { value: s.case_id, text: `state of ${s.title}` }));
  }

  function renderCustomerOptions() {
    const base = scenarioById($("blank-state-select").value);
    const select = clear($("customer-select"));
    for (const id of base ? base.customers : []) select.append(el("option", { value: id, text: id }));
    select.append(el("option", { value: "GUEST-1", text: "GUEST-1 (no orders)" }));
  }

  function syncPickerToSession() {
    const s = app.session;
    if (!s) return;
    const blank = !s.case_id;
    $("scenario-select").value = blank ? BLANK : s.case_id;
    $("blank-options").hidden = !blank;
  }

  /** Session id, customer, request and business date (renderAll runs only once a session exists). */
  function renderSessionFacts() {
    const facts = clear($("session-facts"));
    const s = app.session;
    const notes = $("scenario-notes");
    const add = (k, v) => facts.append(el("dt", { text: k }), el("dd", { text: v }));
    add("session", s.session_id);
    add("customer", s.customer_id);
    add("request", s.request_id);
    add("business date", s.initial_state.as_of || "–");
    const scenario = scenarioById(s.case_id);
    notes.textContent = scenario
      ? `${plural(scenario.turns.length, "scripted turn")}${scenario.fault_texts.length ? ` · fault: ${scenario.fault_texts.join("; ")}` : ""}`
      : "Blank conversation: type any request for this customer.";
  }

  function renderScript() {
    const list = clear($("script-list"));
    const scenario = app.session ? scenarioById(app.session.case_id) : null;
    if (!scenario) {
      list.append(el("li", {}, el("span", { class: "hint", text: "No script for a blank conversation." })));
      return;
    }
    const done = app.session.turns.length;
    scenario.turns.forEach((turn, index) => {
      const confirm = turn.confirm_proposal === "last";
      const label = `T${index + 1}${confirm ? " · confirms the last proposal" : ""}`;
      const fill = el("button", { type: "button", class: "btn btn-small", text: "Fill", "data-testid": `fill-${index + 1}`,
        "aria-label": `Fill scripted turn ${index + 1}`, onclick: () => fillScript(index) });
      fill.disabled = app.busy;
      list.append(el("li", { "data-next": index === done, "data-done": index < done },
        el("div", {}, el("span", { class: "script-tag", text: label }), el("div", { text: turn.message })), fill));
    });
  }

  function renderFailureLab() {
    const s = app.session;
    const skuSelect = $("stock-sku");
    const previous = skuSelect.value;
    clear(skuSelect);
    for (const p of s ? s.final_state.products : []) skuSelect.append(el("option", { value: p.sku, text: `${p.sku} (stock ${p.stock})` }));
    if (previous) skuSelect.value = previous;
    const status = $("fault-status");
    const faults = s ? s.faults : null;
    if (faults && faults.timeout_armed) {
      status.dataset.state = "armed";
      status.textContent = "Armed: the next commit will persist, then its answer is lost.";
    } else if (faults && faults.timeout_fired) {
      status.dataset.state = "fired";
      status.textContent = "Fired: a commit timed out after it persisted. Arm again to repeat.";
    } else {
      status.dataset.state = "idle";
      status.textContent = "Not armed.";
    }
    for (const id of ["stock-apply", "arm-timeout"]) {
      const node = document.querySelector(`[data-testid="${id}"]`);
      if (node) node.disabled = app.busy || !s;
    }
  }

  // ============================================================================================
  // 7. Conversation: messages, decision pills, proposal card, execution status, actionable errors

  function renderChat() {
    const log = clear($("chat-log"));
    const records = turnRecords();
    if (!records.length) {
      log.append(el("p", { class: "empty-chat", text: "Pick a scenario, fill its first scripted message (or type one), and press Send." }));
    }
    const latestProposalTurn = records.reduce((acc, r) => (r.output && r.output.proposal ? r.turn : acc), null);
    for (const record of records) {
      log.append(customerMessage(record));
      log.append(deskMessage(record, record.turn === latestProposalTurn));
    }
    log.scrollTop = log.scrollHeight;
  }

  function customerMessage(record) {
    const meta = el("div", { class: "msg-meta" }, el("span", { text: `Customer · turn ${record.turn}` }));
    if (record.confirm_proposal_id) {
      meta.append(el("span", { class: "tag tag-confirm", text: `explicit confirmation of ${record.confirm_proposal_id}` }));
    }
    return el("div", { class: "msg msg-customer", "data-turn": record.turn }, meta, el("div", { class: "bubble", text: record.message }));
  }

  function deskMessage(record, latestProposal) {
    const entries = entriesOfTurn(record.turn);
    const output = record.output;
    const decision = output ? output.decision : "working";
    const pill = el("span", { class: "pill", "data-decision": decision, "data-testid": "decision", text: output ? decision : "working" });
    const meta = el("div", { class: "msg-meta" }, el("span", { text: "Desk" }), pill);
    if (output && record.latency_ms !== undefined && record.latency_ms !== null) meta.append(el("span", { text: fmtMs(record.latency_ms) }));
    const node = el("div", { class: "msg msg-desk", "data-turn": record.turn, "data-decision": decision, "data-testid": "desk-message" }, meta);
    if (!output) {
      node.append(el("div", { class: "bubble" }, el("span", { class: "working-dots", text: liveActivity() })));
    } else {
      node.append(el("div", { class: "bubble", "data-testid": "response", text: output.response }));
    }
    if (record.confirm_proposal_id || (output && output.confirmed_proposal_id)) node.append(executionPanel(entries, output));
    if (output && output.decision === "propose") {
      const card = app.session.cards[output.proposal.proposal_id];
      node.append(proposalCard(card || output.proposal, latestProposal && !app.busy));
    }
    if (output && output.decision === "deny") node.append(denyNotice(entries));
    if (output && output.decision === "error") node.append(errorNotice(entries));
    if (output && output.decision === "clarify") {
      const gate = entries.find((e) => e.kind === "gate" && e.error);
      if (gate) node.append(el("div", { class: "notice warn" }, el("strong", { text: "Needs one more detail" }), el("span", { text: reasonText(gate.error.code) })));
    }
    return node;
  }

  function liveActivity() {
    const entries = app.live ? app.live.entries : [];
    const last = entries[entries.length - 1];
    if (!last) return "The desk is reading the message";
    if (last.kind === "llm_step") return `${AGENT_LABEL[last.agent]} agent chose ${toolSummary(last.tool, last.arguments)}`;
    return `${AGENT_LABEL[last.agent] || last.agent}: ${last.tool}`;
  }

  function proposalCard(card, interactive) {
    const refund = card.remedy === "refund";
    const status = card.status || "open";
    const head = el("div", { class: "card-head" },
      el("span", { class: "card-title", text: `Proposal ${card.proposal_id} · ${refund ? "Refund" : "Size exchange"}` }),
      el("span", { class: "card-status", "data-status": status, "data-testid": "proposal-status", text: status }));
    const item = `${card.product_id || ""} ${card.size ? `size ${card.size}` : ""}`.trim();
    let main;
    if (refund) {
      main = el("div", {},
        el("div", { class: "amount", "data-testid": "proposal-amount", text: card.amount_text || money(card.amount_cents) }),
        el("div", { class: "amount-sub", text: `refund of the amount paid for ${card.original_sku} (${item}) · no taxes, fees or shipping` }));
    } else {
      const stock = card.replacement_stock;
      main = el("div", { class: "exchange-line" },
        el("div", { class: "sku-box" }, el("small", { text: "you return" }), el("strong", { text: card.original_sku }), el("small", { text: item })),
        el("span", { "aria-hidden": "true", text: "→" }),
        el("div", { class: "sku-box" }, el("small", { text: "you receive" }), el("strong", { "data-testid": "proposal-replacement", text: card.replacement_sku }),
          el("small", { "data-testid": "replacement-stock", text: `size ${card.replacement_size || "?"} · ${stock === null || stock === undefined ? "stock unknown" : `${stock} in stock now`}` })),
        el("div", { class: "amount-sub", text: `no charge (${money(card.amount_cents)})` }));
    }
    const facts = el("dl", { class: "facts" });
    const fact = (k, v) => facts.append(el("div", {}, el("dt", { text: k }), el("dd", { text: v })));
    fact("Order / line", lineRef(card.order_id, card.line_id));
    fact("Delivered on", card.delivered_on || "–");
    fact("Since delivery", card.days_since_delivery === null || card.days_since_delivery === undefined ? "–" : `day ${card.days_since_delivery} of ${card.window_days}`);
    fact("Business date", card.as_of || "–");
    const checks = el("ul", { class: "check-list", "aria-label": "Eligibility checks" },
      el("li", { "data-ok": card.days_since_delivery <= card.window_days, text: `within the ${card.window_days ?? app.meta.window_days}-day window` }),
      el("li", { "data-ok": card.unopened === true, text: card.unopened === true ? "unopened" : "opened" }),
      el("li", { "data-ok": card.final_sale === false, text: card.final_sale === false ? "not final sale" : "final sale" }));
    const node = el("div", { class: "card", "data-testid": "proposal-card", "data-proposal": card.proposal_id }, head, main, facts, checks,
      el("div", { class: "hash", text: `binding ${card.binding_hash_short || ""} · binds customer, line, remedy, amount and replacement` }));
    if (status === "open" && interactive) {
      const confirm = el("button", { type: "button", class: "btn btn-ok", "data-testid": "confirm", text: card.confirm_label });
      const later = el("button", { type: "button", class: "btn", "data-testid": "not-now", text: "Not now" });
      confirm.addEventListener("click", () => confirmProposal(card, [confirm, later]));
      later.addEventListener("click", () => notNow([confirm, later]));
      node.append(el("div", { class: "card-actions" }, confirm, later),
        el("p", { class: "card-note", text: "Nothing moves until you confirm. Confirming approves exactly this proposal; the desk re-checks eligibility and stock before committing." }));
    }
    return node;
  }

  /** Execution status of a confirmation turn, from its audit entries and (when done) its decision. */
  function executionSteps(entries, output) {
    const find = (pred) => entries.find(pred);
    const grant = find((e) => e.kind === "grant");
    const withheld = find((e) => (e.kind === "guard" && e.tool === "grant_withheld") || e.kind === "confirmation");
    const preamble = find((e) => e.kind === "route" && e.tool === "idempotency_preamble");
    const executeAsked = find((e) => e.kind === "llm_step" && e.tool === "execute_proposal");
    const execute = find((e) => e.kind === "tool" && e.tool === "execute_proposal");
    const timedOut = execute && execute.error && execute.error.code === "TIMEOUT";
    const status = entries.filter((e) => e.kind === "tool" && e.tool === "get_request_status").pop();
    const reconciled = timedOut && status && status.result && status.result.status === "completed";
    const completed = output && output.decision === "completed";
    const steps = [{ id: "proposed", label: "Proposed", state: "done" }];
    steps.push({ id: "confirmed", label: withheld && !grant ? "Confirmation withheld" : "Confirmed", state: grant ? "done" : withheld ? "failed" : output ? "failed" : "active" });
    const branch = { kind: null };
    if (preamble && preamble.result && preamble.result.replay) {
      steps.push({ id: "replay", label: "Already completed: replay, no new effects", state: "recovered" });
      branch.kind = "replay";
    } else if (grant) {
      const refused = execute && execute.error && !timedOut && codeClass(execute.error.code) !== "format";
      steps.push({ id: "revalidating", label: "Revalidating", state: refused ? "failed" : execute ? "done" : executeAsked ? "active" : output ? "skipped" : "pending" });
      const committed = execute && (execute.committed_effects || []).length > 0;
      steps.push({ id: "committing", label: "Committing", state: committed ? "done" : refused ? "skipped" : execute ? "failed" : "pending" });
      if (refused) branch.kind = "denied";
      if (timedOut) {
        steps.push({ id: "recovered", label: reconciled ? "Recovered: timeout after commit, reconciled" : "Timeout after commit: reconciling", state: reconciled ? "recovered" : output ? "failed" : "active" });
        branch.kind = "recovered";
      }
    }
    steps.push({ id: "completed", label: branch.kind === "denied" ? "Denied" : "Completed", state: completed ? "done" : branch.kind === "denied" ? "failed" : output ? "failed" : "pending" });
    return { steps, branch: branch.kind, execute, status, preamble, completed };
  }

  function executionPanel(entries, output) {
    const view = executionSteps(entries, output);
    const list = el("ol", { class: "stepper", "aria-label": "Execution status", "data-testid": "stepper" });
    for (const step of view.steps) {
      const mark = { done: "✓", failed: "✗", recovered: "↺", active: "…", skipped: "–", pending: "" }[step.state] || "";
      list.append(el("li", { "data-step": step.id, "data-state": step.state }, mark ? el("span", { "aria-hidden": "true", text: mark }) : null, step.label));
    }
    const panel = el("div", { class: "exec" }, list);
    if (view.branch === "recovered" && view.completed) {
      const tx = view.status && view.status.result && view.status.result.record ? view.status.result.record.transaction_id : "";
      panel.append(el("div", { class: "notice info", "data-testid": "recovered-notice" },
        el("strong", { text: "Recovered after a timeout" }),
        el("span", { text: `The commit persisted, but its answer was lost. The desk checked the records instead of retrying, found ${tx}, and did not commit a second time.` })));
    }
    if (view.branch === "replay" && view.completed) {
      panel.append(el("div", { class: "notice ok", "data-testid": "replay-notice" },
        el("strong", { text: "Already completed earlier" }),
        el("span", { text: `This repeats ${view.preamble.result.transaction_id}. No new refund, charge or stock change was made.` })));
    }
    return panel;
  }

  function refusalOf(entries) {
    const deciding = ["create_proposal", "check_proposal", "execute_proposal"];
    return entries.filter((e) => e.kind === "tool" && deciding.includes(e.tool) && e.error && codeClass(e.error.code) === "refusal").pop();
  }

  /** Reason codes of a deny that code decided on the facts, without a tool call (guard code_deny). */
  function codeDenyReasons(entries) {
    const guard = entries.filter((e) => e.kind === "guard" && e.tool === "code_deny").pop();
    return guard && guard.result ? guard.result.reasons || [] : [];
  }

  function denyNotice(entries) {
    const refusal = refusalOf(entries);
    const details = refusal && refusal.error.details ? refusal.error.details : {};
    const codes = details.reasons || (refusal ? [refusal.error.code] : codeDenyReasons(entries));
    const alternatives = details.alternatives || [];
    const revalidation = refusal && refusal.tool === "execute_proposal";
    const stockEvent = allEntries().filter((e) => e.kind === "fault" && e.tool === "stock_change" && e.result && e.result.applied).pop();
    const node = el("div", { class: "notice bad", "data-testid": "deny-notice" },
      el("strong", { text: revalidation ? "Revalidation failed before the commit: nothing was charged or moved" : "Not possible under the return rules" }));
    for (const code of codes) node.append(el("span", { text: reasonText(code) }));
    if (revalidation && stockEvent) node.append(el("span", { text: `External stock change: ${stockEvent.result.sku} ${stockEvent.result.old_stock} → ${stockEvent.result.new_stock}.` }));
    if (alternatives.length) {
      const target = denyTarget(refusal, entries);
      node.append(el("span", { text: "Valid alternatives now (fills the message box; a new proposal still needs your confirmation):" }),
        el("div", { class: "alt-list", "data-testid": "alternatives" }, alternatives.map((a) => el("button", {
          type: "button", class: "chip chip-action", "data-testid": "alternative", "data-sku": a.sku,
          title: `${a.sku} (size ${a.size})`, text: `Exchange for size ${a.size} instead`,
          onclick: () => fillComposer(alternativeMessage(target, a)) }))));
    }
    return node;
  }

  /** The order line a refusal was about: its proposal's line, else the line grounded this turn. */
  function denyTarget(refusal, entries) {
    const pid = refusal.error.details && refusal.error.details.proposal_id;
    const proposal = pid ? (app.session.proposals || []).find((p) => p.proposal_id === pid) : null;
    if (proposal) return { order_id: proposal.order_id, line_id: proposal.line_id };
    const grounded = entries.filter((e) => e.kind === "target").pop();
    return grounded ? grounded.arguments || null : null;
  }

  function alternativeMessage(target, alternative) {
    const where = target && target.order_id ? `line ${target.line_id} of order ${target.order_id}` : "the item";
    return `Please exchange ${where} for size ${alternative.size}.`;
  }

  function errorNotice(entries) {
    const llmError = entries.find((e) => e.kind === "llm_step" && e.error && e.error.code === "LLM_ERROR");
    const timeout = entries.find((e) => e.kind === "tool" && e.error && e.error.code === "TIMEOUT");
    if (llmError) {
      const node = el("div", { class: "notice bad", "data-testid": "error-notice" },
        el("strong", { text: "The local model did not answer, so nothing was changed" }),
        el("span", { text: llmError.error.message }));
      if (app.health && app.health.start) node.append(commandRow("start it:", app.health.start.run_script));
      return node;
    }
    if (timeout) {
      return el("div", { class: "notice warn", "data-testid": "error-notice" },
        el("strong", { text: "The commit timed out and is not verified yet" }),
        el("span", { text: "Do not send it again. The desk checks the request status before doing anything else, so a repeat can never refund twice." }));
    }
    return el("div", { class: "notice bad", "data-testid": "error-notice" }, el("strong", { text: "The desk could not finish this turn; nothing was changed." }));
  }

  // ============================================================================================
  // 8. Team view: the agents and specialists as a live graph (SPEC §10.1)

  // Goals are short enough to wrap into at most two lines at every layout width (no truncation).
  const NODES = [
    { id: "customer", name: "Customer", kind: "actor", badge: "PERSON", goal: "describes the request, confirms it", icon: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0" },
    { id: "resolution", name: "Resolution agent", kind: "llm", badge: "LLM", goal: "talks to the customer, asks if unclear", icon: "M4 5h16v11H9l-5 4z" },
    { id: "operations", name: "Operations agent", kind: "llm", badge: "LLM", goal: "checks the facts, prepares the remedy", icon: "M4 7h16v13H4zM9 7V4h6v3M4 12h16" },
    { id: "orchestrator", name: "Orchestrator", kind: "code", badge: "CODE", goal: "runs the turn and routes every handoff", icon: "M12 3v18M5 8l7-5 7 5M5 16l7 5 7-5" },
    { id: "grounding", name: "Grounding", kind: "code", badge: "CODE", goal: "named orders, lines, sizes must exist", icon: "M12 21s-7-6-7-11a7 7 0 0114 0c0 5-7 11-7 11zM12 12a2 2 0 100-4 2 2 0 000 4z" },
    { id: "policy", name: "Policy", kind: "code", badge: "CODE", goal: "in window, unopened, not final sale", icon: "M12 3v18M5 7h14M5 7l-3 7h6zM19 7l-3 7h6z" },
    { id: "approval", name: "Approval gate", kind: "code", badge: "CODE", goal: "nothing moves without a confirmation", icon: "M6 11h12v10H6zM9 11V7a3 3 0 016 0v4" },
    { id: "gateway", name: "Transaction gateway", kind: "code", badge: "CODE", goal: "one atomic commit, never applied twice", icon: "M4 8h13l-3-3M20 16H7l3 3" },
    { id: "recovery", name: "Recovery", kind: "code", badge: "CODE", goal: "after a timeout, checks records first", icon: "M20 12a8 8 0 11-3-6.3M20 4v5h-5" },
    { id: "replyguard", name: "Reply guard", kind: "code", badge: "CODE", goal: "replies never claim what records lack", icon: "M12 3l8 3v6c0 5-4 8-8 9-4-1-8-4-8-9V6z" },
    { id: "store", name: "Store (SQLite)", kind: "data", badge: "DATA", goal: "holds every committed effect", icon: "M5 6c0-2 14-2 14 0v12c0 2-14 2-14 0zM5 6c0 2 14 2 14 0M5 12c0 2 14 2 14 0" },
    { id: "warehouse", name: "Warehouse", kind: "actor", badge: "EXTERNAL", goal: "outside events, e.g. stock changes", icon: "M3 9l9-5 9 5v11H3zM8 20v-6h8v6" },
  ];
  const NODE_BY_ID = Object.fromEntries(NODES.map((n) => [n.id, n]));
  const CODE_NODES = NODES.filter((n) => n.kind === "code" && n.id !== "orchestrator").map((n) => n.id);

  // Cards are HTML on a CSS grid (real text: it wraps and keeps its pixel size at any panel width);
  // the edges are an SVG layer behind them, the moving packets an SVG layer in front. Layouts are
  // [column, row] cells; "list" (phones) stacks the cards in reading order without edges.
  const LAYOUTS = {
    compact: { cols: 3, at: { customer: [0, 0], resolution: [1, 0], operations: [2, 0], grounding: [0, 1], orchestrator: [1, 1], gateway: [2, 1],
      approval: [0, 2], policy: [1, 2], store: [2, 2], replyguard: [0, 3], recovery: [1, 3], warehouse: [2, 3] } },
    wide: { cols: 4, at: { customer: [0, 0], resolution: [1, 0], operations: [2, 0], warehouse: [3, 0], grounding: [0, 1], orchestrator: [1, 1],
      gateway: [2, 1], store: [3, 1], replyguard: [0, 2], approval: [1, 2], policy: [2, 2], recovery: [3, 2] } },
    list: { cols: 1, at: null },
  };
  const LAYOUT_MIN_WIDTH = { wide: 760, compact: 440 }; // graph width in CSS px; below compact: list
  const EDGES = [
    ["customer", "resolution"], ["resolution", "operations"], ["resolution", "orchestrator"], ["resolution", "grounding"],
    ["orchestrator", "grounding"], ["orchestrator", "operations"], ["operations", "gateway"], ["gateway", "store"],
    ["gateway", "policy"], ["orchestrator", "approval"], ["approval", "gateway"], ["warehouse", "store"],
    ["recovery", "store"], ["gateway", "recovery"], ["orchestrator", "replyguard"],
  ];
  const edgeKey = (a, b) => [a, b].sort().join("--");
  const EDGE_KEYS = new Set(EDGES.map(([a, b]) => edgeKey(a, b)));
  const DWELL_MS = 350; // SPEC §10.1: every event stays on screen at least this long at 1x

  let graph = { layout: null, nodes: {}, edges: {}, centers: {}, edgeLayer: null, packetLayer: null };

  function nodeCard(n, cell) {
    const status = el("p", { class: "node-status", "data-role": "status" });
    const counters = el("span", { class: "node-counters", "data-role": "counters" });
    const icon = svgEl("svg", { class: "node-icon", viewBox: "0 0 24 24", "aria-hidden": "true" }, svgEl("path", { d: n.icon }));
    // The badge sits in the footer with the counters, so the name gets the card's full width.
    const card = el("div", { class: "node", "data-node": n.id, "data-kind": n.kind, "data-status": "idle", "data-current": "false",
      role: "group", "aria-label": `${n.name} (${n.badge}): ${n.goal}`, title: `${n.name}: ${n.goal}` },
      el("div", { class: "node-head" }, icon, el("span", { class: "node-name", text: n.name })),
      el("p", { class: "node-goal", text: n.goal }), status,
      el("div", { class: "node-foot" }, el("span", { class: "node-badge", "data-testid": "node-badge", text: n.badge }), counters));
    if (cell) {
      card.style.gridColumn = String(cell[0] + 1);
      card.style.gridRow = String(cell[1] + 1);
    }
    return { card, status, counters };
  }

  function buildGraph(layoutName) {
    const layout = LAYOUTS[layoutName];
    const wrap = clear($("team-graph"));
    wrap.dataset.layout = layoutName;
    graph = { layout: layoutName, nodes: {}, edges: {}, centers: {}, edgeLayer: null, packetLayer: null };
    const grid = el("div", { class: "team-grid" });
    grid.style.setProperty("--cols", String(layout.cols));
    for (const n of NODES) {
      graph.nodes[n.id] = nodeCard(n, layout.at ? layout.at[n.id] : null);
      grid.append(graph.nodes[n.id].card);
    }
    graph.edgeLayer = svgEl("svg", { class: "team-edges", "aria-hidden": "true", preserveAspectRatio: "none" });
    for (const [a, b] of EDGES) {
      const line = svgEl("line", { class: "edge", "data-edge": edgeKey(a, b), "data-active": "false", "data-visited": "false" });
      graph.edges[edgeKey(a, b)] = line;
      graph.edgeLayer.append(line);
    }
    graph.packetLayer = svgEl("svg", { class: "team-packets", "aria-hidden": "true", preserveAspectRatio: "none" });
    wrap.append(graph.edgeLayer, grid, graph.packetLayer);
    positionEdges();
  }

  /** Card centres in the wrap's pixel space; the SVG layers use the same space (1 unit = 1 px). */
  function positionEdges() {
    if (!graph.layout || graph.layout === "list") return;
    const box = $("team-graph").getBoundingClientRect();
    for (const n of NODES) {
      const r = graph.nodes[n.id].card.getBoundingClientRect();
      graph.centers[n.id] = [Math.round(r.left - box.left + r.width / 2), Math.round(r.top - box.top + r.height / 2)];
    }
    for (const layer of [graph.edgeLayer, graph.packetLayer]) layer.setAttribute("viewBox", `0 0 ${Math.round(box.width)} ${Math.round(box.height)}`);
    for (const [a, b] of EDGES) {
      const [x1, y1] = graph.centers[a];
      const [x2, y2] = graph.centers[b];
      setProps(graph.edges[edgeKey(a, b)], { x1, y1, x2, y2 });
    }
  }

  /** What one audit entry means on the team graph: node updates, the edge it travels, a narration.
   *  `ctx` carries facts across the entries of one turn (the last handoff, whether Ops is active). */
  function describeEntry(e, ctx) {
    const fx = [];
    const add = (node, status, text, from, extra = {}) => fx.push({ node, status, text, edge: from ? [from, node] : null, ...extra });
    const err = e.error || null;
    const code = err ? err.code : null;
    const args = e.arguments || {};
    const result = e.result || {};
    switch (e.kind) {
      case "llm_step": {
        const agent = e.agent === "operations" ? "operations" : "resolution";
        if (agent === "operations" && !ctx.opsActive) {
          ctx.opsActive = true;
          const brief = ctx.handoff ? `${ctx.handoff.intent}${ctx.handoff.order_id ? ` ${lineRef(ctx.handoff.order_id, ctx.handoff.line_id)}` : ""}` : "brief";
          add("operations", "working", `received brief: ${brief}`, "resolution", { packet: `brief: ${brief}` });
        }
        if (code === "LLM_ERROR") {
          add(agent, "blocked", "the model did not answer");
          break;
        }
        add(agent, err ? "blocked" : "working", err ? `protocol error ${code}` : toolSummary(e.tool, args), null, { llm: result });
        if (agent === "resolution" && e.tool === "handoff_to_operations") ctx.handoff = args;
        if (agent === "resolution" && e.tool === "reply_to_customer") add("customer", "info", "reads the reply", "resolution");
        break;
      }
      case "grounding": {
        const target = result.unique_target;
        if (target) add("grounding", "ok", `one target: ${lineRef(target[0], target[1])} · remedy ${result.remedy_cue}`, "orchestrator");
        else if (result.not_found_code) add("grounding", "blocked", `${result.not_found_code === "ORDER_NOT_FOUND" ? "unknown order" : "unknown line"} ${[...(result.unknown_order_refs || []), ...(result.unknown_line_refs || [])].join(", ")}`.trim(), "orchestrator");
        else add("grounding", "info", `${plural((result.candidates || []).length, "candidate line")}, no unique target yet`, "orchestrator");
        break;
      }
      case "target":
        add("grounding", "ok", `grounded ${lineRef(args.order_id, args.line_id)} · ${args.intent}${args.requested_size ? ` to ${args.requested_size}` : ""}${groundedByNote(args.grounded_by)}`, "orchestrator");
        break;
      case "gate":
        add("grounding", "blocked", gateVerdict(code, args), "resolution");
        break;
      case "guard":
        describeGuard(e, add);
        break;
      case "grant":
        add("approval", "ok", `customer confirmed ${args.proposal_id}`, "orchestrator");
        break;
      case "confirmation":
        add("approval", "blocked", "no earlier proposal to confirm", "orchestrator");
        break;
      case "tool":
        describeTool(e, ctx, add);
        break;
      case "route":
        if (result.replay) add("recovery", "ok", `replay: ${result.transaction_id} already processed, no new effects`, "gateway");
        else add("recovery", "blocked", "request already completed for another item", "gateway");
        break;
      case "fallback":
        // Fallbacks are read-only now: code checks status or a proposal's verdict, it never acts for an agent.
        add("orchestrator", "info", `read-only check by code: ${e.tool}${result.reason ? ` (${short(result.reason, 50)})` : ""}`, null,
          { edge: ["orchestrator", ["check_proposal", "get_request_status"].includes(e.tool) ? "operations" : "resolution"] });
        break;
      case "fault":
        if (e.tool === "stock_change" && !err) {
          if (result.applied) add("warehouse", "info", `${result.sku} stock set to ${result.new_stock} (was ${result.old_stock})`, null,
            { edge: ["warehouse", "store"], chips: [{ kind: "stock", text: `${result.sku} stock ${result.old_stock}→${result.new_stock} (external)` }] });
          else add("warehouse", "blocked", `unknown SKU ${args.sku}: ignored`);
        } else if (e.tool === "timeout_after_commit" && !err) {
          add("recovery", "info", "armed: the next commit will time out after it persists");
        } else {
          add("warehouse", "blocked", `fault ignored: ${code}`);
        }
        break;
      case "ops_report": {
        ctx.opsActive = false;
        const pid = result.proposal ? ` ${result.proposal.proposal_id}` : "";
        const reasons = (result.reason_codes || []).length ? ` (${result.reason_codes.join(", ")})` : "";
        add("resolution", "working", `report: ${result.outcome}${pid}${reasons}`, "operations", { packet: `report: ${result.outcome}${pid}` });
        add("operations", "ok", `reported ${result.outcome}${pid}`);
        break;
      }
      default: // an entry kind this page does not know yet: still shown, on the Orchestrator card
        add("orchestrator", code ? "blocked" : "info", `${e.kind} · ${e.tool}${code ? `: ${reasonText(code)}` : ""}`);
    }
    if (groundedBy(e) === "agent") add("grounding", "ok", agentGroundingText(e), "resolution");
    return { fx, narration: narrate(e, fx) };
  }

  // Why code accepted a target line (desk/grounding.py TargetCheck.grounded_by); "agent" gets its own line.
  const GROUNDED_BY_NOTE = { customer_ids: "the IDs the customer typed", only_line: "the only line it can be",
    "agent+grounding": "the agent's pick, code's reading agrees" };
  const groundedByNote = (by) => (GROUNDED_BY_NOTE[by] ? ` · ${GROUNDED_BY_NOTE[by]}` : "");

  /** Who grounded a target: "agent" means the model named it and code accepted it on cited evidence. */
  function groundedBy(e) {
    return e.grounded_by || (e.result || {}).grounded_by || (e.arguments || {}).grounded_by || null;
  }

  /** The evidence an agent-grounded entry cites (a string, a list or an object of strings). */
  function evidenceText(e) {
    const result = e.result || {};
    const args = e.arguments || {};
    const raw = e.evidence ?? result.evidence ?? args.evidence ?? result.cited ?? args.cited ?? null;
    const parts = Array.isArray(raw) ? raw : raw && typeof raw === "object" ? Object.values(raw) : raw === null ? [] : [raw];
    return parts.filter((p) => typeof p === "string" || typeof p === "number").map(String).join("; ");
  }

  function agentGroundingText(e) {
    const args = e.arguments || {};
    const result = e.result || {};
    const order = args.order_id || result.order_id;
    const where = order ? ` ${lineRef(order, args.line_id || result.line_id)}` : "";
    const cites = evidenceText(e);
    return `agent grounded${where}${cites ? ` · cites “${short(cites, 60)}”` : ""}`;
  }

  /** "unknown order ORD-99" rather than a bare code: the card says which id did not exist. */
  function gateVerdict(code, args) {
    if (code === "ORDER_NOT_FOUND" && args.order_id) return `unknown order ${args.order_id}`;
    if (code === "LINE_NOT_FOUND" && args.line_id) return `unknown line ${args.line_id} in ${args.order_id}`;
    return `rejected handoff: ${reasonText(code)}`;
  }

  /** "model said 'processed' on a propose; template used" from the guard's violation list. */
  function replyGuardVerdict(decision, violations) {
    const claim = (violations || []).find((v) => v.startsWith("success_claim:") || v.startsWith("failure_claim:"));
    if (claim) return `model said "${claim.split(":")[1]}" on a ${decision}; template used`;
    return `template used: ${(violations || []).slice(0, 2).join(", ")}`;
  }

  function describeGuard(e, add) {
    const args = e.arguments || {};
    const result = e.result || {};
    if (e.tool === "clarify_veto") add("grounding", "blocked", `veto: line and remedy are clear, hand off ${args.intent || ""}`.trim(), "resolution");
    else if (e.tool === "grant_withheld") add("approval", "blocked", `withheld: ${result.reason || "the message changes the request"}`, "orchestrator");
    else if (e.tool === "reply_guard") add("replyguard", "blocked", replyGuardVerdict(args.decision, result.violations), "orchestrator");
    else if (e.tool === "report_mismatch") add("replyguard", "blocked", `Ops said "${args.claimed}", records say "${result.actual}"`, "orchestrator");
    else if (e.tool === "duplicate_call") add("orchestrator", "info", `repeat call answered from cache: ${(args.tool || "")}`);
    else if (e.tool === "dropped_calls") add("orchestrator", "info", "extra tool calls in one answer were dropped");
    else if (e.tool === "code_deny") add("policy", "blocked", `code said no on the facts: ${(result.reasons || []).map(reasonText).join(" ")}`, "orchestrator");
    else if (e.tool === "handoffs_closed") add("orchestrator", "info", `handoffs closed: ${result.reason || "only a question is allowed now"}`);
    else add("orchestrator", "info", `guard ${e.tool}${e.error ? `: ${reasonText(e.error.code)}` : ""}`);
  }

  /** An extra hop to animate (the packet and edge highlight) without changing any card. */
  function edgeHop(add, from, to) {
    add(to, null, null, from, { edgeOnly: true });
  }

  function describeTool(e, ctx, add) {
    const err = e.error || null;
    const code = err ? err.code : null;
    const result = e.result || {};
    const details = (err && err.details) || {};
    const by = e.agent === "orchestrator" ? null : "operations";
    switch (e.tool) {
      case "prefetch_account_card":
        add("grounding", "ok", `account card: ${plural(result.count || 0, "line")}`, "orchestrator");
        return;
      case "find_order_lines":
        add("grounding", "info", `lookup: ${plural(result.count || 0, "line")}${result.order_not_on_account ? " (order not on this account)" : ""}`, "resolution");
        return;
      case "check_eligibility":
        if (err) add("policy", "blocked", reasonText(code), "gateway");
        else add("policy", result.eligible ? "ok" : "blocked", result.eligible ? `eligible · day ${result.days_since_delivery} of ${result.window_days}` : (result.reasons || []).map(reasonText).join(" "), "gateway");
        return;
      case "check_proposal": // read-only dry run after Operations ended without a decision
        if (!err) add("policy", "info", "dry run: would be eligible · no proposal made", "gateway");
        else if (codeClass(code) === "refusal") add("policy", "blocked", `dry run: ${(details.reasons || [code]).map(reasonText).join(" ")}`, "gateway");
        else add("policy", "info", `dry run: ${reasonText(code)}`, "gateway");
        return;
      case "list_exchange_options":
        if (err) add("policy", "info", `options: ${code}`, "gateway");
        else add("policy", "info", `${plural(result.valid_count || 0, "valid size")}${result.requested_sku ? ` · asked ${result.requested_sku}` : ""}`, "gateway");
        return;
      case "create_proposal":
        if (!err) {
          const p = result.proposal || {};
          const elig = p.eligibility || {};
          add("policy", "ok", `eligible · day ${elig.days_since_delivery} of ${elig.window_days}`, "gateway");
          add("gateway", "ok", `${result.reused ? "same proposal re-presented" : "proposal ready"}: ${p.proposal_id} · nothing moved`, by);
        } else if (codeClass(code) === "refusal") {
          add("policy", "blocked", (details.reasons || [code]).map(reasonText).join(" "), "gateway");
          add("gateway", "info", "no proposal created", by);
        } else if (codeClass(code) === "gate") {
          add("approval", "blocked", `waiting for the customer: ${reasonText(code)}`, "gateway");
        } else {
          add("gateway", "blocked", `refused ${code} (agent slip, nothing moved)`, by);
        }
        return;
      case "execute_proposal":
        describeExecute(e, ctx, add, by);
        return;
      case "get_request_status":
        if (result.status === "completed") {
          const tx = result.record ? result.record.transaction_id : "";
          add("recovery", "ok", ctx.timedOut ? `reconciled: ${tx} found, no second commit` : `status: ${tx} already completed`, "gateway");
          if (ctx.timedOut) edgeHop(add, "recovery", "store"); // the status check reads the store
        } else {
          add("recovery", "info", "status: no completed record", "gateway");
        }
        return;
      default:
        add("gateway", err ? "blocked" : "info", `${e.tool}${code ? ` ${code}` : ""}`, by);
    }
  }

  function describeExecute(e, ctx, add, by) {
    const err = e.error || null;
    const code = err ? err.code : null;
    const result = e.result || {};
    const details = (err && err.details) || {};
    const chips = effectChips(e.committed_effects);
    const tx = result.transaction ? result.transaction.transaction_id : "";
    if (!err && result.replay) {
      add("recovery", "ok", `replay: ${tx} already processed`, "gateway");
    } else if (!err) {
      add("gateway", "ok", `committed ${tx} atomically`, by);
      add("store", "ok", chips.map((c) => c.text).join(" · "), "gateway", { chips });
    } else if (code === "TIMEOUT") {
      ctx.timedOut = true;
      add("gateway", "ok", `committed ${tx}, then the answer was lost`, by);
      add("store", "ok", chips.map((c) => c.text).join(" · "), "gateway", { chips });
      add("recovery", "working", "timeout after commit: check the records first", "gateway");
    } else if (code === "CONFIRMATION_REQUIRED") {
      add("approval", "blocked", "no explicit confirmation this turn: blocked", "gateway");
    } else if (codeClass(code) === "refusal") {
      add("policy", "blocked", `revalidation: ${(details.reasons || [code]).map(reasonText).join(" ")}`, "gateway");
      add("gateway", "blocked", "nothing committed; proposal invalidated", by);
    } else {
      add("gateway", "blocked", `refused ${code}`, by);
    }
  }

  function narrate(e, allFx) {
    const fx = allFx.filter((f) => !f.edgeOnly);
    const main = fx[fx.length - 1];
    if (!main) return `${e.kind} ${e.tool}`;
    const mark = main.status === "blocked" ? " ✗" : main.status === "ok" ? " ✓" : "";
    const from = main.edge ? `${NODE_BY_ID[main.edge[0]].name} → ` : "";
    const first = fx.length > 1 && fx[0].edge ? `${NODE_BY_ID[fx[0].edge[0]].name} → ${NODE_BY_ID[fx[0].node].name}: ${fx[0].text}; ` : "";
    return `${first}${from}${NODE_BY_ID[main.node].name}: ${main.text}${mark}`;
  }

  function emptyNode() {
    return { status: "idle", text: "", checks: 0, blocks: 0, llm: { steps: 0, prompt: 0, cached: 0, completion: 0, ms: 0, decodeMs: 0 } };
  }

  /** The graph's state after the first `upto` entries of a turn (pure: replay = refold). */
  function foldTeam(entries, upto) {
    const nodes = Object.fromEntries(NODES.map((n) => [n.id, emptyNode()]));
    const visited = new Set();
    const chips = [];
    const log = [];
    const ctx = { handoff: null, opsActive: false, timedOut: false };
    let current = { fx: [], narration: "" };
    for (let i = 0; i < upto && i < entries.length; i += 1) {
      const described = describeEntry(entries[i], ctx);
      for (const f of described.fx) applyEffect(nodes, f, visited, chips);
      const main = described.fx.filter((f) => !f.edgeOnly).pop() || {};
      log.push({ text: described.narration, status: main.status });
      current = described;
    }
    return { nodes, visited, chips, log, current };
  }

  function applyEffect(nodes, f, visited, chips) {
    if (f.edge && EDGE_KEYS.has(edgeKey(...f.edge))) visited.add(edgeKey(...f.edge));
    if (f.edgeOnly) return;
    const node = nodes[f.node];
    if (f.status === "working" && NODE_BY_ID[f.node].kind === "llm") {
      for (const other of ["resolution", "operations"]) if (other !== f.node && nodes[other].status === "working") nodes[other].status = "idle";
    }
    node.status = f.status;
    node.text = f.text;
    if (f.status === "blocked") node.blocks += 1;
    else if (NODE_BY_ID[f.node].kind === "code") node.checks += 1;
    if (f.llm) {
      node.llm.steps += 1;
      node.llm.prompt += f.llm.prompt_tokens || 0;
      node.llm.cached += f.llm.cached_tokens || 0;
      node.llm.completion += f.llm.completion_tokens || 0;
      node.llm.ms += f.llm.latency_ms || 0;
      node.llm.decodeMs += f.llm.predicted_ms || 0;
    }
    if (f.chips) chips.push(...f.chips);
  }

  /** Per-turn counters: LLM agents get steps, time, tok/s and prompt/cached/completion tokens. */
  function countersLines(id, n) {
    const kind = NODE_BY_ID[id].kind;
    if (kind === "llm") {
      if (!n.llm.steps) return [modelChipText() || "no steps this turn"];
      const toks = n.llm.decodeMs > 0 ? ` · ${Math.round((n.llm.completion / n.llm.decodeMs) * 1000)} tok/s` : "";
      return [`${plural(n.llm.steps, "step")} · ${fmtMs(n.llm.ms)}${toks}`, `${n.llm.prompt} in · ${n.llm.cached} cached · ${n.llm.completion} out`];
    }
    if (kind === "code") return [n.blocks ? `${plural(n.checks, "check")} · ${plural(n.blocks, "block")}` : plural(n.checks, "check")];
    return [];
  }

  // --- the player: live queue and replay share one cursor over one turn's entries --------------

  const player = { turn: null, cursor: 0, timer: null, playing: false, live: false, speed: 1 };

  function playerEntries() {
    return player.turn === null ? [] : entriesOfTurn(player.turn);
  }

  function turnFinished(turn) {
    return !(app.live && app.live.turn === turn && !app.live.finished);
  }

  function tick() {
    player.timer = null;
    const entries = playerEntries();
    if (player.cursor < entries.length) {
      player.cursor += 1;
      renderTeam(true);
      player.timer = window.setTimeout(tick, DWELL_MS / player.speed);
      return;
    }
    if (player.playing) player.playing = false;
    if (player.live && turnFinished(player.turn)) player.live = false;
    renderTeam(false);
  }

  /** Live: called for every streamed entry; plays at once if the queue was idle. */
  function kick() {
    if (!player.timer) tick();
  }

  function stopPlayer() {
    if (player.timer) window.clearTimeout(player.timer);
    player.timer = null;
    player.playing = false;
  }

  function showTurn(turn, atEnd = true) {
    stopPlayer();
    player.live = false;
    player.turn = turn;
    player.cursor = atEnd ? entriesOfTurn(turn).length : 0;
    renderReplayPicker();
    renderTeam(false);
  }

  function startReplay() {
    if (player.turn === null || app.busy) return;
    stopPlayer();
    player.live = false;
    player.playing = true;
    player.cursor = 0;
    renderTeam(false);
    player.timer = window.setTimeout(tick, DWELL_MS / player.speed);
  }

  function stepReplay(delta) {
    if (player.turn === null || app.busy) return;
    stopPlayer();
    const total = playerEntries().length;
    player.cursor = Math.max(0, Math.min(total, player.cursor + delta));
    renderTeam(delta > 0);
  }

  function renderTeam(animate) {
    if (!graph.layout) return;
    const entries = playerEntries();
    const state = foldTeam(entries, player.cursor);
    const record = app.session ? app.session.turns.find((t) => t.turn === player.turn) : null;
    const finishedHere = player.cursor >= entries.length && turnFinished(player.turn) && record && record.output;
    const currentNodes = new Set(state.current.fx.filter((f) => !f.edgeOnly).map((f) => f.node));
    const currentEdges = new Set(state.current.fx.filter((f) => f.edge && EDGE_KEYS.has(edgeKey(...f.edge))).map((f) => edgeKey(...f.edge)));
    const customer = state.nodes.customer;
    if (finishedHere) {
      const d = record.output.decision;
      customer.status = d === "completed" ? "ok" : d === "deny" || d === "error" ? "blocked" : "info";
      customer.text = `hears: ${d}`;
      if (!entries.some((e) => e.kind === "guard" && e.tool === "reply_guard")) {
        state.nodes.replyguard.status = "ok";
        state.nodes.replyguard.text = "reply matches the records";
        state.nodes.replyguard.checks += 1;
      }
    } else if (!customer.text && record) {
      customer.text = `“${short(record.message, 40)}”`;
    } else if (!customer.text && app.live && app.live.turn === player.turn) {
      customer.text = `“${short(app.live.message, 40)}”`;
    }
    for (const n of NODES) {
      const view = graph.nodes[n.id];
      const s = state.nodes[n.id];
      view.card.dataset.status = s.status;
      view.card.dataset.current = currentNodes.has(n.id) ? "true" : "false";
      view.status.textContent = statusMark(s.status) + (s.text || "idle");
      view.status.title = s.text || "";
      const counters = countersLines(n.id, s);
      clear(view.counters).append(...counters.map((line, i) => el("span", { class: i ? "counter-detail" : "", text: line })));
    }
    for (const [key, line] of Object.entries(graph.edges)) {
      line.dataset.active = currentEdges.has(key) ? "true" : "false";
      line.dataset.visited = state.visited.has(key) ? "true" : "false";
    }
    if (animate && !reducedMotion.matches) {
      for (const f of state.current.fx) if (f.edge && EDGE_KEYS.has(edgeKey(...f.edge))) animatePacket(f.edge[0], f.edge[1]);
    }
    renderMission(state, entries);
    renderStoreChips(state.chips);
    renderTurnStrip(entries, record);
    $("replay-pos").textContent = `${player.cursor} / ${entries.length}`;
    $("replay-pos").dataset.step = String(player.cursor);
    $("replay-pos").dataset.steps = String(entries.length);
    $("replay").dataset.playing = player.playing ? "true" : "false";
    const backlog = entries.length - player.cursor;
    $("catching-up").hidden = !(player.live && backlog > 2);
    $("catching-up").textContent = `catching up… ${backlog} queued`;
    for (const id of ["replay-play", "replay-back", "replay-fwd", "replay-turn"]) $(id).disabled = app.busy || player.turn === null;
  }

  const statusMark = (status) => ({ ok: "✓ ", blocked: "✗ ", working: "● ", info: "• " }[status] || "");

  function modelChipText() {
    const h = app.health;
    if (!h) return "";
    return h.mode === "fake-policy" ? "fake-policy double" : `${h.model} · ${h.quantization}`;
  }

  function animatePacket(from, to) {
    if (graph.layout === "list" || !graph.centers[from] || !graph.centers[to]) return; // a list has no edges to travel
    const [x1, y1] = graph.centers[from];
    const [x2, y2] = graph.centers[to];
    const dot = svgEl("circle", { class: "packet", r: 7, cx: 0, cy: 0 });
    const motion = svgEl("animateMotion", { dur: `${Math.round(DWELL_MS / player.speed)}ms`, fill: "freeze", path: `M${x1},${y1} L${x2},${y2}`, begin: "indefinite" });
    dot.append(motion);
    graph.packetLayer.append(dot);
    motion.beginElement(); // SMIL: supported by every current browser engine
    window.setTimeout(() => dot.remove(), DWELL_MS / player.speed + 120);
  }

  function renderMission(state, entries) {
    const now = state.current.narration || (entries.length ? "Ready to replay this turn." : "Waiting for the first event.");
    $("mission-now").textContent = now;
    const list = clear($("mission-log"));
    for (const item of state.log.slice(-40)) list.append(el("li", { "data-status": item.status || "", text: item.text }));
    list.scrollTop = list.scrollHeight;
  }

  function renderStoreChips(chips) {
    const box = clear($("store-chips"));
    if (chips.length) box.append(el("span", { class: "hint", text: "Store:" }));
    for (const chip of chips) box.append(el("span", { class: `chip chip-${chip.kind}`, "data-chip": chip.kind, text: chip.text }));
  }

  /** "Turn 2 · 7 steps · 2 LLM agents · 4 specialists consulted · 1 block · 1.9 s · completed" */
  function turnSummary(entries, record) {
    const full = foldTeam(entries, entries.length);
    const llmSteps = entries.filter((e) => e.kind === "llm_step");
    const agents = new Set(llmSteps.map((e) => e.agent)).size;
    const consulted = CODE_NODES.filter((id) => full.nodes[id].checks + full.nodes[id].blocks > 0).length;
    const blocks = Object.values(full.nodes).reduce((sum, n) => sum + n.blocks, 0);
    const parts = [`Turn ${player.turn}`, plural(llmSteps.length, "step"), plural(agents, "LLM agent"), `${plural(consulted, "specialist")} consulted`, plural(blocks, "block")];
    if (record && record.output && player.cursor >= entries.length) parts.push(fmtMs(record.latency_ms), record.output.decision);
    else if (player.live || !record || !record.output) parts.push("running");
    else parts.push(`step ${player.cursor} of ${entries.length}`);
    return parts;
  }

  function renderTurnStrip(entries, record) {
    const strip = clear($("turn-strip"));
    if (player.turn === null) {
      strip.textContent = "No turn yet.";
      return;
    }
    if (player.turn === 0) {
      strip.append(el("strong", { text: "Setup" }), ` · ${plural(entries.length, "event")} before the first turn`);
      return;
    }
    const parts = turnSummary(entries, record);
    strip.append(el("strong", { text: parts[0] }), ` · ${parts.slice(1).join(" · ")}`);
  }

  function renderReplayPicker() {
    const select = clear($("replay-turn"));
    const turns = [...new Set(allEntries().map((e) => e.turn))].sort((a, b) => a - b);
    for (const t of turns) select.append(el("option", { value: t, text: t === 0 ? "setup" : `turn ${t}` }));
    if (player.turn !== null) select.value = String(player.turn);
  }

  function chooseLayout() {
    const width = $("team-graph").clientWidth;
    const wanted = width >= LAYOUT_MIN_WIDTH.wide ? "wide" : width >= LAYOUT_MIN_WIDTH.compact ? "compact" : "list";
    if (wanted !== graph.layout) {
      buildGraph(wanted);
      renderTeam(false);
    }
    positionEdges(); // the cards may have changed height (text wraps differently at a new width)
  }

  // ============================================================================================
  // 9. Audit trail: every entry, grouped by turn, collapsible JSON

  const auditOpen = new Set(); // seq numbers the viewer expanded (kept across re-renders)

  function entryFlag(e) {
    if (e.kind === "guard") return "guard";
    if (e.kind === "fallback") return "fallback";
    if (e.kind === "gate") return "gate";
    if (e.kind === "fault") return "fault";
    if (e.error && e.kind !== "tool") return "error";
    if (e.error && codeClass(e.error.code) === "failure") return "error";
    return "";
  }

  function entryTitle(e) {
    const args = e.arguments || {};
    const result = e.result || {};
    if (e.kind === "llm_step" && e.tool === "handoff_to_operations") {
      return [el("span", { class: "arrow", text: "Resolution → Operations " }), `handoff ${args.intent}${args.order_id ? ` ${lineRef(args.order_id, args.line_id)}` : ""}${args.requested_size ? ` → ${args.requested_size}` : ""}`];
    }
    if (e.kind === "ops_report") {
      return [el("span", { class: "arrow", text: "Operations → Resolution " }), `report ${result.outcome}${result.proposal ? ` ${result.proposal.proposal_id}` : ""}${(result.reason_codes || []).length ? ` (${result.reason_codes.join(", ")})` : ""}`];
    }
    if (e.kind === "llm_step") return [`${e.agent === "resolution" ? "Resolution" : "Operations"} chose ${toolSummary(e.tool, args)}`];
    if (e.kind === "tool") {
      const outcome = e.error ? `→ ${e.error.code}` : e.tool === "create_proposal" && result.proposal ? `→ ${result.proposal.proposal_id}` : e.tool === "execute_proposal" && result.transaction ? `→ ${result.transaction.transaction_id}` : "→ ok";
      return [`${e.tool}(${Object.values(args).filter((v) => v !== null && v !== "").join(", ")}) ${outcome}`];
    }
    if (e.kind === "grounding") return [`${result.unique_target ? `one target ${lineRef(result.unique_target[0], result.unique_target[1])}` : "no unique target"} · remedy cue ${result.remedy_cue}`];
    if (e.kind === "gate") return [`rejected the handoff: ${e.error ? e.error.code : ""}`];
    if (e.kind === "target") return [`grounded target ${lineRef(args.order_id, args.line_id)} · ${args.intent}${args.requested_size ? ` (size ${args.requested_size})` : ""}`];
    if (e.kind === "grant") return [`explicit confirmation of ${args.proposal_id} (this turn only)`];
    if (e.kind === "route") return [result.replay ? `already completed: replay ${result.transaction_id}, no new effects` : "already completed for another item: refuse"];
    if (e.kind === "fallback") return [`read-only check by code: ${e.tool}${result.reason ? ` (${result.reason})` : ""}`];
    if (e.kind === "guard" && Array.isArray(result.reasons)) return [`${e.tool}: ${result.reasons.map(reasonText).join(" ")}`];
    if (e.kind === "guard") return [`${e.tool}${result.reason ? `: ${result.reason}` : result.violations ? `: ${result.violations.join(", ")}` : ""}`];
    if (e.kind === "fault" && e.tool === "stock_change" && result.sku) return [`external stock change ${result.sku} ${result.old_stock} → ${result.new_stock}${result.applied ? "" : " (not applied)"}`];
    if (e.kind === "fault") return [`${e.tool}${result.armed ? ": armed for the next commit" : ""}`];
    return [`${e.tool}${e.error ? `: ${reasonText(e.error.code)}` : ""}`];
  }

  function entryBadges(e) {
    const badges = [];
    if (e.kind === "llm_step" && e.result) {
      const r = e.result;
      badges.push(el("span", { class: "mini", title: "step latency", text: fmtMs(r.latency_ms) }));
      badges.push(el("span", { class: "mini", title: "prompt tokens / cached / completion", text: `${r.prompt_tokens}/${r.cached_tokens}/${r.completion_tokens} tok` }));
      if (r.predicted_ms > 0) badges.push(el("span", { class: "mini", text: `${Math.round((r.completion_tokens / r.predicted_ms) * 1000)} tok/s` }));
    }
    if ((e.committed_effects || []).length) badges.push(el("span", { class: "mini mini-effect", text: `${plural(e.committed_effects.length, "effect")} committed` }));
    if (e.error) badges.push(el("span", { class: "mini mini-bad", text: e.error.code }));
    if (e.visible_to_agent === false) badges.push(el("span", { class: "mini", title: "the agent never saw this result", text: "hidden from agent" }));
    if (groundedBy(e) === "agent") badges.push(el("span", { class: "mini mini-agent", title: evidenceText(e) || "no evidence text", text: "grounded by agent" }));
    return badges;
  }

  function auditEntry(e) {
    const details = el("details", { class: "audit-entry", "data-seq": e.seq, "data-kind": e.kind, "data-flag": entryFlag(e),
      "data-effects": (e.committed_effects || []).length > 0, "data-handoff": e.tool === "handoff_to_operations" || e.kind === "ops_report", "data-testid": "audit-entry" });
    if (auditOpen.has(e.seq)) details.open = true;
    details.addEventListener("toggle", () => (details.open ? auditOpen.add(e.seq) : auditOpen.delete(e.seq)));
    const chips = effectChips(e.committed_effects);
    details.append(el("summary", {},
      el("span", { class: "seq", text: `#${e.seq}` }),
      el("span", { class: "agent-chip", "data-agent": e.agent, text: AGENT_LABEL[e.agent] || e.agent }),
      el("span", { class: "entry-title" }, el("span", { class: "entry-kind", text: e.kind }), ...entryTitle(e)),
      el("span", { class: "entry-badges" }, ...entryBadges(e))));
    const body = el("div", { class: "entry-body" });
    if (chips.length) body.append(el("div", { class: "store-chips" }, chips.map((c) => el("span", { class: `chip chip-${c.kind}`, text: c.text }))));
    body.append(el("h4", { text: "arguments" }), el("pre", { text: json(e.arguments) }));
    if (e.result !== null && e.result !== undefined) body.append(el("h4", { text: "result" }), el("pre", { text: json(e.result) }));
    if (e.error) body.append(el("h4", { text: "error" }), el("pre", { text: json(e.error) }));
    if ((e.committed_effects || []).length) body.append(el("h4", { text: "committed effects" }), el("pre", { text: json(e.committed_effects) }));
    details.append(body);
    return details;
  }

  function renderAudit() {
    const box = clear($("audit-timeline"));
    const hideGrounding = $("audit-hide-grounding").checked;
    const records = turnRecords();
    const entries = allEntries();
    if (!entries.length) {
      box.append(el("p", { class: "hint", text: "No audit entries yet." }));
      return;
    }
    const turns = [...new Set(entries.map((e) => e.turn))].sort((a, b) => a - b);
    for (const turn of turns) {
      const record = records.find((r) => r.turn === turn);
      const head = el("h3", {}, turn === 0 ? "Setup" : `Turn ${turn}`);
      if (record) head.append(el("span", { class: "turn-msg", text: `“${short(record.message, 90)}”` }));
      if (record && record.output) head.append(el("span", { class: "pill", "data-decision": record.output.decision, text: record.output.decision }));
      const group = el("section", { class: "turn-group", "data-turn-group": turn }, head);
      for (const e of entries.filter((x) => x.turn === turn)) {
        if (hideGrounding && e.kind === "grounding") continue;
        group.append(auditEntry(e));
      }
      box.append(group);
    }
  }

  // ============================================================================================
  // 10. State inspector: products, lines, records, events (diff against the initial state)

  function table(headers, rows, testid) {
    if (!rows.length) return el("p", { class: "empty", "data-testid": testid, text: "none" });
    return el("table", { "data-testid": testid },
      el("thead", {}, el("tr", {}, headers.map((h) => el("th", { scope: "col", text: h })))),
      el("tbody", {}, rows));
  }

  const yesNo = (value) => el("span", { class: value === true ? "yes" : "no", text: value === true ? "yes" : value === false ? "no" : String(value) });

  function renderInspector() {
    const s = app.session;
    const body = clear($("drawer-body"));
    if (!s) return;
    const initial = s.initial_state;
    const current = s.final_state;
    const initialStock = Object.fromEntries((initial.products || []).map((p) => [p.sku, p.stock]));
    const productRows = (current.products || []).map((p) => {
      const before = initialStock[p.sku];
      const changed = before !== p.stock;
      return el("tr", { "data-sku": p.sku, "data-changed": changed },
        el("td", { class: "mono", text: p.sku }), el("td", { text: p.product_id }), el("td", { text: p.size }),
        el("td", { class: "num", text: money(p.price_cents) }),
        el("td", { class: "num", "data-testid": "stock" }, changed ? el("span", { class: "diff", text: `${before} → ${p.stock}` }) : String(p.stock)));
    });
    const initialLines = {};
    for (const o of initial.orders || []) for (const l of o.lines || []) initialLines[`${o.order_id}/${l.line_id}`] = l;
    const lineRows = [];
    for (const o of current.orders || []) {
      for (const l of o.lines || []) {
        const before = initialLines[`${o.order_id}/${l.line_id}`] || {};
        const changed = before.returned !== l.returned;
        lineRows.push(el("tr", { "data-line": `${o.order_id}/${l.line_id}`, "data-changed": changed },
          el("td", { class: "mono", text: `${o.order_id}/${l.line_id}` }), el("td", { text: o.customer_id }), el("td", { text: o.status }),
          el("td", { text: o.delivered_on || "–" }), el("td", { class: "mono", text: l.sku }), el("td", { class: "num", text: money(l.paid_cents) }),
          el("td", {}, yesNo(l.unopened)), el("td", {}, yesNo(l.final_sale)),
          el("td", { "data-testid": "returned" }, changed ? el("span", { class: "diff", text: `${before.returned} → ${l.returned}` }) : yesNo(l.returned))));
      }
    }
    const recordRows = (items, idField, extra) => (items || []).map((r) => el("tr", {},
      el("td", { class: "mono", text: r[idField] }), el("td", { class: "mono", text: r.transaction_id }), el("td", { class: "mono", text: `${r.order_id}/${r.line_id}` }),
      el("td", { class: "mono", text: r.proposal_id }), ...extra(r)));
    const idem = Object.entries(current.idempotency || {}).map(([key, r]) => el("tr", {},
      el("td", { class: "mono", text: key }), el("td", { text: r.status }), el("td", { text: r.action }), el("td", { class: "mono", text: r.transaction_id }),
      el("td", { class: "num", text: money(r.amount_cents) })));
    const proposals = (s.proposals || []).map((p) => el("tr", {},
      el("td", { class: "mono", text: p.proposal_id }), el("td", { text: p.remedy }), el("td", { class: "mono", text: `${p.order_id}/${p.line_id}` }),
      el("td", { class: "mono", text: p.replacement_sku || "–" }), el("td", { class: "num", text: money(p.amount_cents) }), el("td", { text: p.status })));
    const events = (s.external_events || []).map((ev) => el("tr", {},
      el("td", { class: "mono", text: ev.event_id }), el("td", { class: "mono", text: ev.sku }), el("td", { class: "num", text: `${ev.old_stock} → ${ev.new_stock}` }),
      el("td", { text: ev.before_turn === null ? "now (failure lab)" : `before turn ${ev.before_turn}` }), el("td", {}, yesNo(ev.applied))));
    const section = (title, content, wide = false) => el("section", { class: wide ? "inspect inspect-wide" : "inspect" }, el("h3", { text: title }), content);
    body.append(
      section("Products (stock vs initial)", table(["SKU", "product", "size", "price", "stock"], productRows, "products")),
      section("Order lines", table(["order/line", "customer", "status", "delivered", "SKU", "paid", "unopened", "final sale", "returned"], lineRows, "lines"), true),
      section("Refunds", table(["refund", "transaction", "order/line", "proposal", "amount"], recordRows(current.refunds, "refund_id", (r) => [el("td", { class: "num", text: money(r.amount_cents) })]), "records-refunds")),
      section("Exchanges", table(["exchange", "transaction", "order/line", "proposal", "original → replacement"], recordRows(current.exchanges, "exchange_id", (r) => [el("td", { class: "mono", text: `${r.original_sku} → ${r.replacement_sku}` })]), "records-exchanges")),
      section("Returns", table(["return", "transaction", "order/line", "proposal", "remedy"], recordRows(current.returns, "return_id", (r) => [el("td", { text: `${r.remedy} → ${r.disposition}` })]), "records-returns")),
      section("Quarantine (not sellable)", table(["quarantine", "transaction", "order/line", "proposal", "SKU × qty"], recordRows(current.quarantine, "quarantine_id", (r) => [el("td", { class: "mono", text: `${r.sku} × ${r.quantity}` })]), "records-quarantine")),
      section("Idempotency", table(["request", "status", "action", "transaction", "amount"], idem, "records-idempotency")),
      section("Proposals", table(["proposal", "remedy", "order/line", "replacement", "amount", "status"], proposals, "records-proposals")),
      section("External events", table(["event", "SKU", "stock", "when", "applied"], events, "records-events")));
    const changedStock = productRows.filter((r) => r.dataset.changed === "true").length;
    $("drawer-summary").textContent = [plural((current.refunds || []).length, "refund"), plural((current.exchanges || []).length, "exchange"),
      `${(current.quarantine || []).length} quarantined`, plural(changedStock, "stock change"), plural((s.external_events || []).length, "external event")].join(" · ");
  }

  // ============================================================================================
  // 11. Actions

  function renderAll() {
    syncPickerToSession();
    renderSessionFacts();
    renderScript();
    renderFailureLab();
    renderChat();
    renderAudit();
    renderInspector();
    renderReplayPicker();
    renderTeam(false);
    renderBusy();
  }

  function renderBusy() {
    $("send-btn").disabled = app.busy || !app.session;
    $("reset-btn").disabled = app.busy || !app.session;
    $("scenario-select").disabled = app.busy;
    $("composer-input").setAttribute("aria-busy", app.busy ? "true" : "false");
    renderConfirmAttach();
  }

  const STATUS_NOTE = {
    executed: "already executed: the desk answers with the original result, nothing moves again",
    invalidated: "invalidated: the desk will refuse it",
    superseded: "superseded by a newer proposal: the desk will refuse it",
  };

  /** The composer's confirmation box is offered only for an open proposal, or for the one a scripted
   *  confirm turn names (labelled with its status). A tick binds one proposal: it is cleared when
   *  the latest proposal changes, and after every send. */
  function renderConfirmAttach() {
    const offer = app.session ? confirmOffer() : null;
    const box = $("confirm-attach-box");
    const open = Boolean(offer) && offer.status === "open";
    const scripted = Boolean(offer) && app.confirm.scripted === offer.pid;
    const hidden = !(open || scripted);
    $("confirm-attach").hidden = hidden;
    if (box.checked && (hidden || app.confirm.tickedFor !== offer.pid)) clearConfirmation();
    $("confirm-attach-id").textContent = offer ? offer.pid : "";
    $("confirm-attach-note").textContent = offer && !open ? ` (${STATUS_NOTE[offer.status] || offer.status})` : "";
  }

  function clearConfirmation() {
    $("confirm-attach-box").checked = false;
    app.confirm.tickedFor = null;
    app.confirm.scripted = null;
  }

  function adoptSession(state) {
    app.session = state;
    app.live = null;
    clearConfirmation();
    prefs.set("session", state.session_id);
    const turns = [...new Set(state.action_audit.map((e) => e.turn))];
    showTurn(turns.length ? Math.max(...turns) : null);
    renderAll();
  }

  async function startSession() {
    const choice = $("scenario-select").value;
    $("blank-options").hidden = choice !== BLANK;
    let body;
    if (choice === BLANK) {
      const base = scenarioById($("blank-state-select").value) || app.scenarios[0];
      body = { initial_state: base.initial_state, customer_id: $("customer-select").value || base.customer_id };
    } else {
      body = { case_id: choice };
    }
    try {
      const created = await request("POST", "/api/sessions", body);
      app.notice = null;
      adoptSession(created.state);
      $("progress").textContent = `New session ${created.session_id}: ${choice === BLANK ? "blank conversation" : scenarioById(choice).title}.`;
    } catch (error) {
      showNotice("Could not start the session", error.message);
    }
  }

  /** One turn. `card` is the proposal card this message explicitly confirms (null: no confirmation). */
  async function sendTurn(message, card) {
    if (!app.session || app.busy) return;
    const text = message.trim();
    if (!text) {
      $("progress").textContent = "Type a message first.";
      return;
    }
    const body = { message: text, ...(card ? confirmationFields(card) : { confirm_proposal_id: null }) };
    clearConfirmation(); // an approval covers this one message only
    app.busy = true;
    app.live = { turn: app.session.turns.length + 1, message: text, confirmId: body.confirm_proposal_id, entries: [], finished: false, output: null };
    stopPlayer();
    player.turn = app.live.turn;
    player.cursor = 0;
    player.live = true;
    renderAll();
    $("progress").textContent = `Turn ${app.live.turn} running…`;
    try {
      await streamTurn(app.session.session_id, body, onTurnEvent);
      if (app.live && !app.live.finished) throw new Error("the stream ended before the turn finished");
      $("composer-input").value = "";
    } catch (error) {
      const refused = error.status === 409; // refused before anything ran: busy, or a stale proposal
      showNotice(refused ? "Not sent" : "The turn could not be completed",
        `${error.message}. The page now shows the current state from the server.`, "warn");
      if (refused) app.live = null;
      await reloadState();
      if (refused) showTurn(latestTurn());
    } finally {
      app.busy = false;
      if (app.live) app.live.finished = true;
      renderAll();
      kick();
    }
  }

  function onTurnEvent(type, data) {
    if (type === "start") {
      app.live.turn = data.turn;
      player.turn = data.turn;
    } else if (type === "audit") {
      app.live.entries.push(data);
      if (data.kind === "llm_step" && data.error && data.error.code === "LLM_ERROR") {
        app.llmFailedAt = data.seq;
        refreshHealth();
      }
      if (data.kind === "llm_step") {
        const numbers = data.result || {};
        app.telemetry = app.telemetry || { memory: null };
        app.telemetry.last_step = { agent: data.agent, latency_ms: numbers.latency_ms, error: data.error ? data.error.code : null,
          decode_tok_s: numbers.predicted_ms > 0 ? Math.round((numbers.completion_tokens / numbers.predicted_ms) * 10000) / 10 : null };
        renderTelemetry();
      }
      $("progress").textContent = `Turn ${app.live.turn}: ${liveActivity()}`;
      renderChat();
      renderAudit();
      renderReplayPicker();
      kick();
    } else if (type === "turn") {
      app.session = data.state;
      app.live.finished = true;
      app.live.output = data.output;
      $("progress").textContent = `Turn ${data.turn} finished: ${data.output.decision} (${fmtMs(data.latency_ms)}).`;
      refreshTelemetry();
    } else if (type === "error") {
      app.live.finished = true;
      showNotice("The desk hit an internal error", data.message);
    }
  }

  async function reloadState() {
    if (!app.session) return;
    try {
      app.session = await request("GET", `/api/sessions/${encodeURIComponent(app.session.session_id)}/state`);
    } catch (error) {
      showNotice("Could not reload the session", error.message);
    }
  }

  // The card's button always says just this: a scenario's scripted wording may name another remedy
  // ("Yes, refund ...") and the desk would rightly withhold a grant for a changed request.
  const NEUTRAL_CONFIRMATION = "Yes, I confirm this proposal.";

  function confirmProposal(card, buttons) {
    for (const b of buttons) b.disabled = true; // synchronously: a double click sends one request
    sendTurn(NEUTRAL_CONFIRMATION, card);
  }

  function notNow(buttons) {
    for (const b of buttons) b.disabled = true;
    sendTurn("No, cancel that for now.", null);
  }

  function fillScript(index) {
    const scenario = scenarioById(app.session.case_id);
    const turn = scenario.turns[index];
    $("composer-input").value = turn.message;
    const pid = lastPresentedProposalId();
    const confirms = turn.confirm_proposal === "last" && Boolean(pid);
    clearConfirmation();
    if (confirms) {
      app.confirm.scripted = pid;
      app.confirm.tickedFor = pid;
      $("confirm-attach-box").checked = true;
    }
    renderConfirmAttach();
    $("composer-input").focus();
    $("progress").textContent = turn.confirm_proposal === "last"
      ? pid ? `Filled turn ${index + 1}; it carries the explicit confirmation of ${pid}.` : `Filled turn ${index + 1}; there is no earlier proposal to confirm.`
      : `Filled turn ${index + 1}.`;
  }

  /** A deny's alternative: fill the message box (the customer still sends it and confirms the new proposal). */
  function fillComposer(text) {
    clearConfirmation();
    renderConfirmAttach();
    $("composer-input").value = text;
    $("composer-input").focus();
    $("progress").textContent = "Filled the message box. Press Send to ask for it; nothing moves until you confirm the new proposal.";
  }

  async function resetSession() {
    if (!app.session || app.busy) return;
    try {
      const state = await request("POST", `/api/sessions/${encodeURIComponent(app.session.session_id)}/reset`, { fresh_request_id: false });
      adoptSession(state);
      $("progress").textContent = "State reset to the scenario's initial state.";
    } catch (error) {
      showNotice("Reset failed", error.message);
    }
  }

  async function postFault(body, done) {
    if (!app.session || app.busy) return;
    try {
      const answer = await request("POST", `/api/sessions/${encodeURIComponent(app.session.session_id)}/faults`, body);
      app.session = answer.state;
      $("progress").textContent = done(answer);
      const last = Math.max(...answer.state.action_audit.map((e) => e.turn));
      showTurn(last);
      renderAll();
    } catch (error) {
      showNotice("The failure lab could not apply that", error.message);
    }
  }

  function setStock(event) {
    event.preventDefault();
    const sku = $("stock-sku").value;
    const raw = $("stock-value").value.trim();
    const value = Number(raw);
    if (raw === "" || !Number.isInteger(value) || value < 0) { // an empty field must not mean 0
      showNotice("Stock must be a whole number of 0 or more", `Got "${$("stock-value").value}".`);
      return;
    }
    postFault({ type: "stock_change", sku, new_stock: value },
      (a) => (a.event.applied ? `External stock change: ${a.event.sku} ${a.event.old_stock} → ${a.event.new_stock}.` : `${sku} is not in the catalog; nothing changed.`));
  }

  function armTimeout() {
    postFault({ type: "timeout_after_commit" }, () => "Armed: the next commit will persist and then time out.");
  }

  // ============================================================================================
  // 12. Start-up and event wiring

  function selectTab(which) {
    const team = which === "team";
    $("tab-team").setAttribute("aria-selected", String(team));
    $("tab-audit").setAttribute("aria-selected", String(!team));
    $("tab-team").tabIndex = team ? 0 : -1;
    $("tab-audit").tabIndex = team ? -1 : 0;
    $("panel-team").hidden = !team;
    $("panel-audit").hidden = team;
    if (team) chooseLayout();
    prefs.set("tab", which);
  }

  function setTheater(on) {
    document.body.classList.toggle("theater", on);
    $("theater-btn").setAttribute("aria-pressed", String(on));
    prefs.set("theater", on);
    if (on) selectTab("team");
    window.requestAnimationFrame(() => {
      chooseLayout();
      $("chat-log").scrollTop = $("chat-log").scrollHeight;
    });
  }

  function setDrawer(open) {
    $("drawer-toggle").setAttribute("aria-expanded", String(open));
    $("drawer-body").hidden = !open;
    prefs.set("drawer", open);
  }

  function isTyping(target) {
    return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
  }

  function wireEvents() {
    $("scenario-select").addEventListener("change", startSession);
    $("blank-state-select").addEventListener("change", () => {
      renderCustomerOptions();
      startSession();
    });
    $("customer-select").addEventListener("change", startSession);
    $("reset-btn").addEventListener("click", resetSession);
    $("stock-form").addEventListener("submit", setStock);
    $("arm-timeout").addEventListener("click", armTimeout);
    $("composer").addEventListener("submit", (event) => {
      event.preventDefault();
      const offer = confirmOffer();
      const attach = $("confirm-attach-box").checked && !$("confirm-attach").hidden && offer && offer.card;
      sendTurn($("composer-input").value, attach ? offer.card : null);
    });
    $("confirm-attach-box").addEventListener("change", () => {
      app.confirm.tickedFor = $("confirm-attach-box").checked ? lastPresentedProposalId() : null;
    });
    $("composer-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $("composer").requestSubmit();
      }
    });
    $("tab-team").addEventListener("click", () => selectTab("team"));
    $("tab-audit").addEventListener("click", () => selectTab("audit"));
    document.querySelector('[role="tablist"]').addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const next = $("tab-team").getAttribute("aria-selected") === "true" ? "audit" : "team";
      selectTab(next);
      $(`tab-${next}`).focus();
    });
    $("theater-btn").addEventListener("click", () => setTheater(!document.body.classList.contains("theater")));
    document.addEventListener("keydown", (event) => {
      if ((event.key === "t" || event.key === "T") && !event.metaKey && !event.ctrlKey && !event.altKey && !isTyping(event.target)) {
        setTheater(!document.body.classList.contains("theater"));
      }
    });
    $("panel-team").addEventListener("keydown", (event) => {
      if (isTyping(event.target)) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepReplay(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        stepReplay(1);
      }
    });
    $("replay-play").addEventListener("click", startReplay);
    $("replay-back").addEventListener("click", () => stepReplay(-1));
    $("replay-fwd").addEventListener("click", () => stepReplay(1));
    $("replay-speed").addEventListener("change", () => {
      player.speed = Number($("replay-speed").value) || 1;
    });
    $("replay-turn").addEventListener("change", () => showTurn(Number($("replay-turn").value)));
    $("drawer-toggle").addEventListener("click", () => setDrawer($("drawer-toggle").getAttribute("aria-expanded") !== "true"));
    $("audit-hide-grounding").addEventListener("change", renderAudit);
    $("audit-expand").addEventListener("click", () => {
      for (const e of allEntries()) auditOpen.add(e.seq);
      renderAudit();
    });
    $("audit-collapse").addEventListener("click", () => {
      auditOpen.clear();
      renderAudit();
    });
    if ("ResizeObserver" in window) new ResizeObserver(() => chooseLayout()).observe($("team-graph"));
    reducedMotion.addEventListener("change", () => renderTeam(false));
  }

  async function restoreOrStart() {
    const saved = prefs.get("session", null);
    if (saved) {
      try {
        adoptSession(await request("GET", `/api/sessions/${encodeURIComponent(saved)}/state`));
        return;
      } catch (_) {
        /* the server restarted: start fresh */
      }
    }
    await startSession();
  }

  async function init() {
    buildGraph("compact");
    wireEvents();
    setDrawer(prefs.get("drawer", false));
    selectTab(prefs.get("tab", "team"));
    setTheater(prefs.get("theater", false));
    try {
      [app.meta, app.scenarios] = await Promise.all([request("GET", "/api/meta"), request("GET", "/api/scenarios")]);
    } catch (error) {
      showNotice("The desk server is not reachable", `${error.message}. Start it with scripts/run.sh and reload this page.`);
      return;
    }
    renderScenarioPicker();
    renderCustomerOptions();
    await refreshHealth();
    await restoreOrStart();
    refreshTelemetry();
    window.setInterval(() => {
      if (!document.hidden) refreshHealth();
    }, 3000);
    window.setInterval(() => {
      if (!document.hidden) refreshTelemetry();
    }, 2000);
  }

  // A read-only handle for debugging in the console (and for the E2E tests' assertions).
  window.ReturnsDesk = { app, player, describeEntry, foldTeam, executionSteps, short,
    entryTitle: (e) => entryTitle(e).map((p) => (p instanceof Node ? p.textContent : p)).join(""),
    entryBadges: (e) => entryBadges(e).map((b) => b.textContent) };
  init();
})();
