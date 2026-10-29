"""BusinessTools: the only path from agents to business state (SPEC §4).

Agents choose *which* tool to call; this gateway decides *what happens*:

- Identity is injected, never supplied: the session's customer_id and request_id, the
  grounded target the orchestrator validated (``set_target``) and the turn's confirmation
  grant (``set_grant``). No tool takes a customer_id or a proposal_id from an LLM.
- Every Ops tool starts with the same idempotency check: once the request is completed,
  a call for the same target replays the original transaction (zero effects) and a call
  for anything else is refused with REQUEST_ALREADY_COMPLETED.
- Policy is re-run inside the commit transaction; a proposal that no longer holds is
  invalidated and nothing else changes.
- Every call, including failures, is written to the audit with the effects it committed.

Faults are applied here too (at the tool boundary, never as text for the LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from desk import policy
from desk.audit import AuditLog
from desk.domain import (
    ACTORS,
    BINDING_MISMATCH,
    CONFIRMATION_REQUIRED,
    EXCHANGE,
    EXECUTE_LIMIT,
    EXECUTED,
    INTERNAL,
    INVALID_ARGS,
    INVALIDATED,
    NO_EXCHANGE_AVAILABLE,
    NO_TARGET,
    OPEN,
    PROPOSAL_NOT_OPEN,
    REFUND,
    REMEDIES,
    REMEDY_MISMATCH,
    REPLACEMENT_NOT_REQUESTED,
    REQUEST_ALREADY_COMPLETED,
    SIZE_UNSPECIFIED,
    SKU_NOT_FOUND,
    SUPERSEDED,
    TIMEOUT,
    TOOL_NOT_ALLOWED,
    Eligibility,
    GroundedTarget,
    LineSummary,
    Proposal,
    ToolResult,
    binding_hash,
    make_binding,
)
from desk.faults import FaultPlan, apply_stock_change
from desk.store import CommitRefused, Store, Transaction, record_return

AliasFn = Callable[[str], list[str]]


def default_aliases(product_id: str) -> list[str]:
    return [product_id.lower()]


normalize_ref = policy.normalize_id  # compare IDs the way customers type them: "#ord 7q" == "ORD-7Q"


@dataclass
class _Outcome:
    """A tool's answer plus what the audit must know about it."""

    result: ToolResult
    effects: list[dict[str, Any]] = field(default_factory=list)
    visible_to_agent: bool = True
    audit_data: dict[str, Any] | None = None  # the real data when the agent is shown something else


class BusinessTools:
    def __init__(
        self,
        store: Store,
        customer_id: str,
        request_id: str,
        faults: FaultPlan | None = None,
        audit: AuditLog | None = None,
        alias_fn: AliasFn | None = None,
    ) -> None:
        self.store = store
        self.customer_id = customer_id
        self.request_id = request_id
        self.faults = faults if faults is not None else FaultPlan([])
        self.audit = audit if audit is not None else AuditLog()
        self.alias_fn = alias_fn or default_aliases
        self.turn = 0
        self._target: GroundedTarget | None = None
        self._grant: str | None = None
        self._executed_this_turn = False
        self._fault_turns_done: set[int] = set()
        self._audit_fault_plan()

    # === Turn lifecycle (orchestrator) =====================================================

    def begin_turn(self, turn_no: int) -> list[dict[str, Any]]:
        """Reset per-turn state (target, grant, execute budget) and apply this turn's
        external faults first, as the contract requires. Returns the fault events."""
        self.turn = turn_no
        self.audit.set_turn(turn_no)
        self._target = None
        self._grant = None
        self._executed_this_turn = False
        return self.apply_turn_faults(turn_no)

    def apply_turn_faults(self, turn_no: int) -> list[dict[str, Any]]:
        """Apply stock changes due before ``turn_no`` (once per turn, in list order)."""
        if turn_no in self._fault_turns_done:
            return []
        self._fault_turns_done.add(turn_no)
        events = []
        for fault in self.faults.stock_changes_for(turn_no):
            event = apply_stock_change(self.store, fault["sku"], fault["new_stock"], turn_no)
            self.audit.record(
                kind="fault", agent="external", tool="stock_change", arguments=fault, result=event, visible_to_agent=False
            )
            events.append(event)
        return events

    def set_grant(self, proposal_id: str | None) -> None:
        """The proposal the customer explicitly confirmed this turn (or None)."""
        self._grant = proposal_id
        if proposal_id is not None:
            self.audit.record(kind="grant", agent="orchestrator", tool="set_grant", arguments={"proposal_id": proposal_id})

    def set_target(
        self,
        order_id: str,
        line_id: str,
        intent: str,
        requested_sku: str | None = None,
        *,
        requested_size: str | None = None,
        requested_product: str | None = None,
        grounded_by: str | None = None,
        evidence: str | None = None,
    ) -> GroundedTarget:
        """Inject the grounded line Ops will act on. ``requested_sku`` wins when the
        orchestrator already resolved the size; otherwise ``requested_size`` (e.g. "L",
        "UP") is resolved here against the catalog. ``grounded_by``/``evidence`` only go
        to the audit: whose reading picked this line."""
        remedy = intent.removeprefix("propose_")  # accept the handoff intent names too
        if remedy not in REMEDIES:
            raise ValueError(f"target intent must be one of {REMEDIES}, got {intent!r}")
        self._target = GroundedTarget(order_id, line_id, remedy, requested_sku or None, requested_size,
                                      requested_product, grounded_by, evidence)
        self.audit.record(kind="target", agent="orchestrator", tool="set_target", arguments=self._target.to_dict())
        return self._target

    @property
    def target(self) -> GroundedTarget | None:
        return self._target

    @property
    def grant(self) -> str | None:
        return self._grant

    # === Fault controls for the UI =========================================================

    def inject_stock_change(self, sku: str, new_stock: int) -> dict[str, Any]:
        event = apply_stock_change(self.store, sku, new_stock, before_turn=None)
        self.audit.record(
            kind="fault", agent="external", tool="stock_change", arguments={"sku": sku, "new_stock": new_stock},
            result=event, visible_to_agent=False,
        )  # fmt: skip
        return event

    def arm_timeout(self) -> None:
        self.faults.arm_timeout()
        self.audit.record(kind="fault", agent="external", tool="timeout_after_commit", result={"armed": True}, visible_to_agent=False)

    # === Read helpers (not tools; not audited) =============================================

    def completed_record(self) -> dict[str, Any] | None:
        return self.store.idempotency_record(self.request_id)

    def open_proposal(self) -> dict[str, Any] | None:
        proposal = self.store.open_proposal(self.request_id)
        return proposal.to_dict() if proposal else None

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        proposal = self.store.proposal(proposal_id)
        return proposal.to_dict() if proposal else None

    def proposals(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.store.proposals()]

    def external_events(self) -> list[dict[str, Any]]:
        return self.store.events()

    def final_state(self) -> dict[str, Any]:
        return self.store.export()

    def customer_lines(self) -> list[dict[str, Any]]:
        """The customer's line summaries for code-side grounding (no prompt, so not audited)."""
        return self._line_summaries(self.store.export())

    # === Resolution tools ==================================================================

    def find_order_lines(self, order_ref: str | None = "ALL", actor: str = "resolution") -> ToolResult:
        return self._call(actor, "find_order_lines", {"order_ref": order_ref}, lambda: self._find_order_lines(order_ref))

    def account_card(self, actor: str = "orchestrator") -> ToolResult:
        """All of the customer's lines plus the catalog sizes of those products (prefetch)."""
        return self._call(actor, "prefetch_account_card", {}, self._account_card)

    # === Operations tools ==================================================================

    def check_eligibility(self, order_id: str | None = None, line_id: str | None = None,
                          actor: str = "orchestrator") -> ToolResult:
        """Read-only eligibility of a line of this customer (default: the injected target). The
        orchestrator uses it to BLOCK: a line the customer named by its IDs that policy refuses is
        denied even when the agent only asked a question (SPEC §6.0)."""
        arguments = {"order_id": order_id, "line_id": line_id} if order_id is not None else {}
        return self._call(actor, "check_eligibility", arguments, lambda: self._check_eligibility(order_id, line_id))

    def check_proposal(self, remedy: str, replacement_sku: str | None = None, actor: str = "orchestrator") -> ToolResult:
        """Read-only dry run of create_proposal: the same answer, but nothing is stored. Used when
        Operations ended without a decision: a refusal may then be reported, a proposal never made."""
        arguments = {"remedy": remedy, "replacement_sku": replacement_sku}
        return self._call(actor, "check_proposal", arguments, lambda: self._check_proposal(remedy, replacement_sku or None))

    def list_exchange_options(self, actor: str = "operations") -> ToolResult:
        return self._call(actor, "list_exchange_options", {}, self._list_exchange_options)

    def create_proposal(self, remedy: str, replacement_sku: str | None = None, actor: str = "operations") -> ToolResult:
        arguments = {"remedy": remedy, "replacement_sku": replacement_sku}
        return self._call(actor, "create_proposal", arguments, lambda: self._create_proposal(remedy, replacement_sku or None))

    def execute_proposal(self, actor: str = "operations") -> ToolResult:
        return self._call(actor, "execute_proposal", {}, self._execute_proposal)

    def get_request_status(self, actor: str = "operations") -> ToolResult:
        return self._call(actor, "get_request_status", {}, self._get_request_status)

    # === Plumbing ==========================================================================

    def _call(self, actor: str, tool: str, arguments: dict[str, Any], body: Callable[[], ToolResult | _Outcome]) -> ToolResult:
        """Run one tool body and audit it. A tool always answers: an unexpected exception
        (the store already rolled back) becomes an INTERNAL failure in the audit."""
        if actor not in ACTORS:
            raise ValueError(f"unknown actor {actor!r}")
        try:
            outcome = body()
        except Exception as exc:  # noqa: BLE001 - converted to a typed failure, never swallowed silently
            outcome = ToolResult.failure(INTERNAL, f"{type(exc).__name__}: {exc}")
        if isinstance(outcome, ToolResult):
            outcome = _Outcome(outcome)
        result = outcome.result
        self.audit.record(
            kind="tool",
            agent=actor,
            tool=tool,
            arguments=arguments,
            result=outcome.audit_data if outcome.audit_data is not None else result.data,
            error=result.error,
            committed_effects=outcome.effects,
            visible_to_agent=outcome.visible_to_agent,
        )
        return result

    def _short_circuit(self) -> ToolResult | None:
        """Idempotency first: None while the request is open, else replay or refusal."""
        record = self.store.idempotency_record(self.request_id)
        if record is None:
            return None
        if self._matches_completed(record):
            return ToolResult.success({"replay": True, "transaction": record})
        return ToolResult.failure(
            REQUEST_ALREADY_COMPLETED,
            "this request was already completed for a different item or remedy",
            {"transaction_id": record.get("transaction_id")},
        )

    def _matches_completed(self, record: dict[str, Any]) -> bool:
        """Is this call about the transaction already completed? A grant matches by
        proposal_id or identical binding; a target by line, remedy and replacement.
        Fields absent from a record (e.g. one supplied in the input) do not mismatch."""
        if self._grant is not None:
            if record.get("proposal_id") == self._grant:
                return True
            proposal = self.store.proposal(self._grant)
            return proposal is not None and _record_binding_hash(record) == proposal.binding_hash
        if self._target is None:
            return True

        def same(field_name: str, value: Any) -> bool:
            return field_name not in record or record[field_name] == value

        target = self._target
        if not (same("order_id", target.order_id) and same("line_id", target.line_id) and same("action", target.intent)):
            return False
        if target.intent == EXCHANGE and (target.requested_sku or target.requested_size or target.requested_product):
            requested, _ = self._requested_replacement(self.store.export())
            return same("replacement_sku", requested)
        return True

    def _target_line(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        assert self._target is not None
        order = policy.find_customer_order(doc, self.customer_id, self._target.order_id)
        return policy.find_line(order, self._target.line_id) if order else None

    def _requested_replacement(self, doc: dict[str, Any]) -> tuple[str | None, str | None]:
        """The replacement the customer asked for: (sku, None), (None, code) or (None, None)."""
        target = self._target
        assert target is not None
        if target.requested_sku:
            return target.requested_sku, None
        line = self._target_line(doc)
        original_sku = line.get("sku") if line else None
        return policy.resolve_size(target.requested_size, original_sku, doc.get("products", []), target.requested_product)

    def _audit_fault_plan(self) -> None:
        """Make the fault plan visible in the trace before anything happens."""
        for skipped in self.faults.skipped:
            fault = skipped["fault"]
            tool = str(fault.get("type")) if isinstance(fault, dict) else "invalid_fault"
            self.audit.record(
                kind="fault", agent="external", tool=tool, arguments={"index": skipped["index"], "fault": fault},
                error={"code": skipped["reason"], "message": "never applied"}, visible_to_agent=False,
            )  # fmt: skip
        if self.faults.timeout_armed:
            self.audit.record(kind="fault", agent="external", tool="timeout_after_commit", result={"armed": True}, visible_to_agent=False)

    # === Tool bodies =======================================================================

    def _line_summaries(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        summaries = []
        for order in doc.get("orders", []):
            if order.get("customer_id") != self.customer_id:
                continue
            for line in order.get("lines", []):
                product = policy.find_product(doc, line.get("sku")) or {}
                product_id = product.get("product_id")
                delivered = policy.parse_date(order.get("delivered_on"))
                summary = LineSummary(
                    order_id=order.get("order_id"),
                    line_id=line.get("line_id"),
                    sku=line.get("sku"),
                    product_id=product_id,
                    size=product.get("size"),
                    aliases=list(self.alias_fn(product_id)) if product_id else [],
                    paid_cents=line.get("paid_cents"),
                    delivered_on=delivered.isoformat() if delivered else None,
                )
                summaries.append(summary.to_dict())
        return summaries

    def _find_order_lines(self, order_ref: Any) -> ToolResult:
        lines = self._line_summaries(self.store.export())
        if order_ref is None or normalize_ref(order_ref) in ("", "all", "*"):
            return ToolResult.success({"matches": lines, "count": len(lines)})
        wanted = normalize_ref(order_ref)
        matches = [line for line in lines if normalize_ref(line["order_id"]) == wanted]
        data: dict[str, Any] = {"matches": matches, "count": len(matches)}
        if not matches:
            # Same answer for "someone else's order" and "no such order": never leak.
            data["order_not_on_account"] = True
        return ToolResult.success(data)

    def _account_card(self) -> ToolResult:
        doc = self.store.export()
        lines = self._line_summaries(doc)
        product_ids = {line["product_id"] for line in lines if line["product_id"] is not None}
        catalog = [
            {"sku": p.get("sku"), "product_id": p.get("product_id"), "size": p.get("size")}
            for p in doc.get("products", [])
            if p.get("product_id") in product_ids
        ]
        return ToolResult.success({"lines": lines, "count": len(lines), "catalog": catalog})

    def _check_eligibility(self, order_id: str | None, line_id: str | None) -> ToolResult:
        done = self._short_circuit()
        if done is not None and not done.ok:
            return done
        if order_id is None:
            if self._target is None:
                return done or ToolResult.failure(NO_TARGET, "no grounded target has been handed off")
            order_id, line_id = self._target.order_id, self._target.line_id
        report = policy.eligibility(self.store.export(), self.customer_id, order_id, line_id)
        data = report.to_dict()
        if done is not None:
            data.update(completed_by_this_request=True, replay=True, transaction=done.data["transaction"])
        return ToolResult.success(data)

    def _list_exchange_options(self) -> ToolResult:
        done = self._short_circuit()
        if done is not None:
            return done
        if self._target is None:
            return ToolResult.failure(NO_TARGET, "no grounded target has been handed off")
        doc = self.store.export()
        line = self._target_line(doc)
        if line is None:
            report = policy.eligibility(doc, self.customer_id, self._target.order_id, self._target.line_id)
            return ToolResult.success({"original_sku": None, "options": [], "valid_count": 0, "reasons": report.reasons})
        original_sku = line.get("sku")
        options = policy.options_for_sku(doc, original_sku)
        requested_sku, requested_code = self._requested_replacement(doc)
        return ToolResult.success(
            {
                "original_sku": original_sku,
                "options": [option.to_dict() for option in options],
                "valid_count": len(policy.valid_options(options)),
                "requested_sku": requested_sku,
                "requested_code": requested_code,
                "reasons": [] if policy.find_product(doc, original_sku) else [SKU_NOT_FOUND],
            }
        )

    def _create_proposal(self, remedy: str, replacement_sku: str | None) -> ToolResult:
        answer = self._proposal_answer(remedy, replacement_sku)
        if isinstance(answer, ToolResult):
            return answer
        report, chosen, amount, doc = answer
        return self._store_proposal(report, remedy, chosen, amount, doc)

    def _check_proposal(self, remedy: str, replacement_sku: str | None) -> ToolResult:
        answer = self._proposal_answer(remedy, replacement_sku)
        if isinstance(answer, ToolResult):
            return answer
        _report, chosen, amount, _doc = answer
        return ToolResult.success({"would_propose": {"remedy": remedy, "replacement_sku": chosen, "amount_cents": amount},
                                   "stored": False})

    def _proposal_answer(
        self, remedy: str, replacement_sku: str | None
    ) -> ToolResult | tuple[Eligibility, str | None, int, dict[str, Any]]:
        """Every check create_proposal makes, in order: a refusal or error, or what it would bind."""
        done = self._short_circuit()
        if done is not None:
            return done
        if self._grant is not None:
            # A confirmation turn executes what was confirmed; it never proposes something new.
            return ToolResult.failure(TOOL_NOT_ALLOWED, "no new proposals on a confirmation turn; execute or report")
        target = self._target
        if target is None:
            return ToolResult.failure(NO_TARGET, "no grounded target has been handed off")
        if remedy not in REMEDIES:
            return ToolResult.failure(INVALID_ARGS, f"remedy must be one of {list(REMEDIES)}")
        if remedy != target.intent:
            return ToolResult.failure(REMEDY_MISMATCH, f"the customer asked for a {target.intent}, not a {remedy}")

        doc = self.store.export()
        report = policy.eligibility(doc, self.customer_id, target.order_id, target.line_id)
        if not report.eligible:
            return _refusal(report.reasons, "this line is not eligible", {"eligibility": report.to_dict()})
        line = self._target_line(doc)
        if remedy == REFUND:
            if replacement_sku is not None:
                return ToolResult.failure(INVALID_ARGS, "replacement_sku only applies to exchanges")
            return report, None, policy.refund_amount(line), doc
        chosen, refused = self._choose_replacement(doc, report.sku, replacement_sku)
        if refused is not None:
            return refused
        return report, chosen, 0, doc

    def _choose_replacement(
        self, doc: dict[str, Any], original_sku: str, replacement_sku: str | None
    ) -> tuple[str | None, ToolResult | None]:
        """Code picks the replacement from what the customer asked for; the agent's
        ``replacement_sku`` must agree with it (it can never choose a size for them)."""
        alternatives = [{"sku": o.sku, "size": o.size} for o in policy.valid_options(policy.options_for_sku(doc, original_sku))]
        details = {"alternatives": alternatives}
        requested, code = self._requested_replacement(doc)
        if code is not None:
            return None, _refusal([code], "the requested replacement is not possible", details)
        if requested is None:  # (an unknown original SKU already came back as SKU_NOT_FOUND)
            if not alternatives:
                return None, _refusal([NO_EXCHANGE_AVAILABLE], "no other size can be exchanged", details)
            if len(alternatives) > 1:
                return None, ToolResult.failure(SIZE_UNSPECIFIED, "the customer has not chosen a size", details)
            requested = alternatives[0]["sku"]  # exactly one valid size: propose it (CLARIFICATIONS #5)
        if replacement_sku is not None and replacement_sku != requested:
            return None, ToolResult.failure(REPLACEMENT_NOT_REQUESTED, f"the customer's request maps to {requested}", details)
        reasons = policy.validate_replacement(doc, original_sku, requested)
        if reasons:
            return None, _refusal(reasons, f"{requested} is not a valid exchange", {**details, "requested_sku": requested})
        return requested, None

    def _store_proposal(
        self, report: Eligibility, remedy: str, replacement_sku: str | None, amount_cents: int, doc: dict[str, Any]
    ) -> ToolResult:
        """Identical binding -> the same open proposal; a different one supersedes it.

        Eligibility was checked just before, outside this transaction. That is safe: a
        proposal changes no business state, and execute re-runs every rule inside its commit."""
        target = self._target
        assert target is not None
        binding = make_binding(
            customer_id=self.customer_id,
            order_id=target.order_id,
            line_id=target.line_id,
            remedy=remedy,
            amount_cents=amount_cents,
            replacement_sku=replacement_sku,
            request_id=self.request_id,
        )
        digest = binding_hash(binding)
        with self.store.transaction() as tx:
            current = tx.open_proposal(self.request_id)
            if current is not None and current.binding_hash == digest:
                return ToolResult.success({"proposal": current.to_dict(), "reused": True, "superseded": None})
            if current is not None:
                tx.put_proposal(current.with_status(SUPERSEDED))
            proposal = Proposal(
                proposal_id=tx.next_id("P"),
                request_id=self.request_id,
                customer_id=self.customer_id,
                order_id=target.order_id,
                line_id=target.line_id,
                remedy=remedy,
                original_sku=report.sku,
                replacement_sku=replacement_sku,
                amount_cents=amount_cents,
                currency=doc.get("currency", "USD"),
                eligibility={
                    "delivered_on": report.delivered_on,
                    "as_of": report.as_of,
                    "days_since_delivery": report.days_since_delivery,
                    "window_days": report.window_days,
                },
                binding_hash=digest,
                status=OPEN,
                created_turn=self.turn,
            )
            tx.put_proposal(proposal)
        superseded = current.proposal_id if current is not None else None
        return ToolResult.success({"proposal": proposal.to_dict(), "reused": False, "superseded": superseded})

    def _execute_proposal(self) -> ToolResult | _Outcome:
        done = self._short_circuit()
        if done is not None:
            return done
        if self._grant is None:
            return ToolResult.failure(CONFIRMATION_REQUIRED, "the customer has not explicitly confirmed a proposal")
        if self._executed_this_turn:
            return ToolResult.failure(EXECUTE_LIMIT, "already executed once this turn; check get_request_status")
        self._executed_this_turn = True
        proposal_id = self._grant
        try:
            with self.store.transaction() as tx:
                proposal = tx.proposal(proposal_id)
                refused = self._refuse_execution(tx, proposal_id, proposal)
                if refused is None:
                    record = record_return(tx, proposal, tx.doc["as_of"])
                    tx.put_proposal(proposal.with_status(EXECUTED))
                effects = list(tx.effects)
        except CommitRefused as guard:
            self._invalidate(proposal_id)
            return _refusal([guard.code], guard.message, {"proposal_status": INVALIDATED})
        if refused is not None:
            return refused
        data = {"replay": False, "proposal_id": proposal_id, "status": EXECUTED, "transaction": record}
        # The commit is durable at this point. The simulated timeout only loses the answer.
        if self.faults.consume_timeout():
            timeout = ToolResult.failure(TIMEOUT, "commit response lost (timeout)")
            return _Outcome(timeout, effects=effects, visible_to_agent=False, audit_data=data)
        return _Outcome(ToolResult.success(data), effects=effects)

    def _refuse_execution(self, tx: Transaction, proposal_id: str, proposal: Proposal | None) -> ToolResult | None:
        """All checks between "confirmed" and "committed", inside the commit's lock."""
        if proposal is None or proposal.status != OPEN:
            status = proposal.status if proposal else None
            return ToolResult.failure(
                PROPOSAL_NOT_OPEN, f"proposal {proposal_id} is not open", {"proposal_id": proposal_id, "status": status}
            )
        # Recompute the hash with THIS session's identity: a proposal bound to another
        # customer or request, or one whose stored facts were altered, cannot execute.
        expected = make_binding(**{**proposal.binding(), "customer_id": self.customer_id, "request_id": self.request_id})
        if binding_hash(expected) != proposal.binding_hash:
            return ToolResult.failure(BINDING_MISMATCH, "the confirmed proposal does not match this request")
        reasons = policy.eligibility(tx.doc, proposal.customer_id, proposal.order_id, proposal.line_id).reasons
        if proposal.remedy == EXCHANGE:
            reasons = reasons + policy.validate_replacement(tx.doc, proposal.original_sku, proposal.replacement_sku)
        if not reasons:
            return None
        tx.put_proposal(proposal.with_status(INVALIDATED))
        alternatives = []
        if proposal.remedy == EXCHANGE:
            options = policy.valid_options(policy.options_for_sku(tx.doc, proposal.original_sku))
            alternatives = [{"sku": o.sku, "size": o.size} for o in options]
        details = {"proposal_id": proposal_id, "proposal_status": INVALIDATED, "alternatives": alternatives}
        return _refusal(reasons, "the proposal no longer holds", details)

    def _invalidate(self, proposal_id: str) -> None:
        """After a commit guard fired: the rollback left the (checked-open) proposal open,
        so invalidate it in a transaction of its own."""
        with self.store.transaction() as tx:
            tx.put_proposal(tx.proposal(proposal_id).with_status(INVALIDATED))

    def _get_request_status(self) -> ToolResult:
        record = self.store.idempotency_record(self.request_id)
        if record is None:
            return ToolResult.success({"request_id": self.request_id, "status": "not_found"})
        return ToolResult.success({"request_id": self.request_id, "status": "completed", "record": record})


def _refusal(reasons: list[str], message: str, details: dict[str, Any]) -> ToolResult:
    """A business "no": the first reason is the code, all reasons travel in details."""
    return ToolResult.failure(reasons[0], f"{message}: {', '.join(reasons)}", {"reasons": list(reasons), **details})


def _record_binding_hash(record: dict[str, Any]) -> str:
    return binding_hash(
        make_binding(
            customer_id=record.get("customer_id"),
            order_id=record.get("order_id"),
            line_id=record.get("line_id"),
            remedy=record.get("action"),
            amount_cents=record.get("amount_cents"),
            replacement_sku=record.get("replacement_sku"),
            request_id=record.get("request_id"),
        )
    )
