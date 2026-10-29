"""SQLite-backed business store with an exact-shape export and atomic commits.

Two promises drive the design (SPEC §2):

1. **Exact shape.** ``final_state`` is the input document itself, deep-copied once and
   then changed in place (stock, ``line.returned``) or appended to (returns, refunds,
   exchanges, quarantine, idempotency). Unknown fields and key order survive untouched,
   so ``json.dumps(Store(s).export()) == json.dumps(s)`` for an untouched state.
   Proposals and external events are NOT business state: they live in their own tables
   and are returned separately.

2. **All or nothing.** A business action is one ``BEGIN IMMEDIATE … COMMIT``: the
   idempotency row and the returned-line row (both primary keys, so a duplicate refund
   is physically impossible), ledger rows and the new document version. The in-memory
   document is swapped only after COMMIT succeeds; any exception rolls everything back.

IDs are deterministic per-session counters (``P-0001``, ``TX-0001``, …) stored in the
same transaction, so a rolled-back attempt never burns an ID.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from desk import policy
from desk.domain import (
    ALREADY_RETURNED,
    LINE_NOT_FOUND,
    OPEN,
    OUT_OF_STOCK,
    REFUND,
    REQUEST_ALREADY_COMPLETED,
    Proposal,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS state_doc (version INTEGER PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (request_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS returned_lines (
    customer_id TEXT NOT NULL, order_id TEXT NOT NULL, line_id TEXT NOT NULL,
    PRIMARY KEY (customer_id, order_id, line_id));
CREATE TABLE IF NOT EXISTS proposals (proposal_id TEXT PRIMARY KEY, json TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ledger (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL, type TEXT NOT NULL, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""

# Record collection -> (ID prefix, ID field). Counters start after any records already in
# the input so new IDs do not repeat the numbering of existing ones.
RECORD_IDS = {
    "returns": ("RET", "return_id"),
    "refunds": ("RF", "refund_id"),
    "exchanges": ("EX", "exchange_id"),
    "quarantine": ("Q", "quarantine_id"),
}


class CommitRefused(Exception):
    """A last-line guard inside the commit refused the write; the transaction rolls back."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Transaction:
    """Writes allowed inside one ``Store.transaction()``. ``doc`` is a private working
    copy; it becomes the store's document only if the whole transaction commits."""

    def __init__(self, conn: sqlite3.Connection, doc: dict[str, Any]) -> None:
        self._conn = conn
        self.doc = doc
        self.doc_changed = False
        self.effects: list[dict[str, Any]] = []

    # --- reads inside the lock ---------------------------------------------------------

    def proposal(self, proposal_id: str) -> Proposal | None:
        return _proposal(self._conn, proposal_id)

    def open_proposal(self, request_id: str) -> Proposal | None:
        return _open_proposal(self._conn, request_id)

    # --- IDs, proposals, events (not part of final_state) -------------------------------

    def next_id(self, prefix: str) -> str:
        row = self._conn.execute("SELECT value FROM counters WHERE name = ?", (prefix,)).fetchone()
        value = (row[0] if row else 0) + 1
        self._conn.execute(
            "INSERT INTO counters (name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
            (prefix, value),
        )
        return f"{prefix}-{value:04d}"

    def put_proposal(self, proposal: Proposal) -> None:
        # Upsert (not REPLACE) keeps the rowid, so proposals stay in creation order.
        self._conn.execute(
            "INSERT INTO proposals (proposal_id, json, status) VALUES (?, ?, ?) "
            "ON CONFLICT(proposal_id) DO UPDATE SET json = excluded.json, status = excluded.status",
            (proposal.proposal_id, json.dumps(proposal.to_dict()), proposal.status),
        )

    def add_event(self, event: dict[str, Any]) -> None:
        self._conn.execute("INSERT INTO events (event_id, json) VALUES (?, ?)", (event["event_id"], json.dumps(event)))

    # --- business writes (these change final_state) --------------------------------------

    def set_stock(self, sku: str, new_stock: Any) -> tuple[bool, Any]:
        """Plain assignment used by external stock faults. Returns (found, old_stock)."""
        product = policy.find_product(self.doc, sku)
        if product is None:
            return False, None
        old_stock = product.get("stock")
        product["stock"] = new_stock
        self.doc_changed = True
        return True, old_stock

    def decrement_stock(self, sku: str, transaction_id: str) -> None:
        """Take one replacement unit. Never lets stock go below zero, whatever validated earlier."""
        product = policy.find_product(self.doc, sku)
        stock = product.get("stock") if product else None
        if not (policy.is_int(stock) and stock >= 1):
            raise CommitRefused(OUT_OF_STOCK, f"{sku} has no sellable stock")
        product["stock"] = stock - 1
        self._effect(transaction_id, {"type": "stock", "sku": sku, "old_stock": stock, "new_stock": stock - 1})

    def mark_returned(self, customer_id: str, order_id: str, line_id: str, transaction_id: str) -> None:
        # Keyed by customer too: the schema does not promise globally unique order ids, and another
        # customer's returned "O-1/L1" must not block this customer's "O-1/L1" (review B5).
        try:
            self._conn.execute("INSERT INTO returned_lines (customer_id, order_id, line_id) VALUES (?, ?, ?)",
                               (customer_id, order_id, line_id))
        except sqlite3.IntegrityError as exc:
            raise CommitRefused(ALREADY_RETURNED, f"{order_id}/{line_id} was already returned") from exc
        order = policy.find_customer_order(self.doc, customer_id, order_id)
        line = policy.find_line(order, line_id) if order else None
        if line is None:
            raise CommitRefused(LINE_NOT_FOUND, f"{order_id}/{line_id} is not on this account")
        line["returned"] = True
        self._effect(transaction_id, {"type": "line_returned", "order_id": order_id, "line_id": line_id})

    def append_record(self, collection: str, record: dict[str, Any]) -> None:
        # setdefault only matters for malformed inputs that omit a collection; the
        # contract guarantees all of them, so a valid input never gains a key.
        self.doc.setdefault(collection, []).append(record)
        _, id_field = RECORD_IDS[collection]
        effect = {"type": "record", "collection": collection, "id": record[id_field], "amount_cents": record["amount_cents"]}
        self._effect(record["transaction_id"], effect)

    def insert_idempotency(self, request_id: str, record: dict[str, Any]) -> None:
        try:
            self._conn.execute("INSERT INTO idempotency (request_id, json) VALUES (?, ?)", (request_id, json.dumps(record)))
        except sqlite3.IntegrityError as exc:
            raise CommitRefused(REQUEST_ALREADY_COMPLETED, f"{request_id} already has a completed record") from exc
        self.doc.setdefault("idempotency", {})[request_id] = record
        self._effect(record["transaction_id"], {"type": "idempotency", "request_id": request_id})

    def _effect(self, transaction_id: str, effect: dict[str, Any]) -> None:
        self.doc_changed = True
        self.effects.append(effect)
        self._conn.execute(
            "INSERT INTO ledger (transaction_id, type, json) VALUES (?, ?, ?)",
            (transaction_id, effect["type"], json.dumps(effect)),
        )


class Store:
    """Pass ``initial_state`` to start a session; pass ``None`` to reopen a store file."""

    def __init__(self, initial_state: dict[str, Any] | None, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)
        existing = self._conn.execute("SELECT json FROM state_doc ORDER BY version DESC LIMIT 1").fetchone()
        if initial_state is None:
            if existing is None:
                raise ValueError(f"no store to reopen at {path!r}")
            self._doc = json.loads(existing[0])
            return
        if existing is not None:
            raise ValueError(f"store at {path!r} already holds a session; use Store.open()")
        policy.parse_as_of(initial_state)  # fail at load, not in the middle of a turn
        self._doc = copy.deepcopy(initial_state)
        self._seed()

    @classmethod
    def open(cls, path: str) -> Store:
        return cls(None, path)

    def _seed(self) -> None:
        """Version 1 = the input; tables mirror what the input already says."""
        doc = self._doc
        self._conn.execute("BEGIN IMMEDIATE")
        self._conn.execute("INSERT INTO state_doc (version, json) VALUES (1, ?)", (json.dumps(doc),))
        records = doc.get("idempotency")
        records = records if isinstance(records, dict) else {}
        for request_id, record in records.items():
            if isinstance(record, dict) and record.get("status") == "completed":
                self._conn.execute(
                    "INSERT INTO idempotency (request_id, json) VALUES (?, ?)", (request_id, json.dumps(record))
                )
        for order in doc.get("orders", []):
            for line in order.get("lines", []):
                if line.get("returned") is not False:  # same fail-closed reading as policy
                    self._conn.execute(
                        "INSERT OR IGNORE INTO returned_lines (customer_id, order_id, line_id) VALUES (?, ?, ?)",
                        (str(order.get("customer_id")), order.get("order_id"), line.get("line_id")),
                    )
        seeds = {prefix: len(doc.get(collection) or []) for collection, (prefix, _) in RECORD_IDS.items()}
        seeds["TX"] = len(records)
        for prefix, value in seeds.items():
            self._conn.execute("INSERT INTO counters (name, value) VALUES (?, ?)", (prefix, value))
        self._conn.execute("COMMIT")

    # --- transactions ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """One atomic unit of work. The document is replaced only after COMMIT succeeds."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            tx = Transaction(self._conn, copy.deepcopy(self._doc))
            try:
                yield tx
                if tx.doc_changed:
                    self._conn.execute(
                        "INSERT INTO state_doc (version, json) VALUES (?, ?)",
                        (self.version() + 1, json.dumps(tx.doc)),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            if tx.doc_changed:
                self._doc = tx.doc

    # --- reads (always copies; callers can never mutate committed state) ---------------

    @property
    def as_of(self) -> str:
        return self._doc["as_of"]

    def export(self) -> dict[str, Any]:
        """The current business document: final_state for the adapter, reads for policy."""
        with self._lock:
            return copy.deepcopy(self._doc)

    def version(self) -> int:
        return self._conn.execute("SELECT MAX(version) FROM state_doc").fetchone()[0]

    def idempotency_record(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            return _idempotency_record(self._conn, request_id)

    def proposal(self, proposal_id: str) -> Proposal | None:
        with self._lock:
            return _proposal(self._conn, proposal_id)

    def open_proposal(self, request_id: str) -> Proposal | None:
        with self._lock:
            return _open_proposal(self._conn, request_id)

    def proposals(self) -> list[Proposal]:
        with self._lock:
            rows = self._conn.execute("SELECT json FROM proposals ORDER BY rowid").fetchall()
        return [Proposal.from_dict(json.loads(row[0])) for row in rows]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT json FROM events ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def ledger(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT seq, transaction_id, json FROM ledger ORDER BY seq").fetchall()
        return [{"seq": seq, "transaction_id": tx_id, **json.loads(body)} for seq, tx_id, body in rows]

    def returned_lines(self) -> list[tuple[str, str, str]]:
        """(customer_id, order_id, line_id) of every line that may not be returned again."""
        with self._lock:
            return [tuple(row) for row in self._conn.execute("SELECT customer_id, order_id, line_id FROM returned_lines")]

    def close(self) -> None:
        self._conn.close()


# --- record shapes (SPEC §2) -----------------------------------------------------------


def common_fields(proposal: Proposal, transaction_id: str, business_date: str) -> dict[str, Any]:
    """Fields every transaction record carries, so any record alone identifies the
    customer, order line, request, proposal and replacement (CONTRACT)."""
    return {
        "transaction_id": transaction_id,
        "request_id": proposal.request_id,
        "proposal_id": proposal.proposal_id,
        "customer_id": proposal.customer_id,
        "order_id": proposal.order_id,
        "line_id": proposal.line_id,
        "sku": proposal.original_sku,
        "replacement_sku": proposal.replacement_sku,
        "amount_cents": proposal.amount_cents,
        "currency": proposal.currency,
        "business_date": business_date,
        "status": "completed",
    }


def record_return(tx: Transaction, proposal: Proposal, business_date: str) -> dict[str, Any]:
    """Write every record of one executed proposal inside ``tx``; returns the idempotency
    record (which is also "the transaction" shown to agents and in turn outputs).

    The returned original goes to quarantine and its SKU's stock is never touched; an
    exchange takes exactly one unit of the replacement SKU.
    """
    transaction_id = tx.next_id("TX")
    return_id = tx.next_id("RET")
    common = common_fields(proposal, transaction_id, business_date)
    tx.mark_returned(proposal.customer_id, proposal.order_id, proposal.line_id, transaction_id)
    tx.append_record("returns", {"return_id": return_id, "remedy": proposal.remedy, "disposition": "quarantine", **common})
    result: dict[str, Any] = {"transaction_id": transaction_id, "return_id": return_id}
    if proposal.remedy == REFUND:
        refund_id = tx.next_id("RF")
        tx.append_record("refunds", {"refund_id": refund_id, "return_id": return_id, **common})
        result["refund_id"] = refund_id
    else:
        tx.decrement_stock(proposal.replacement_sku, transaction_id)
        exchange_id = tx.next_id("EX")
        exchange = {
            "exchange_id": exchange_id,
            "return_id": return_id,
            "original_sku": proposal.original_sku,
            "replacement_sku": proposal.replacement_sku,
            **common,
        }
        tx.append_record("exchanges", exchange)
        result["exchange_id"] = exchange_id
    quarantine_id = tx.next_id("Q")
    quarantine = {"quarantine_id": quarantine_id, "return_id": return_id, "quantity": 1, "reason": "customer_return", **common}
    tx.append_record("quarantine", quarantine)
    result.update(quarantine_id=quarantine_id, amount_cents=proposal.amount_cents, replacement_sku=proposal.replacement_sku)
    record = {"status": "completed", "action": proposal.remedy, **common, "result": result}
    tx.insert_idempotency(proposal.request_id, record)
    return record


# --- shared SQL reads ------------------------------------------------------------------


def _idempotency_record(conn: sqlite3.Connection, request_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT json FROM idempotency WHERE request_id = ?", (request_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _proposal(conn: sqlite3.Connection, proposal_id: str) -> Proposal | None:
    row = conn.execute("SELECT json FROM proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
    return Proposal.from_dict(json.loads(row[0])) if row else None


def _open_proposal(conn: sqlite3.Connection, request_id: str) -> Proposal | None:
    rows = conn.execute("SELECT json FROM proposals WHERE status = ? ORDER BY rowid DESC", (OPEN,)).fetchall()
    for (body,) in rows:
        proposal = Proposal.from_dict(json.loads(body))
        if proposal.request_id == request_id:
            return proposal
    return None
