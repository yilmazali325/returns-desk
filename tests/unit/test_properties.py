"""Property tests: random states and random tool-call sequences must never break the
business invariants, whatever order agents (or a confused LLM) call the tools in.

``DeskMachine`` is a Hypothesis state machine: each rule is one thing an agent, the
orchestrator or the outside world can do; Hypothesis runs them in random orders and
shrinks any failure to a minimal sequence. Two requests share one store, so
cross-request duplicates are exercised too. After every step we check: stock is never
negative and equals a simple model (faults assign, exchanges take one), each line is
returned at most once, refunds equal the refunded lines' paid_cents, replays and refusals
change nothing, nothing changes without a grant, every record names its customer, line,
request and proposal, and the exported keys are exactly the input keys.
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from typing import Any

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from desk import policy
from desk.audit import AuditLog
from desk.domain import INTERNAL, TIMEOUT, ToolResult
from desk.faults import FaultPlan
from desk.store import Store
from desk.tools import BusinessTools

CUSTOMER = "C7"
SIZES = ["S", "M", "L", "XL"]
COMMON = ("transaction_id", "request_id", "proposal_id", "customer_id", "order_id", "line_id", "sku",
          "amount_cents", "currency", "business_date", "status")  # fmt: skip
RECORD_LISTS = ("returns", "refunds", "exchanges", "quarantine")


# --- random states ---------------------------------------------------------------------------


@st.composite
def states(draw: st.DrawFn) -> dict[str, Any]:
    as_of = date(2026, 3, 15) + timedelta(days=draw(st.integers(-800, 800)))
    products = []
    for product_id in ("HOODIE", "CAP"):
        base_price = draw(st.sampled_from([2000, 5000]))
        for size in draw(st.lists(st.sampled_from(SIZES), min_size=1, max_size=4, unique=True)):
            price = draw(st.sampled_from([base_price, base_price, base_price + 500]))
            stock = draw(st.integers(0, 3))
            products.append({"sku": f"{product_id}-{size}", "product_id": product_id, "size": size, "price_cents": price, "stock": stock})
    skus = [p["sku"] for p in products] + ["GONE-1"]
    orders = []
    for order_no in range(draw(st.integers(1, 3))):
        # The first order's first line is always returnable so most runs reach commits;
        # everything else is random and exercises every refusal path.
        happy = order_no == 0
        lines = [
            {
                "line_id": f"L{line_no + 1}",
                "sku": draw(st.sampled_from(skus)),
                "paid_cents": draw(st.integers(0, 6000)),
                "unopened": True if happy and line_no == 0 else draw(st.sampled_from([True, True, False, None])),
                "final_sale": False if happy and line_no == 0 else draw(st.sampled_from([False, False, True])),
                "returned": False if happy and line_no == 0 else draw(st.sampled_from([False, False, True])),
            }
            for line_no in range(draw(st.integers(1, 3)))
        ]
        days = draw(st.integers(0, 30) if happy else st.one_of(st.integers(-3, 40), st.none()))
        orders.append(
            {
                "order_id": f"ORD-{order_no + 1}",
                "customer_id": CUSTOMER if happy else draw(st.sampled_from([CUSTOMER, "C8"])),
                "status": "delivered" if happy else draw(st.sampled_from(["delivered", "shipped"])),
                "delivered_on": None if days is None else (as_of - timedelta(days=days)).isoformat(),
                "lines": lines,
            }
        )
    return {
        "as_of": as_of.isoformat(),
        "currency": "USD",
        "products": products,
        "orders": orders,
        "returns": [],
        "refunds": [],
        "exchanges": [],
        "quarantine": [],
        "idempotency": {},
        "x_extra": draw(st.integers()),
    }


INDEX = st.sampled_from([0, 0, 1, 2])  # biased toward the always-returnable line
INTENT = st.sampled_from(["refund", "exchange"])
SIZE = st.sampled_from([None, "S", "M", "L", "XL", "UP", "DOWN"])


# --- invariants ----------------------------------------------------------------------------------


def _without_stock(doc: dict[str, Any]) -> dict[str, Any]:
    doc = copy.deepcopy(doc)
    for product in doc["products"]:
        product.pop("stock")
    return doc


def check_invariants(initial: dict[str, Any], final: dict[str, Any], expected_stock: dict[str, Any], commits: int) -> None:
    assert list(final) == list(initial)

    stock = {p["sku"]: p["stock"] for p in final["products"]}
    assert stock == expected_stock
    assert all(value >= 0 for value in stock.values())

    returned_keys = [(r["order_id"], r["line_id"]) for r in final["returns"]]
    assert len(returned_keys) == len(set(returned_keys)), "a line was returned twice"
    assert len(final["quarantine"]) == len(final["returns"]) == len(final["refunds"]) + len(final["exchanges"])
    assert len(final["idempotency"]) == len(final["returns"]) == commits
    assert len({r["request_id"] for r in final["returns"]}) == len(final["returns"]), "one transaction per request"

    lines = {(o["order_id"], line["line_id"]): (o, line) for o in initial["orders"] for line in o["lines"]}
    for key, (order, line) in lines.items():
        now = policy.find_line(policy.find_order(final, key[0]), key[1])
        assert now["returned"] == (line["returned"] or key in returned_keys)
        if key in returned_keys:
            assert order["customer_id"] == CUSTOMER and line["returned"] is False

    refunded = [lines[(r["order_id"], r["line_id"])][1]["paid_cents"] for r in final["refunds"]]
    assert sum(r["amount_cents"] for r in final["refunds"]) == sum(refunded)
    assert all(r["amount_cents"] == paid for r, paid in zip(final["refunds"], refunded))

    for exchange in final["exchanges"]:
        original = policy.find_product(initial, exchange["original_sku"])
        replacement = policy.find_product(initial, exchange["replacement_sku"])
        assert replacement["product_id"] == original["product_id"]
        assert not policy.same_size(replacement["size"], original["size"])
        assert replacement["price_cents"] == original["price_cents"] and exchange["amount_cents"] == 0

    for collection in RECORD_LISTS:
        for record in final[collection]:
            assert all(record.get(name) is not None for name in COMMON)
            assert record["customer_id"] == CUSTOMER and record["business_date"] == initial["as_of"]


# --- the state machine ---------------------------------------------------------------------------


class DeskMachine(RuleBasedStateMachine):
    @initialize(state=states())
    def load(self, state: dict[str, Any]) -> None:
        self.initial = copy.deepcopy(state)
        self.store = Store(copy.deepcopy(state))
        self.desks = [BusinessTools(self.store, CUSTOMER, f"REQ-{name}", FaultPlan([]), AuditLog()) for name in "AB"]
        self.active = 0
        self.turn = 0
        self.expected_stock = {p["sku"]: p["stock"] for p in state["products"]}
        self.commits = 0
        self.replays = 0

    @property
    def desk(self) -> BusinessTools:
        return self.desks[self.active]

    def _call(self, tool_call: Any, *, grant_allows_mutation: bool = False) -> ToolResult:
        """Run one gateway call and check the per-step rules around it."""
        before = self.store.export()
        result: ToolResult = tool_call()
        after = self.store.export()
        assert result.code != INTERNAL, result
        if result.is_replay or (not result.ok and result.code != TIMEOUT):
            assert after == before, "a replay or refusal changed state"
            self.replays += result.is_replay
        if not (grant_allows_mutation and self.desk.grant is not None):
            assert _without_stock(after) == _without_stock(before), "mutation without a grant"
        committed = result.code == TIMEOUT or (result.ok and result.data.get("status") == "executed")
        if committed:
            self.commits += 1
            transaction = self.desk.completed_record()
            if transaction["action"] == "exchange":
                self.expected_stock[transaction["replacement_sku"]] -= 1
        return result

    def _target(self, order_index: int, line_index: int, intent: str, size: str | None) -> None:
        order = self.initial["orders"][order_index % len(self.initial["orders"])]
        line = order["lines"][line_index % len(order["lines"])]
        self.desk.set_target(order["order_id"], line["line_id"], intent, requested_size=size)

    # What a customer message leads to: a new turn, a grounded target, a proposal.
    # The two "returnable"/"latest" rules steer runs toward commits (otherwise ~95% of
    # random runs never commit); the general rules keep every refusal path in play.
    @rule(intent=INTENT, size=SIZE)
    def customer_asks_about_returnable_line(self, intent: str, size: str | None) -> None:
        self.customer_asks(0, 0, intent, size)

    @rule(order_index=INDEX, line_index=INDEX, intent=INTENT, size=SIZE)
    def customer_asks(self, order_index: int, line_index: int, intent: str, size: str | None) -> None:
        self.turn += 1
        self.desk.begin_turn(self.turn)
        self._target(order_index, line_index, intent, size)
        self._call(lambda: self.desk.create_proposal(intent))

    @rule()
    def customer_confirms_latest(self) -> None:
        self.customer_confirms("latest")

    @rule(choice=st.sampled_from(["latest", "first", "bogus", None]))
    def customer_confirms(self, choice: str | None) -> None:
        self.turn += 1
        self.desk.begin_turn(self.turn)
        own = [p["proposal_id"] for p in self.desk.proposals() if p["request_id"] == self.desk.request_id]
        grant = {"latest": own[-1] if own else None, "first": own[0] if own else None, "bogus": "P-9999"}.get(choice)
        self.desk.set_grant(grant)
        self._call(self.desk.execute_proposal, grant_allows_mutation=True)

    # what a confused agent might do in between
    @rule(order_index=INDEX, line_index=INDEX, intent=INTENT, size=SIZE)
    def retarget(self, order_index: int, line_index: int, intent: str, size: str | None) -> None:
        self._target(order_index, line_index, intent, size)

    @rule(remedy=INTENT, replacement=st.sampled_from([None, "HOODIE-L", "HOODIE-S", "CAP-M"]))
    def agent_proposes(self, remedy: str, replacement: str | None) -> None:
        self._call(lambda: self.desk.create_proposal(remedy, replacement))

    @rule()
    def agent_executes_again(self) -> None:
        self._call(self.desk.execute_proposal, grant_allows_mutation=True)

    @rule()
    def agent_reads(self) -> None:
        for tool in (self.desk.check_eligibility, self.desk.list_exchange_options, self.desk.get_request_status):
            self._call(tool)

    @rule()
    def switch_request(self) -> None:
        self.active = 1 - self.active

    # what the outside world does
    @rule(sku_index=st.integers(0, 9), new_stock=st.integers(0, 3))
    def stock_fault(self, sku_index: int, new_stock: int) -> None:
        skus = sorted(self.expected_stock) + ["NOPE-1"]
        sku = skus[sku_index % len(skus)]
        before = _without_stock(self.store.export())
        self.desk.inject_stock_change(sku, new_stock)
        assert _without_stock(self.store.export()) == before
        if sku in self.expected_stock:
            self.expected_stock[sku] = new_stock

    @rule()
    def arm_timeout(self) -> None:
        self.desk.arm_timeout()

    @invariant()
    def business_invariants(self) -> None:
        check_invariants(self.initial, self.store.export(), self.expected_stock, self.commits)

    def teardown(self) -> None:
        if not hasattr(self, "desks"):
            return
        event(f"commits: {self.commits}")  # --hypothesis-show-statistics proves commits are reached
        event(f"replays: {'yes' if self.replays else 'no'}")
        effect_sets = [e for d in self.desks for e in d.audit.entries() if e["kind"] == "tool" and e["committed_effects"]]
        assert len(effect_sets) == self.commits  # exactly one effects set per committed transaction


TestDeskMachine = DeskMachine.TestCase
TestDeskMachine.settings = settings(
    max_examples=500, stateful_step_count=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


# --- single-function properties ---------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(days=st.integers(-5, 40), paid=st.integers(0, 100_000))
def test_eligible_iff_delivered_0_to_30_days_ago(days, paid):
    as_of = date(2028, 2, 29)
    state = {
        "as_of": as_of.isoformat(),
        "products": [],
        "orders": [
            {
                "order_id": "O",
                "customer_id": CUSTOMER,
                "status": "delivered",
                "delivered_on": (as_of - timedelta(days=days)).isoformat(),
                "lines": [{"line_id": "L", "sku": "S", "paid_cents": paid, "unopened": True, "final_sale": False, "returned": False}],
            }
        ],
    }
    report = policy.eligibility(state, CUSTOMER, "O", "L")
    assert report.eligible is (0 <= days <= 30)
    assert report.days_since_delivery == days
    assert policy.refund_amount(state["orders"][0]["lines"][0]) == paid


JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=8),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3),
    max_leaves=12,
)


@settings(max_examples=150, deadline=None)
@given(state=states(), extras=st.dictionaries(st.text(min_size=1, max_size=8), JSON_VALUES, max_size=4))
def test_export_round_trips_byte_identically_with_unknown_fields(state, extras):
    state = {**state, **{key: value for key, value in extras.items() if key not in state}}
    state["orders"][0]["x_unknown"] = extras
    assert json.dumps(Store(state).export()) == json.dumps(state)
