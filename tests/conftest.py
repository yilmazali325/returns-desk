"""Shared fixtures for the business-core tests.

``base_state()`` deliberately uses IDs unlike the public cases (C7 / ORD-55 / X9 /
HOODIE) so tests prove behaviour comes from the rules, not from fixture names. The
public cases are loaded separately for round-trip and end-to-end gateway checks.

Everything is exposed as fixtures (never ``from conftest import …``): pytest rebinds the
``conftest`` module name when other test folders add their own conftest.py.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
# pyproject.toml has no build-system, so the project is not installed into .venv;
# make `import desk` / `import server` work for `.venv/bin/pytest` from any directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desk.audit import AuditLog  # noqa: E402
from desk.faults import FaultPlan  # noqa: E402
from desk.store import Store  # noqa: E402
from desk.tools import BusinessTools  # noqa: E402

PUBLIC_CASES_PATH = ROOT / "candidate_assets" / "public_cases.json"

CUSTOMER = "C7"
OTHER_CUSTOMER = "C8"
REQUEST = "REQ-T1"

# Every transaction record must carry these (SPEC §2, CONTRACT "Records must identify").
COMMON_FIELDS = (
    "transaction_id",
    "request_id",
    "proposal_id",
    "customer_id",
    "order_id",
    "line_id",
    "sku",
    "replacement_sku",
    "amount_cents",
    "currency",
    "business_date",
    "status",
)


def base_state() -> dict[str, Any]:
    """as_of 2026-03-15; ORD-55 delivered 14 days earlier; paid 4200 < catalog 5000."""
    return {
        "as_of": "2026-03-15",
        "currency": "USD",
        "products": [
            {"sku": "HOODIE-XS", "product_id": "HOODIE", "size": "XS", "price_cents": 4200, "stock": 2},
            {"sku": "HOODIE-S", "product_id": "HOODIE", "size": "S", "price_cents": 5000, "stock": 2},
            {"sku": "HOODIE-M", "product_id": "HOODIE", "size": "M", "price_cents": 5000, "stock": 1},
            {"sku": "HOODIE-L", "product_id": "HOODIE", "size": "L", "price_cents": 5000, "stock": 3},
            {"sku": "HOODIE-XL", "product_id": "HOODIE", "size": "XL", "price_cents": 5500, "stock": 4},
            {"sku": "CAP-M", "product_id": "CAP", "size": "M", "price_cents": 2000, "stock": 5},
        ],
        "orders": [
            {
                "order_id": "ORD-55",
                "customer_id": CUSTOMER,
                "status": "delivered",
                "delivered_on": "2026-03-01",
                "lines": [
                    {"line_id": "X9", "sku": "HOODIE-M", "paid_cents": 4200, "unopened": True, "final_sale": False, "returned": False},
                    {"line_id": "X10", "sku": "CAP-M", "paid_cents": 1500, "unopened": True, "final_sale": False, "returned": False},
                ],
            },
            {
                "order_id": "ORD-66",
                "customer_id": OTHER_CUSTOMER,
                "status": "delivered",
                "delivered_on": "2026-03-01",
                "lines": [
                    {"line_id": "Z1", "sku": "HOODIE-S", "paid_cents": 5000, "unopened": True, "final_sale": False, "returned": False},
                ],
            },
        ],
        "returns": [],
        "refunds": [],
        "exchanges": [],
        "quarantine": [],
        "idempotency": {},
    }  # fmt: skip


def load_public_cases() -> list[dict[str, Any]]:
    return json.loads(PUBLIC_CASES_PATH.read_text())["cases"]


class Lookup:
    """Tiny readers for assertions on exported states."""

    @staticmethod
    def product(state: dict[str, Any], sku: str) -> dict[str, Any]:
        return next(p for p in state["products"] if p["sku"] == sku)

    @staticmethod
    def line(state: dict[str, Any], order_id: str, line_id: str) -> dict[str, Any]:
        order = next(o for o in state["orders"] if o["order_id"] == order_id)
        return next(item for item in order["lines"] if item["line_id"] == line_id)


@pytest.fixture
def state() -> dict[str, Any]:
    return base_state()


@pytest.fixture
def make_state() -> Callable[[], dict[str, Any]]:
    return base_state


@pytest.fixture
def lookup() -> Lookup:
    return Lookup()


@pytest.fixture
def common_fields() -> tuple[str, ...]:
    return COMMON_FIELDS


@pytest.fixture(scope="session")
def public_cases() -> list[dict[str, Any]]:
    return load_public_cases()


@pytest.fixture
def make_tools() -> Callable[..., BusinessTools]:
    """Factory: ``make_tools(state, faults=[...], turn_count=3, request_id=..., store=...)``."""

    def factory(
        initial: dict[str, Any] | None = None,
        *,
        customer_id: str = CUSTOMER,
        request_id: str = REQUEST,
        faults: list[Any] | None = None,
        turn_count: int | None = None,
        store: Store | None = None,
        audit: AuditLog | None = None,
        alias_fn: Callable[[str], list[str]] | None = None,
    ) -> BusinessTools:
        store = store or Store(copy.deepcopy(initial if initial is not None else base_state()))
        plan = FaultPlan(faults or [], turn_count)
        return BusinessTools(store, customer_id, request_id, plan, audit if audit is not None else AuditLog(), alias_fn)

    return factory


@pytest.fixture(scope="session")
def browser() -> Iterator[Any]:
    """One headless Chromium for every browser test (tests/e2e and the review UI tests). Playwright's
    sync API runs an event loop per instance, and a second instance in the same session fails, so
    no test module may start its own."""
    from playwright.sync_api import sync_playwright  # only browser tests pay for the import

    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()
