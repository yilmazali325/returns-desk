"""Pure-rule tests: eligibility boundaries and date edge cases, strict booleans, the
no-leak rule for foreign orders, exchange validity and size resolution."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from desk import policy
from desk.domain import (
    ALREADY_RETURNED,
    DELIVERY_IN_FUTURE,
    DIFFERENT_PRODUCT,
    FINAL_SALE,
    LINE_NOT_FOUND,
    MISSING_DELIVERY_DATE,
    NOT_DELIVERED,
    OPENED,
    ORDER_NOT_FOUND,
    OUT_OF_STOCK,
    PRICE_MISMATCH,
    RETURN_WINDOW_EXPIRED,
    SAME_SIZE,
    SIZE_NOT_OFFERED,
    SKU_NOT_FOUND,
)

CUSTOMER = "C7"


def _deliver(state: dict, as_of: str, delivered_on: object) -> dict:
    state["as_of"] = as_of
    state["orders"][0]["delivered_on"] = delivered_on
    return state


def _x9(state: dict) -> dict:
    return state["orders"][0]["lines"][0]


# --- dates -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-03-01", date(2026, 3, 1)),
        ("2026-03-01T23:59:59Z", date(2026, 3, 1)),  # date.fromisoformat alone raises on this
        ("2028-02-29", date(2028, 2, 29)),
        ("2027-02-29", None),
        ("2026-13-01", None),
        ("soon", None),
        ("", None),
        (None, None),
        (20260301, None),
    ],
)
def test_parse_date(value, expected):
    assert policy.parse_date(value) == expected


def test_parse_as_of_rejects_a_state_without_a_business_date():
    with pytest.raises(ValueError, match="as_of"):
        policy.parse_as_of({"as_of": "yesterday"})


# --- eligibility: window boundaries --------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "eligible", "reasons"),
    [
        (0, True, []),
        (1, True, []),
        (30, True, []),
        (31, False, [RETURN_WINDOW_EXPIRED]),
        (-1, False, [DELIVERY_IN_FUTURE]),
    ],
)
def test_window_is_0_to_30_days_inclusive(state, days, eligible, reasons):
    as_of = date(2026, 3, 15)
    _deliver(state, as_of.isoformat(), (as_of - timedelta(days=days)).isoformat())
    result = policy.eligibility(state, CUSTOMER, "ORD-55", "X9")
    assert (result.eligible, result.reasons, result.days_since_delivery) == (eligible, reasons, days)


@pytest.mark.parametrize(
    ("as_of", "delivered_on", "days"),
    [
        ("2026-03-31", "2026-03-01", 30),  # end of a 31-day month
        ("2026-03-01", "2026-01-30", 30),  # across a 28-day February
        ("2026-03-01", "2026-01-29", 31),
        ("2028-03-01", "2028-01-31", 30),  # leap year: February has 29 days
        ("2028-03-30", "2028-02-29", 30),  # delivered on the leap day itself
        ("2028-03-31", "2028-02-29", 31),
        ("2027-01-01", "2026-12-02", 30),  # across a year end
    ],
)
def test_month_ends_leap_day_and_year_end(state, as_of, delivered_on, days):
    result = policy.eligibility(_deliver(state, as_of, delivered_on), CUSTOMER, "ORD-55", "X9")
    assert result.days_since_delivery == days
    assert result.eligible is (days <= 30)


@pytest.mark.parametrize("delivered_on", [None, "", "soon", "2027-02-29", 20260301, ["2026-03-01"]])
def test_missing_or_invalid_delivery_date_is_refused_not_crashed(state, delivered_on):
    result = policy.eligibility(_deliver(state, "2026-03-15", delivered_on), CUSTOMER, "ORD-55", "X9")
    assert result.reasons == [MISSING_DELIVERY_DATE]
    assert result.delivered_on is None and result.days_since_delivery is None


def test_absent_delivery_date_key(state):
    del state["orders"][0]["delivered_on"]
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [MISSING_DELIVERY_DATE]


def test_timestamp_delivery_dates_use_the_calendar_day(state):
    result = policy.eligibility(_deliver(state, "2026-03-31", "2026-03-01T23:59:59Z"), CUSTOMER, "ORD-55", "X9")
    assert (result.eligible, result.delivered_on, result.as_of) == (True, "2026-03-01", "2026-03-31")


# --- eligibility: strict booleans and status ---------------------------------------------


@pytest.mark.parametrize("value", [False, None, "true", 1, "yes"])
def test_unopened_must_be_exactly_true(state, value):
    _x9(state)["unopened"] = value
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [OPENED]


@pytest.mark.parametrize("value", [True, None, "false", 0])
def test_final_sale_must_be_exactly_false(state, value):
    _x9(state)["final_sale"] = value
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [FINAL_SALE]


@pytest.mark.parametrize("value", [True, None, "false", 0])
def test_returned_must_be_exactly_false(state, value):
    _x9(state)["returned"] = value
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [ALREADY_RETURNED]


def test_missing_boolean_keys_fail_closed(state):
    for key in ("unopened", "final_sale", "returned"):
        del _x9(state)[key]
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [OPENED, FINAL_SALE, ALREADY_RETURNED]


@pytest.mark.parametrize("status", ["shipped", "processing", None, 1, "Delivered", " delivered "])
def test_only_delivered_orders_qualify(state, status):
    """The status must be exactly the contract's "delivered" (review B6; the oracle reads it the same way)."""
    state["orders"][0]["status"] = status
    assert policy.eligibility(state, CUSTOMER, "ORD-55", "X9").reasons == [NOT_DELIVERED]


def test_every_failing_reason_is_reported_in_canonical_order(state):
    state["orders"][0]["status"] = "shipped"
    _deliver(state, "2026-03-15", "2026-01-01")
    _x9(state).update(unopened=False, final_sale=True, returned=True)
    result = policy.eligibility(state, CUSTOMER, "ORD-55", "X9")
    assert result.reasons == [NOT_DELIVERED, RETURN_WINDOW_EXPIRED, OPENED, FINAL_SALE, ALREADY_RETURNED]


def test_eligible_line_carries_its_facts(state):
    result = policy.eligibility(state, CUSTOMER, "ORD-55", "X9")
    assert result.to_dict() == {
        "eligible": True,
        "reasons": [],
        "order_id": "ORD-55",
        "line_id": "X9",
        "sku": "HOODIE-M",
        "product_id": "HOODIE",
        "size": "M",
        "paid_cents": 4200,
        "delivered_on": "2026-03-01",
        "as_of": "2026-03-15",
        "days_since_delivery": 14,
        "window_days": 30,
    }


def test_line_with_unknown_sku_is_still_judged_on_policy(state):
    _x9(state)["sku"] = "GONE-1"
    result = policy.eligibility(state, CUSTOMER, "ORD-55", "X9")
    assert result.eligible and result.product_id is None and result.size is None


# --- eligibility: identity ------------------------------------------------------------------


def test_other_customers_order_is_indistinguishable_from_a_missing_one(state):
    foreign = policy.eligibility(state, CUSTOMER, "ORD-66", "Z1").to_dict()
    missing = policy.eligibility(state, CUSTOMER, "ORD-404", "Z1").to_dict()
    assert foreign["reasons"] == missing["reasons"] == [ORDER_NOT_FOUND]
    assert {**foreign, "order_id": None} == {**missing, "order_id": None}
    # nothing about the other customer's line leaks
    assert foreign["sku"] is foreign["paid_cents"] is foreign["delivered_on"] is None


def test_the_owner_can_see_their_own_order(state):
    assert policy.eligibility(state, "C8", "ORD-66", "Z1").eligible


def test_unknown_line_in_an_owned_order(state):
    result = policy.eligibility(state, CUSTOMER, "ORD-55", "X99")
    assert result.reasons == [LINE_NOT_FOUND] and result.sku is None


def test_duplicate_order_ids_resolve_to_the_customers_own(state):
    state["orders"][1]["order_id"] = "ORD-55"  # another customer's order with the same id
    assert policy.find_customer_order(state, CUSTOMER, "ORD-55")["customer_id"] == CUSTOMER


# --- refund amount -----------------------------------------------------------------------


def test_refund_is_paid_cents_not_catalog_price(state):
    assert policy.refund_amount(_x9(state)) == 4200


# --- exchange validity ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("replacement", "reasons"),
    [
        ("HOODIE-L", []),
        ("HOODIE-S", []),
        ("HOODIE-M", [SAME_SIZE]),
        ("CAP-M", [DIFFERENT_PRODUCT, PRICE_MISMATCH]),
        ("HOODIE-XL", [PRICE_MISMATCH]),
        # 4200 equals what was PAID for the M, but not the M's catalog price (5000)
        ("HOODIE-XS", [PRICE_MISMATCH]),
        ("HOODIE-3XL", [SIZE_NOT_OFFERED]),
    ],
)
def test_validate_replacement(state, replacement, reasons):
    assert policy.validate_replacement(state, "HOODIE-M", replacement) == reasons


@pytest.mark.parametrize("stock", [0, -1, "3", None, True])
def test_replacement_needs_positive_integer_stock(state, lookup, stock):
    lookup.product(state, "HOODIE-L")["stock"] = stock
    assert policy.validate_replacement(state, "HOODIE-M", "HOODIE-L") == [OUT_OF_STOCK]


def test_unknown_original_sku_makes_exchange_impossible(state):
    assert policy.validate_replacement(state, "GONE-1", "HOODIE-L") == [SKU_NOT_FOUND]
    assert policy.options_for_sku(state, "GONE-1") == []


def test_missing_catalog_price_is_a_mismatch(state, lookup):
    del lookup.product(state, "HOODIE-M")["price_cents"]
    del lookup.product(state, "HOODIE-L")["price_cents"]
    assert policy.validate_replacement(state, "HOODIE-M", "HOODIE-L") == [PRICE_MISMATCH]


def test_same_size_with_another_sku_or_case_is_still_same_size(state):
    state["products"].append({"sku": "HOODIE-M2", "product_id": "HOODIE", "size": "m", "price_cents": 5000, "stock": 9})
    assert policy.validate_replacement(state, "HOODIE-M", "HOODIE-M2") == [SAME_SIZE]
    assert "HOODIE-M2" not in [o.sku for o in policy.options_for_sku(state, "HOODIE-M")]


def test_options_cover_every_other_size_in_ladder_order(state):
    options = policy.options_for_sku(state, "HOODIE-M")
    assert [(o.sku, o.valid, o.reasons) for o in options] == [
        ("HOODIE-XS", False, [PRICE_MISMATCH]),
        ("HOODIE-S", True, []),
        ("HOODIE-L", True, []),
        ("HOODIE-XL", False, [PRICE_MISMATCH]),
    ]
    assert [o.sku for o in policy.valid_options(options)] == ["HOODIE-S", "HOODIE-L"]
    assert options[1].to_dict() == {"sku": "HOODIE-S", "size": "S", "price_cents": 5000, "stock": 2, "valid": True, "reasons": []}


def test_unrankable_sizes_sort_last(state):
    state["products"].append({"sku": "HOODIE-TALL", "product_id": "HOODIE", "size": "Tall", "price_cents": 5000, "stock": 1})
    assert policy.options_for_sku(state, "HOODIE-M")[-1].sku == "HOODIE-TALL"


# --- sizes -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("L", "L"),
        ("large", "L"),
        (" Lg ", "L"),
        ("x-large", "XL"),
        ("Extra Large", "XL"),
        ("2XL", "XXL"),
        ("xxxl", "3XL"),
        ("medium", "M"),
        ("32", "32"),
        ("032", "32"),
        ("32.0", "32"),
        ("32.50", "32.5"),
        ("0", "0"),
        ("One Size", "ONESIZE"),
    ],
)
def test_normalize_size(text, label):
    assert policy.normalize_size(text) == label


def test_size_rank_orders_letters_then_numbers():
    assert policy.size_rank("XS") < policy.size_rank("large") < policy.size_rank("5XL") < policy.size_rank("2")
    assert policy.size_rank("30") < policy.size_rank("32.5")
    assert policy.size_rank("Tall") is None


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("L", ("HOODIE-L", None)),
        ("large", ("HOODIE-L", None)),
        ("hoodie-l", ("HOODIE-L", None)),  # a sibling SKU is accepted as the size
        ("UP", ("HOODIE-L", None)),
        ("down", ("HOODIE-S", None)),
        ("M", ("HOODIE-M", None)),  # resolves; validity (SAME_SIZE) is judged separately
        ("SAME", ("HOODIE-M", None)),  # "the same size": the size the customer has (then SAME_SIZE)
        ("same size", ("HOODIE-M", None)),
        ("XXL", (None, SIZE_NOT_OFFERED)),
        (None, (None, None)),
        ("  ", (None, None)),
        ("UNSPECIFIED", (None, None)),
        ("bigger", ("HOODIE-L", None)),
        ("size up", ("HOODIE-L", None)),
        ("ONE_SIZE_UP", ("HOODIE-L", None)),
        ("one size down", ("HOODIE-S", None)),
        ("smaller", ("HOODIE-S", None)),
        ("lrg", ("HOODIE-L", None)),
        ("hoodiel", ("HOODIE-L", None)),  # a SKU typed without its dash
        ("CAP-M", (None, DIFFERENT_PRODUCT)),  # a SKU of another product
    ],
)
def test_resolve_size(state, requested, expected):
    assert policy.resolve_size(requested, "HOODIE-M", state["products"]) == expected


def test_resolve_relative_size_at_the_ends_of_the_ladder(state):
    assert policy.resolve_size("UP", "HOODIE-XL", state["products"]) == (None, SIZE_NOT_OFFERED)
    assert policy.resolve_size("DOWN", "HOODIE-XS", state["products"]) == (None, SIZE_NOT_OFFERED)


def test_resolve_relative_size_needs_a_rankable_original(state):
    state["products"].append({"sku": "HOODIE-TALL", "product_id": "HOODIE", "size": "Tall", "price_cents": 5000, "stock": 1})
    assert policy.resolve_size("UP", "HOODIE-TALL", state["products"]) == (None, SIZE_NOT_OFFERED)


def test_resolve_numeric_sizes():
    catalog = [
        {"sku": f"JEANS-{waist}", "product_id": "JEANS", "size": str(waist), "price_cents": 7000, "stock": 1}
        for waist in (30, 32, 34)
    ]
    assert policy.resolve_size("032", "JEANS-30", catalog) == ("JEANS-32", None)
    assert policy.resolve_size("UP", "JEANS-32", catalog) == ("JEANS-34", None)
    assert policy.resolve_size("DOWN", "JEANS-32", catalog) == ("JEANS-30", None)


def test_resolve_size_checks_the_requested_product(state):
    assert policy.resolve_size("M", "HOODIE-M", state["products"], "CAP") == (None, DIFFERENT_PRODUCT)
    assert policy.resolve_size("M", "HOODIE-M", state["products"], "cap-m") == (None, DIFFERENT_PRODUCT)
    assert policy.resolve_size("L", "HOODIE-M", state["products"], "hoodie") == ("HOODIE-L", None)
    # free text that names no catalog product is grounding's business, not a refusal
    assert policy.resolve_size("L", "HOODIE-M", state["products"], "sweatshirt") == ("HOODIE-L", None)
    assert policy.resolve_size(None, "HOODIE-M", state["products"], "CAP") == (None, DIFFERENT_PRODUCT)


def test_resolve_size_with_unknown_original(state):
    assert policy.resolve_size("L", "GONE-1", state["products"]) == (None, SKU_NOT_FOUND)


def test_find_order_ignores_the_customer_and_misses_cleanly(state):
    assert policy.find_order(state, "ORD-66")["customer_id"] == "C8"
    assert policy.find_order(state, "ORD-404") is None


def test_find_product_accepts_state_or_catalog(state):
    assert policy.find_product(state, "CAP-M") is policy.find_product(state["products"], "CAP-M")
    assert policy.find_product(state, "nope") is None


def test_is_int_excludes_booleans():
    assert policy.is_int(3) and policy.is_int(-1)
    assert not policy.is_int(True) and not policy.is_int("3") and not policy.is_int(3.0)
