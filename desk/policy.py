"""Pure business rules: return eligibility, exchange validity, size resolution, refund amount.

Every function takes the state document (or the catalog) and returns a value; nothing
here mutates state, reads the clock or touches the store. That makes the rules easy to
test exhaustively and lets the gateway re-run them inside the commit transaction
(revalidation) without any side effects.

Fail-closed choices, all from CONTRACT.md / SPEC §3:
- dates are parsed as ``date.fromisoformat(s[:10])``; anything unparseable is a missing date;
- booleans are compared with ``is`` (``"true"``, ``1`` or ``None`` never qualify);
- another customer's order and a nonexistent order produce the identical result.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

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
    WINDOW_DAYS,
    Eligibility,
    ExchangeOption,
)

SIZE_LADDER = ("XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL", "5XL")

# Spoken or abbreviated size words -> ladder label: the ONE size vocabulary of the desk (grounding
# reads customer text with it, the gateway resolves requested sizes with it). Keys are "compact"
# (casefolded, no spaces/dashes/underscores) so "x-large", "X Large" and "xlarge" all match.
# Within one label the spoken words come first: the account card shows the first two.
SIZE_WORDS = {
    "xxs": "XXS", "2xs": "XXS", "xxsmall": "XXS", "extraextrasmall": "XXS",
    "xs": "XS", "xsmall": "XS", "extrasmall": "XS",
    "small": "S", "sm": "S", "s": "S",
    "medium": "M", "med": "M", "m": "M",
    "large": "L", "lg": "L", "lrg": "L", "l": "L",
    "xl": "XL", "xlarge": "XL", "extralarge": "XL",
    "xxl": "XXL", "2xl": "XXL", "xxlarge": "XXL", "extraextralarge": "XXL",
    "3xl": "3XL", "xxxl": "3XL",
    "4xl": "4XL", "xxxxl": "4XL",
    "5xl": "5XL", "xxxxxl": "5XL",
}  # fmt: skip
# A requested size may be a direction instead of a label ("UP" = the next offered size up).
# Compact forms, like SIZE_WORDS keys.
UP_WORDS = frozenset({"up", "sizeup", "onesizeup", "nextsizeup", "bigger", "larger"})
DOWN_WORDS = frozenset({"down", "sizedown", "onesizedown", "nextsizedown", "smaller"})
SAME_WORDS = frozenset({"same", "samesize"})  # "the same size" = the size the customer already has
UNSPECIFIED_WORDS = frozenset({"", "unspecified", "none", "any", "n/a"})

_NUMERIC_SIZE = re.compile(r"\d+(?:\.\d+)?")


# --- Small helpers -------------------------------------------------------------------


def is_int(value: Any) -> bool:
    """True for real integers only (``True`` is an int in Python but not a quantity)."""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_date(value: Any) -> date | None:
    """ISO date from the first 10 characters, or None when missing/invalid."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_as_of(state: dict[str, Any]) -> date:
    """The business date. A state without a valid ``as_of`` cannot be evaluated at all."""
    as_of = parse_date(state.get("as_of"))
    if as_of is None:
        raise ValueError(f"state.as_of must be an ISO date, got {state.get('as_of')!r}")
    return as_of


def find_customer_order(state: dict[str, Any], customer_id: str, order_id: str) -> dict | None:
    """The order only if it belongs to this customer; a foreign order is simply not found."""
    for order in state.get("orders", []):
        if order.get("order_id") == order_id and order.get("customer_id") == customer_id:
            return order
    return None


def find_order(state: dict[str, Any], order_id: str) -> dict | None:
    for order in state.get("orders", []):
        if order.get("order_id") == order_id:
            return order
    return None


def find_line(order: dict[str, Any], line_id: str) -> dict | None:
    for line in order.get("lines", []):
        if line.get("line_id") == line_id:
            return line
    return None


def find_product(catalog_or_state: Any, sku: Any) -> dict | None:
    """Accepts either the state document or the products list."""
    products = catalog_or_state.get("products", []) if isinstance(catalog_or_state, dict) else catalog_or_state
    for product in products:
        if product.get("sku") == sku:
            return product
    return None


def refund_amount(line: dict[str, Any]) -> int:
    """Refund exactly what was paid for the unit; discounts are already allocated."""
    return line["paid_cents"]


# --- Eligibility -----------------------------------------------------------------------


def eligibility(state: dict[str, Any], customer_id: str, order_id: str, line_id: str) -> Eligibility:
    """Every failing reason for returning this line, in ELIGIBILITY_CODES order."""
    as_of = parse_as_of(state)
    order = find_customer_order(state, customer_id, order_id)
    if order is None:
        return _not_found(ORDER_NOT_FOUND, order_id, line_id, as_of)
    line = find_line(order, line_id)
    if line is None:
        return _not_found(LINE_NOT_FOUND, order_id, line_id, as_of)

    delivered = parse_date(order.get("delivered_on"))
    days = (as_of - delivered).days if delivered else None
    reasons: list[str] = []
    if not _is_delivered_status(order.get("status")):
        reasons.append(NOT_DELIVERED)
    if days is None:
        reasons.append(MISSING_DELIVERY_DATE)
    elif days < 0:
        reasons.append(DELIVERY_IN_FUTURE)
    elif days > WINDOW_DAYS:
        reasons.append(RETURN_WINDOW_EXPIRED)
    if line.get("unopened") is not True:
        reasons.append(OPENED)
    if line.get("final_sale") is not False:
        reasons.append(FINAL_SALE)
    if line.get("returned") is not False:
        reasons.append(ALREADY_RETURNED)

    product = find_product(state, line.get("sku")) or {}
    return Eligibility(
        eligible=not reasons,
        reasons=reasons,
        order_id=order_id,
        line_id=line_id,
        sku=line.get("sku"),
        product_id=product.get("product_id"),
        size=product.get("size"),
        paid_cents=line.get("paid_cents"),
        delivered_on=delivered.isoformat() if delivered else None,
        as_of=as_of.isoformat(),
        days_since_delivery=days,
    )


def _is_delivered_status(status: Any) -> bool:
    # The contract's literal value, compared exactly (review B6): "Delivered" or " delivered " is
    # not the disclosed status, and refusing is the reading that can never cause a wrong mutation.
    return status == "delivered"


def _not_found(code: str, order_id: str, line_id: str, as_of: date) -> Eligibility:
    # Deliberately carries no line facts: a foreign order must look exactly like a missing one.
    return Eligibility(
        eligible=False,
        reasons=[code],
        order_id=order_id,
        line_id=line_id,
        sku=None,
        product_id=None,
        size=None,
        paid_cents=None,
        delivered_on=None,
        as_of=as_of.isoformat(),
        days_since_delivery=None,
    )


# --- Sizes -----------------------------------------------------------------------------


def _compact(text: Any) -> str:
    return re.sub(r"[\s\-_]+", "", str(text).casefold())


def normalize_id(value: Any) -> str:
    """Compare IDs the way people type them: "#ord 7q" == "ORD-7Q" (case, '#', '-', '_', spaces ignored)."""
    return re.sub(r"[\s#_\-]+", "", str(value)).casefold()


def normalize_size(text: Any) -> str:
    """Canonical size label: ladder label ("L"), numeric ("32"), or compact uppercase."""
    compact = _compact(text)
    if compact in SIZE_WORDS:
        return SIZE_WORDS[compact]
    if _NUMERIC_SIZE.fullmatch(compact):
        whole, _, fraction = compact.partition(".")
        whole = whole.lstrip("0") or "0"
        fraction = fraction.rstrip("0")
        return f"{whole}.{fraction}" if fraction else whole
    return compact.upper()


def size_rank(size: Any) -> tuple[int, float] | None:
    """Sort key on the size ladder (letters first, then numbers); None if unrankable."""
    label = normalize_size(size)
    if label in SIZE_LADDER:
        return (0, float(SIZE_LADDER.index(label)))
    if _NUMERIC_SIZE.fullmatch(label):
        return (1, float(label))
    return None


def same_size(a: Any, b: Any) -> bool:
    return normalize_size(a) == normalize_size(b)


def resolve_size(
    requested: str | None,
    original_sku: str,
    catalog: list[dict[str, Any]],
    requested_product: str | None = None,
) -> tuple[str | None, str | None]:
    """Map a requested size to a SKU of the original product. The ONE size resolver of the desk:
    the gateway uses it to bind the replacement and the handoff gate uses it to compare readings.

    Accepts a label ("L", "large", "32", "9.5"), a direction ("UP", "bigger", "size down"),
    "SAME" (the size the customer has; validation then refuses it as SAME_SIZE) or a sibling SKU
    ("SCARF-L", "scarfl"). Returns ``(sku, None)``, ``(None, code)``, or ``(None, None)`` when no size
    was requested at all. Stock and price are NOT checked here (validate_replacement's job), so an
    out-of-stock size still resolves and is then refused with the precise reason.

    A requested product only counts when it names ANOTHER catalog product_id or SKU;
    free text such as "t-shirt" is ignored here because grounding owns aliases, and a
    wrong DIFFERENT_PRODUCT would deny a valid exchange.
    """
    original = find_product(catalog, original_sku)
    if original is None:
        return None, SKU_NOT_FOUND
    if requested_product and _names_other_product(requested_product, original, catalog):
        return None, DIFFERENT_PRODUCT
    wanted = _compact(requested if requested is not None else "")
    if wanted in UNSPECIFIED_WORDS:
        return None, None
    if wanted in SAME_WORDS:
        return original["sku"], None
    named = next((p for p in catalog if normalize_id(p.get("sku")) == normalize_id(requested)), None)
    if named is not None and named.get("product_id") != original.get("product_id"):
        return None, DIFFERENT_PRODUCT  # a SKU of another product ("CAP-M" for a hoodie)
    siblings = [p for p in catalog if p.get("product_id") == original.get("product_id")]
    if wanted in UP_WORDS or wanted in DOWN_WORDS:
        return _relative_size(original, siblings, up=wanted in UP_WORDS)
    for product in siblings:
        if same_size(product.get("size"), requested) or normalize_id(requested) == normalize_id(product.get("sku")):
            return product["sku"], None
    return None, SIZE_NOT_OFFERED


def _names_other_product(requested_product: str, original: dict, catalog: list[dict]) -> bool:
    wanted = requested_product.strip().casefold()
    for product in catalog:
        names = {str(product.get("product_id", "")).casefold(), str(product.get("sku", "")).casefold()}
        if wanted in names:
            return product.get("product_id") != original.get("product_id")
    return False


def _relative_size(original: dict, siblings: list[dict], *, up: bool) -> tuple[str | None, str | None]:
    """The next size up or down on the ladder among this product's offered sizes."""
    origin = size_rank(original.get("size"))
    if origin is None:
        return None, SIZE_NOT_OFFERED
    ranked = [(size_rank(p.get("size")), p) for p in siblings]
    if up:
        bigger = sorted((r, p["sku"]) for r, p in ranked if r is not None and r > origin)
        return (bigger[0][1], None) if bigger else (None, SIZE_NOT_OFFERED)
    smaller = sorted(((r, p["sku"]) for r, p in ranked if r is not None and r < origin), reverse=True)
    return (smaller[0][1], None) if smaller else (None, SIZE_NOT_OFFERED)


# --- Exchanges -------------------------------------------------------------------------


def validate_replacement(state: dict[str, Any], original_sku: str, replacement_sku: str) -> list[str]:
    """Every reason this replacement is not a valid exchange for the original SKU.

    Price is compared with the ORIGINAL SKU's catalog price, not with what the customer
    paid (discounts must not make a different-priced item look equal).
    """
    original = find_product(state, original_sku)
    if original is None:
        return [SKU_NOT_FOUND]
    replacement = find_product(state, replacement_sku)
    if replacement is None:
        return [SIZE_NOT_OFFERED]

    reasons: list[str] = []
    if replacement.get("product_id") != original.get("product_id"):
        reasons.append(DIFFERENT_PRODUCT)
    elif same_size(replacement.get("size"), original.get("size")):
        reasons.append(SAME_SIZE)
    original_price, price = original.get("price_cents"), replacement.get("price_cents")
    if not (is_int(original_price) and is_int(price) and price == original_price):
        reasons.append(PRICE_MISMATCH)
    stock = replacement.get("stock")
    if not (is_int(stock) and stock > 0):
        reasons.append(OUT_OF_STOCK)
    return reasons


def options_for_sku(state: dict[str, Any], original_sku: str) -> list[ExchangeOption]:
    """Every other size of the original product, valid or not, in ladder order."""
    original = find_product(state, original_sku)
    if original is None:
        return []
    options = [
        _option(state, original_sku, product)
        for product in state.get("products", [])
        if product.get("product_id") == original.get("product_id")
        and not same_size(product.get("size"), original.get("size"))
    ]
    return sorted(options, key=lambda option: size_rank(option.size) or (2, 0.0))


def valid_options(options: list[ExchangeOption]) -> list[ExchangeOption]:
    return [option for option in options if option.valid]


def _option(state: dict[str, Any], original_sku: str, product: dict[str, Any]) -> ExchangeOption:
    reasons = validate_replacement(state, original_sku, product.get("sku"))
    return ExchangeOption(
        sku=product.get("sku"),
        size=product.get("size"),
        price_cents=product.get("price_cents"),
        stock=product.get("stock"),
        valid=not reasons,
        reasons=reasons,
    )
