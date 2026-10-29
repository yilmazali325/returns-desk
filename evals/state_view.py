"""Business view of a contract state, and a readable diff between two views.

Graders compare what the business cares about -- stock, returned flags, money and the records that
prove it -- not the IDs a candidate generates or the wording it uses. `business_view` projects a
final_state onto exactly those fields; `diff_views` explains every difference in one line each, so
a failing case can be understood without opening the JSON.
"""
from __future__ import annotations

import copy
import json
from typing import Any

# Fields that identify a record's business meaning. Generated IDs (return_id, refund_id,
# transaction_id, proposal_id, ...) are deliberately absent so that any ID scheme compares equal.
RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    "returns": ("request_id", "customer_id", "order_id", "line_id", "remedy", "sku"),
    "refunds": ("request_id", "customer_id", "order_id", "line_id", "amount_cents"),
    "exchanges": ("request_id", "customer_id", "order_id", "line_id", "original_sku", "replacement_sku"),
    "quarantine": ("order_id", "line_id", "sku", "quantity"),
}
IDEMPOTENCY_FIELDS = ("status", "action", "customer_id", "order_id", "line_id", "amount_cents", "replacement_sku")

# Lists whose items are matched by an identity field rather than by position in diffs.
_IDENTITY_KEYS = ("sku", "order_id", "line_id")
_MAX_SNIPPET = 140


def business_view(state: dict) -> dict:
    """Project a contract state onto the fields a grader compares (no generated IDs)."""
    view: dict[str, Any] = {
        "top_level_keys": sorted(state),
        "as_of": state.get("as_of"),
        "currency": state.get("currency"),
        "products": state.get("products"),
        "orders": state.get("orders"),
    }
    for name, fields in RECORD_FIELDS.items():
        view[name] = _project_records(state.get(name), fields)
    view["idempotency"] = _project_idempotency(state.get("idempotency"))
    return copy.deepcopy(view)


def _project(record: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(record, dict):
        return record  # keep malformed entries visible in the diff
    return {name: record.get(name) for name in fields}


def _project_records(records: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(records, list):
        return records
    projected = [_project(record, fields) for record in records]
    # Order of appended records is not business meaning; sort for a stable comparison.
    return sorted(projected, key=lambda item: json.dumps(item, sort_keys=True))


def _project_idempotency(entries: Any) -> Any:
    if not isinstance(entries, dict):
        return entries
    return {key: _project(entries[key], IDEMPOTENCY_FIELDS) for key in sorted(entries)}


def snippet(value: Any) -> str:
    """Compact JSON for a diff line, truncated so one line stays readable."""
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= _MAX_SNIPPET else text[: _MAX_SNIPPET - 3] + "..."


def diff_views(expected: Any, actual: Any, path: str = "state") -> list[str]:
    """Return one human-readable line per difference between two JSON-like values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _diff_dicts(expected, actual, path)
    if isinstance(expected, list) and isinstance(actual, list):
        key = _identity_key(expected, actual)
        if key is not None:
            return _diff_keyed_lists(expected, actual, path, key)
        return _diff_positional_lists(expected, actual, path)
    # type() matters: True == 1 and 3000 == 3000.0 in Python, but not in the contract.
    if expected != actual or type(expected) is not type(actual):
        return [f"{path}: expected {snippet(expected)}, got {snippet(actual)}"]
    return []


def _diff_dicts(expected: dict, actual: dict, path: str) -> list[str]:
    lines: list[str] = []
    for key in expected:
        if key not in actual:
            lines.append(f"{path}.{key}: missing (expected {snippet(expected[key])})")
        else:
            lines.extend(diff_views(expected[key], actual[key], f"{path}.{key}"))
    for key in actual:
        if key not in expected:
            lines.append(f"{path}.{key}: unexpected {snippet(actual[key])}")
    return lines


def _identity_key(expected: list, actual: list) -> str | None:
    items = expected + actual
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    for key in _IDENTITY_KEYS:
        values = [item.get(key) for item in items]
        unique_within_each = all(
            len({item.get(key) for item in side}) == len(side) for side in (expected, actual)
        )
        if all(isinstance(value, str) for value in values) and unique_within_each:
            return key
    return None


def _diff_keyed_lists(expected: list, actual: list, path: str, key: str) -> list[str]:
    lines: list[str] = []
    actual_by_key = {item[key]: item for item in actual}
    expected_keys = {item[key] for item in expected}
    for item in expected:
        label = f"{path}[{item[key]}]"
        if item[key] not in actual_by_key:
            lines.append(f"{label}: missing")
        else:
            lines.extend(diff_views(item, actual_by_key[item[key]], label))
    for item in actual:
        if item[key] not in expected_keys:
            lines.append(f"{path}[{item[key]}]: unexpected {snippet(item)}")
    return lines


def _diff_positional_lists(expected: list, actual: list, path: str) -> list[str]:
    lines: list[str] = []
    if len(expected) != len(actual):
        lines.append(f"{path}: expected {len(expected)} item(s), got {len(actual)}")
    for index, (want, got) in enumerate(zip(expected, actual)):
        lines.extend(diff_views(want, got, f"{path}[{index}]"))
    for index in range(len(actual), len(expected)):
        lines.append(f"{path}[{index}]: missing {snippet(expected[index])}")
    for index in range(len(expected), len(actual)):
        lines.append(f"{path}[{index}]: unexpected {snippet(actual[index])}")
    return lines
