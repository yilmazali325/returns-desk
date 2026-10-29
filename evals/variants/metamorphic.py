"""Metamorphic transforms: rewrites that must not change any decision or business outcome.

- `rename_ids` consistently renames every customer, order, line, SKU, product and request ID in
  the state, faults, intents and message text. A desk keyed to the public fixtures (C100, O100,
  L1, TEE-M) fails here even when the wording is familiar.
- `shift_dates` moves as_of and every delivered_on by the same number of days. Day differences are
  preserved exactly, so eligibility must not change; it catches clock reads and broken date math.
Product IDs keep a garment word ("BASIC-TEE") because the contract's products carry no names: the
garment word in the ID is the only link between "my tee" and the catalog.
"""
from __future__ import annotations

import copy
import re
from datetime import date, timedelta

PRODUCT_RENAMES = {"TEE": "BASIC-TEE", "JACKET": "RAIN-JACKET", "HOODIE": "ZIP-HOODIE",
                   "JEANS": "SLIM-JEANS", "SNEAKER": "TRAIL-SNEAKER", "SCARF": "WOOL-SCARF"}


def id_mapping(case: dict, salt: int) -> dict[str, str]:
    """Deterministic old -> new ID map for everything the case references."""
    state = case["initial_state"]
    mapping: dict[str, str] = {}
    customers = [case["customer_id"]] + [o["customer_id"] for o in state["orders"]]
    for n, customer in enumerate(dict.fromkeys(customers), start=1):
        mapping[customer] = f"K{salt}{n:02d}"
    for n, order in enumerate(state["orders"], start=1):
        mapping[order["order_id"]] = f"ORD-{salt}{n:03d}"
    lines = [line["line_id"] for order in state["orders"] for line in order["lines"]]
    for n, line_id in enumerate(dict.fromkeys(lines), start=1):
        mapping[line_id] = f"LN{salt}-{n}"
    for n, product in enumerate(state["products"], start=1):
        mapping[product["sku"]] = f"SKU-{salt}{n:03d}"
        mapping.setdefault(product["product_id"], PRODUCT_RENAMES.get(product["product_id"], f"P{salt}-{product['product_id']}"))
    mapping[case["request_id"]] = f"RQ-{salt}-{case['case_id']}"
    return mapping


def _rewrite_text(text: str, mapping: dict[str, str]) -> str:
    # Longest first so "L1" never rewrites inside "L10"; boundaries keep tokens whole.
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9])",
                         re.IGNORECASE)
    folded = {key.casefold(): value for key, value in mapping.items()}

    def replace(match: re.Match) -> str:
        new = folded[match.group(1).casefold()]
        return new.lower() if match.group(1).islower() else new

    return pattern.sub(replace, text)


def rename_ids(case: dict, intents: list[dict], salt: int, new_case_id: str) -> tuple[dict, list[dict]]:
    mapping = id_mapping(case, salt)

    def m(value):
        return mapping.get(value, value)

    new = copy.deepcopy(case)
    new["case_id"] = new_case_id
    new["customer_id"] = m(case["customer_id"])
    new["request_id"] = m(case["request_id"])
    state = new["initial_state"]
    for product in state["products"]:
        product["sku"], product["product_id"] = m(product["sku"]), m(product["product_id"])
    for order in state["orders"]:
        order["order_id"], order["customer_id"] = m(order["order_id"]), m(order["customer_id"])
        for line in order["lines"]:
            line["line_id"], line["sku"] = m(line["line_id"]), m(line["sku"])
    for fault in new["faults"]:
        if "sku" in fault:
            fault["sku"] = m(fault["sku"])
    # Messages mention only IDs; product IDs are left alone in text because customers say "tee".
    text_mapping = {k: v for k, v in mapping.items() if k not in {p["product_id"] for p in case["initial_state"]["products"]}}
    for turn in new["turns"]:
        turn["message"] = _rewrite_text(turn["message"], text_mapping)
    new_intents = []
    for intent in intents:
        renamed = dict(intent)
        for key in ("order_id", "line_id", "requested_sku", "requested_product"):
            if renamed.get(key) is not None:
                renamed[key] = m(renamed[key])
        new_intents.append(renamed)
    return new, new_intents


def shift_dates(case: dict, days: int, new_case_id: str) -> dict:
    new = copy.deepcopy(case)
    new["case_id"] = new_case_id
    new["request_id"] = f"{case['request_id']}-shift{days:+d}"
    state = new["initial_state"]
    state["as_of"] = _shift(state["as_of"], days)
    for order in state["orders"]:
        if order.get("delivered_on"):
            order["delivered_on"] = _shift(order["delivered_on"], days)
    return new


def _shift(iso_day: str, days: int) -> str:
    return (date.fromisoformat(iso_day) + timedelta(days=days)).isoformat()
