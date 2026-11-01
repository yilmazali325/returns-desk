"""Deterministic generator for the variant suite: contract inputs plus oracle labels.

    python -m evals.variants.generate [--out-dir evals/variants]

Writes variants_v1.json (contract_version 1 + cases, the exact input schema of
candidate_assets/public_cases.json) and labels_v1.json (per case: family, intents, expected
decisions, proposals, events and final business state). Each family exercises one disclosed rule
or conversation pattern with 3+ different hand-written phrasings (evals/variants/testdata/phrases.py).

Double entry: every case states the decisions its family is about (`expect=`), and the generator
refuses to write a case where the oracle disagrees. A typo in a world (wrong date, wrong stock)
therefore fails loudly instead of quietly turning a 'deny' family into a 'propose' family.
Randomness is seeded per family, so output is byte-identical across runs and adding a family
does not reshuffle the others.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterator

from evals.oracle import build_expectation
from evals.variants.testdata import phrases as P
from evals.variants.metamorphic import rename_ids, shift_dates

SEED = 20261027
OUT_DIR = Path(__file__).resolve().parent
STATE_KEYS = ("as_of", "currency", "products", "orders", "returns", "refunds", "exchanges", "quarantine", "idempotency")


# --------------------------------------------------------------------------- worlds

@dataclass(frozen=True)
class Scheme:
    """One ID style for a customer's account, so no family is tied to C100/O100/L1."""
    customer: str
    orders: tuple[str, ...]
    lines: tuple[str, ...]
    other_customer: str
    other_order: str
    missing_order: str

    def bogus_line(self) -> str:
        return self.lines[0][:-1] + "9"


SCHEMES = (
    Scheme("C100", ("O100", "O101", "O102"), ("L1", "L2", "L3"), "C200", "O200", "O999"),
    Scheme("C-2047", ("O-58812", "O-58840", "O-58901"), ("L1", "L2", "L3"), "C-3310", "O-60017", "O-99999"),
    Scheme("CUST77", ("W2207", "W2215", "W2230"), ("LN-1", "LN-2", "LN-3"), "CUST78", "W3301", "W9999"),
    Scheme("C777", ("A-9", "A-12", "A-15"), ("P1", "P2", "P3"), "C778", "A-31", "A-404"),
)
AS_OF_POOL = (date(2026, 10, 26), date(2026, 11, 30), date(2027, 1, 5), date(2027, 6, 15), date(2028, 3, 1))


def products(product_id: str, price: int, stocks: dict[str, int], prices: dict[str, int] | None = None) -> list[dict]:
    return [{"sku": f"{product_id}-{size}", "product_id": product_id, "size": size,
             "price_cents": (prices or {}).get(size, price), "stock": stock} for size, stock in stocks.items()]


def base_catalog(**stock: int) -> list[dict]:
    """The public catalog (TEE S/M/L 4000, JACKET M/L 9000) with optional stock overrides by size key."""
    tee = products("TEE", 4000, {"S": stock.get("S", 4), "M": stock.get("M", 2), "L": stock.get("L", 3)})
    jacket = products("JACKET", 9000, {"M": stock.get("JM", 1), "L": stock.get("JL", 0)})
    return tee + jacket


def hoodies(**stock: int) -> list[dict]:
    sizes = {"XS": 2, "S": 3, "M": 2, "L": 4, "XL": 1}
    return products("HOODIE", 5500, {size: stock.get(size, n) for size, n in sizes.items()}, prices={"XL": 6500})


def jeans(**stock: int) -> list[dict]:
    sizes = {"28": 1, "30": 3, "32": 2, "34": 5, "36": 2}
    return products("JEANS", 7900, {size: stock.get(f"w{size}", n) for size, n in sizes.items()}, prices={"36": 8900})


def sneakers(**stock: int) -> list[dict]:
    sizes = {"8": 2, "9": 1, "10": 3, "11": 0}
    return products("SNEAKER", 11000, {size: stock.get(f"s{size}", n) for size, n in sizes.items()})


def line(line_id: str, sku: str, paid: int, **flags: bool) -> dict:
    return {"line_id": line_id, "sku": sku, "paid_cents": paid, "unopened": flags.get("unopened", True),
            "final_sale": flags.get("final_sale", False), "returned": flags.get("returned", False)}


def order(order_id: str, customer_id: str, delivered_on: str | None, lines: list[dict], status: str = "delivered") -> dict:
    return {"order_id": order_id, "customer_id": customer_id, "status": status, "delivered_on": delivered_on, "lines": lines}


def ago(as_of: date, days: int) -> str:
    return (as_of - timedelta(days=days)).isoformat()


def make_state(as_of: date, catalog: list[dict], orders: list[dict]) -> dict:
    values = {"as_of": as_of.isoformat(), "currency": "USD", "products": catalog, "orders": orders,
              "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}
    return {key: values[key] for key in STATE_KEYS}


# --------------------------------------------------------------------------- text helpers

def fmt_id(value: str, style: str) -> str:
    return {"plain": value, "hash": "#" + value, "lower": value.lower()}[style]


def size_word(size: str) -> str:
    return f"size {size}" if size.isdigit() else P.SIZE_WORDS.get(size, size)


def size_phrase(size: str, style: int) -> str:
    """'size L' / 'a large' / 'the large' / 'an L'; numeric sizes always keep the word 'size'."""
    if size.isdigit():
        return (f"size {size}", f"a size {size}", f"the size {size}")[style % 3]
    word = P.SIZE_WORDS.get(size, size)
    article = "an" if word[0] in "aeiou" else "a"
    return (f"size {size}", f"{article} {word}", f"the {word}", f"an {size}")[style % 4]


def cycle(rng: random.Random, bank: list[str]) -> Iterator[str]:
    """Shuffled, repeating stream: consecutive draws are distinct until the bank is exhausted."""
    items = list(bank)
    rng.shuffle(items)
    return itertools.cycle(items)


@dataclass
class Turn:
    message: str
    intent: dict
    confirm: bool = False


def say(message: str, kind: str, **fields) -> Turn:
    return Turn(message, {"kind": kind, **fields})


def yes(message: str) -> Turn:
    return Turn(message, {"kind": "confirm"}, confirm=True)


# --------------------------------------------------------------------------- builder

class SuiteBuilder:
    def __init__(self) -> None:
        self.cases: list[dict] = []
        self.labels: list[dict] = []
        self._count: Counter = Counter()

    @staticmethod
    def rng(family: str) -> random.Random:
        return random.Random(f"{SEED}:{family}")

    def add(self, family: str, *, customer: str, as_of: date, catalog: list[dict], orders: list[dict],
            turns: list[Turn], expect: list[str], faults: list[dict] | None = None, note: str = "") -> dict:
        self._count[family] += 1
        case_id = f"v1_{family}_{self._count[family]:02d}"
        case = {
            "case_id": case_id, "customer_id": customer, "request_id": f"REQ-{case_id}",
            "initial_state": make_state(as_of, catalog, orders),
            "turns": [{"message": t.message, **({"confirm_proposal": "last"} if t.confirm else {})} for t in turns],
            "faults": list(faults or []),
        }
        return self.add_case(family, case, [t.intent for t in turns], expect, note)

    def add_case(self, family: str, case: dict, intents: list[dict], expect: list[str], note: str) -> dict:
        expected = build_expectation(case, intents)
        if expected["decisions"] != expect:
            raise AssertionError(f"{case['case_id']}: oracle decided {expected['decisions']}, family expects {expect}")
        self.cases.append(case)
        self.labels.append({"case_id": case["case_id"], "family": family, "note": note, "intents": intents,
                            "expected": expected})
        return case


# --------------------------------------------------------------------------- families: refund rules

def refund_windows(b: SuiteBuilder) -> None:
    """Inclusive 0-30 day window, discounted paid_cents, calendar boundaries."""
    specs = [  # family, as_of, days, catalog, sku, paid
        ("refund_day0", date(2026, 10, 26), 0, base_catalog(), "TEE-M", 2599),
        ("refund_day0", date(2027, 2, 14), 0, hoodies(), "HOODIE-L", 4950),
        ("refund_day0", date(2026, 12, 31), 0, jeans(), "JEANS-32", 6320),
        ("refund_mid_window", date(2026, 10, 26), 7, base_catalog(), "JACKET-M", 7200),
        ("refund_mid_window", date(2027, 6, 15), 15, sneakers(), "SNEAKER-10", 8800),
        ("refund_mid_window", date(2026, 11, 30), 23, base_catalog(), "TEE-L", 4000),
        ("refund_day30", date(2026, 10, 26), 30, base_catalog(), "TEE-S", 2999),
        ("refund_day30", date(2028, 3, 1), 30, hoodies(), "HOODIE-M", 5500),  # leap year: delivered 2028-01-31
        ("refund_day30", date(2027, 1, 5), 30, base_catalog(), "TEE-M", 3150),  # year boundary: 2026-12-06
        ("refund_day30", date(2026, 11, 30), 30, jeans(), "JEANS-30", 7110),  # month end: 2026-10-31
    ]
    for family, as_of, days, catalog, sku, paid in specs:
        rng = b.rng(f"{family}:{as_of}")
        scheme = SCHEMES[len(b.cases) % len(SCHEMES)]
        o, l = fmt_id(scheme.orders[0], rng.choice(["plain", "hash", "lower"])), scheme.lines[0]
        template = rng.choice(P.REFUND_EXPLICIT)
        b.add(family, customer=scheme.customer, as_of=as_of, catalog=catalog,
              orders=[order(scheme.orders[0], scheme.customer, ago(as_of, days), [line(l, sku, paid)])],
              turns=[say(template.format(o=o, l=l), "refund", order_id=scheme.orders[0], line_id=l),
                     yes(rng.choice(P.CONFIRM))],
              expect=["propose", "completed"], note=f"delivered {days} days before as_of; refund exactly {paid}")


def refund_expired(b: SuiteBuilder) -> None:
    rng = b.rng("refund_expired")
    templates = cycle(rng, P.REFUND_EXPLICIT + P.RETURN_ONLY)
    specs = [(date(2026, 10, 26), 31, False), (date(2028, 3, 1), 31, True), (date(2026, 3, 1), 31, False),
             (date(2027, 6, 15), 45, True), (date(2026, 11, 30), 200, False)]
    for n, (as_of, days, then_confirm) in enumerate(specs):
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        turns = [say(next(templates).format(o=fmt_id(o, "plain"), l=l), "refund", order_id=o, line_id=l)]
        if then_confirm:  # confirm_proposal with nothing proposed must not mutate anything
            turns.append(yes(rng.choice(P.CONFIRM)))
        b.add("refund_expired", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, days), [line(l, "TEE-M", 3000)])],
              turns=turns, expect=["deny"] * len(turns), note=f"{days} days > 30")


def _single_line_family(b: SuiteBuilder, family: str, bank: list[str], *, days: int = 10, status: str = "delivered",
                        delivered: bool = True, flags: dict | None = None, count: int = 3, note: str = "") -> None:
    """One owned single-line order that is ineligible for one reason; every phrasing must deny."""
    rng = b.rng(family)
    templates = cycle(rng, bank)
    for n in range(count):
        scheme = SCHEMES[(n + 1) % len(SCHEMES)]
        as_of = AS_OF_POOL[n % len(AS_OF_POOL)]
        o, l = scheme.orders[0], scheme.lines[0]
        style = ("plain", "lower", "hash")[n % 3]
        delivered_on = ago(as_of, days) if delivered else None
        b.add(family, customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, delivered_on, [line(l, "TEE-M", 3000, **(flags or {}))], status)],
              turns=[say(next(templates).format(o=fmt_id(o, style), l=l), "refund", order_id=o, line_id=l)],
              expect=["deny"], note=note)


def refund_ineligible(b: SuiteBuilder) -> None:
    _single_line_family(b, "future_delivery", P.FUTURE_DELIVERY, days=-1, note="delivered_on after as_of")
    # Two more future offsets so the family is not a single boundary point.
    rng = b.rng("future_delivery:more")
    for n, days in enumerate((-3, -10)):
        scheme = SCHEMES[n]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add("future_delivery", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(date(2026, 10, 26), days), [line(l, "TEE-L", 4000)])],
              turns=[say(rng.choice(P.REFUND_EXPLICIT).format(o=o, l=l), "refund", order_id=o, line_id=l)],
              expect=["deny"], note=f"delivered {-days} days in the future")
    _single_line_family(b, "opened", P.OPENED, flags={"unopened": False}, note="unopened=false beats the customer's claim")
    _single_line_family(b, "already_returned", P.ALREADY_RETURNED, flags={"returned": True},
                        note="returned=true with no records: deny, never a second return")
    _single_line_family(b, "final_sale", P.FINAL_SALE_REFUND, flags={"final_sale": True}, count=2, note="final sale")
    scheme = SCHEMES[2]
    o, l = scheme.orders[0], scheme.lines[0]
    b.add("final_sale", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(date(2026, 10, 26), 5), [line(l, "TEE-M", 3000, final_sale=True)])],
          turns=[say(P.EXCHANGE_EXPLICIT[2].format(o=o, l=l, sz=size_phrase("L", 0), g="tee"), "exchange",
                     order_id=o, line_id=l, requested_size="L")],
          expect=["deny"], note="final sale blocks exchanges too")
    rng = b.rng("not_delivered")
    for n, status in enumerate(("shipped", "processing", "cancelled")):
        scheme = SCHEMES[n]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add("not_delivered", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
              orders=[order(o, scheme.customer, None, [line(l, "TEE-M", 3000)], status)],
              turns=[say(P.NOT_DELIVERED[n].format(o=fmt_id(o, rng.choice(["plain", "lower"])), l=l),
                         "refund", order_id=o, line_id=l)],
              expect=["deny"], note=f"status {status}, delivered_on null")


def not_on_account(b: SuiteBuilder) -> None:
    """Another customer's order and a nonexistent order must be refused identically (no leak)."""
    rng = b.rng("foreign_order")
    templates = cycle(rng, P.FOREIGN_ORDER)
    for n in range(3):
        scheme = SCHEMES[n]
        as_of = AS_OF_POOL[n]
        own = order(scheme.orders[0], scheme.customer, ago(as_of, 5), [line(scheme.lines[0], "TEE-S", 4000)])
        foreign = order(scheme.other_order, scheme.other_customer, ago(as_of, 5), [line(scheme.lines[0], "JACKET-M", 9000)])
        message = next(templates).format(o=scheme.other_order, l=scheme.lines[0], other=scheme.other_customer)
        b.add("foreign_order", customer=scheme.customer, as_of=as_of, catalog=base_catalog(), orders=[own, foreign],
              turns=[say(message, "refund", order_id=scheme.other_order, line_id=scheme.lines[0])],
              expect=["deny"], note="the named order is eligible but belongs to someone else")
    rng = b.rng("nonexistent_order")
    templates = cycle(rng, P.MISSING_ORDER)
    for n in range(3):
        scheme = SCHEMES[(n + 1) % len(SCHEMES)]
        as_of = AS_OF_POOL[n + 1]
        own = order(scheme.orders[0], scheme.customer, ago(as_of, 3), [line(scheme.lines[0], "TEE-M", 3000)])
        message = next(templates).format(o=fmt_id(scheme.missing_order, ("plain", "hash", "lower")[n]), l=scheme.lines[0])
        b.add("nonexistent_order", customer=scheme.customer, as_of=as_of, catalog=base_catalog(), orders=[own],
              turns=[say(message, "refund", order_id=scheme.missing_order, line_id=scheme.lines[0])],
              expect=["deny"], note="order exists nowhere; same refusal as a foreign order")
    rng = b.rng("nonexistent_line")
    templates = cycle(rng, P.NONEXISTENT_LINE)
    for n in range(3):
        scheme = SCHEMES[n]
        lines = [line(scheme.lines[0], "TEE-M", 3000)] + ([line(scheme.lines[1], "JACKET-M", 9000)] if n < 2 else [])
        o = scheme.orders[0]
        b.add("nonexistent_line", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(date(2026, 10, 26), 4), lines)],
              turns=[say(next(templates).format(o=o, l=scheme.bogus_line()), "refund", order_id=o, line_id=scheme.bogus_line())],
              expect=["clarify"], note="owned order, unknown line: ask which line, never guess (deny accepted)")


# --------------------------------------------------------------------------- families: exchanges

def _exchange_case(b: SuiteBuilder, family: str, n: int, *, catalog: list[dict], sku: str, paid: int, garment: str,
                   template: str, requested: dict, expect: list[str], size_text: str = "", confirm: bool = True,
                   days: int = 9, note: str = "", faults: list[dict] | None = None, extra_turns: list[Turn] = ()) -> dict:
    scheme = SCHEMES[n % len(SCHEMES)]
    as_of = AS_OF_POOL[n % len(AS_OF_POOL)]
    o, l = scheme.orders[0], scheme.lines[0]
    cur = next(p["size"] for p in catalog if p["sku"] == sku) if any(p["sku"] == sku for p in catalog) else "M"
    message = template.format(o=fmt_id(o, ("plain", "lower", "hash")[n % 3]), l=l, g=garment, sz=size_text,
                              cur=size_word(cur), g2=requested.get("_g2", ""))
    intent = {k: v for k, v in requested.items() if not k.startswith("_")}
    turns = [say(message, "exchange", order_id=o, line_id=l, **intent)]
    if confirm:
        turns.append(yes(P.CONFIRM[(n * 5) % len(P.CONFIRM)]))
    turns += list(extra_turns)
    return b.add(family, customer=scheme.customer, as_of=as_of, catalog=catalog,
                 orders=[order(o, scheme.customer, ago(as_of, days), [line(l, sku, paid)])],
                 turns=turns, expect=expect, faults=faults, note=note)


def exchanges(b: SuiteBuilder) -> None:
    rng = b.rng("exchange_valid")
    templates = cycle(rng, P.EXCHANGE_EXPLICIT)
    valid = [  # catalog, original sku, paid, garment, target size
        (base_catalog(), "TEE-M", 3000, "tee", "S"), (base_catalog(), "TEE-M", 2799, "t-shirt", "L"),
        (hoodies(), "HOODIE-S", 5500, "hoodie", "M"), (jeans(), "JEANS-32", 6900, "jeans", "34"),
        (sneakers(), "SNEAKER-9", 9900, "sneakers", "10"), (base_catalog(), "TEE-L", 4000, "shirt", "M"),
    ]
    for n, (catalog, sku, paid, garment, size) in enumerate(valid):
        _exchange_case(b, "exchange_valid", n, catalog=catalog, sku=sku, paid=paid, garment=garment,
                       template=next(templates), requested={"requested_size": size}, size_text=size_phrase(size, n),
                       expect=["propose", "completed"], note=f"{sku} -> size {size}: same product, same catalog price, in stock")

    def deny_family(family: str, bank: list[str], specs: list[tuple], note: str) -> None:
        templates = cycle(b.rng(family), bank)
        for n, (catalog, sku, garment, requested, size_text) in enumerate(specs):
            _exchange_case(b, family, n, catalog=catalog, sku=sku, paid=3500, garment=garment, template=next(templates),
                           requested=requested, size_text=size_text, expect=["deny"], confirm=False, note=note)

    deny_family("exchange_out_of_stock", P.EXCHANGE_EXPLICIT, [
        (base_catalog(), "JACKET-M", "jacket", {"requested_size": "L"}, size_phrase("L", 1)),
        (hoodies(L=0), "HOODIE-M", "hoodie", {"requested_size": "L"}, size_phrase("L", 0)),
        (sneakers(), "SNEAKER-10", "shoes", {"requested_size": "11"}, size_phrase("11", 0)),
    ], "requested replacement has stock 0: deny, alternatives in text only, no alternative proposal")
    deny_family("exchange_price_mismatch", P.PRICE_MISMATCH, [
        (base_catalog() + products("TEE", 4500, {"XL": 5}), "TEE-M", "tee", {"requested_size": "XL"}, size_phrase("XL", 3)),
        (jeans(), "JEANS-34", "jeans", {"requested_size": "36"}, size_phrase("36", 0)),
        (hoodies(), "HOODIE-M", "hoodie", {"requested_size": "XL"}, size_phrase("XL", 1)),
    ], "replacement catalog price differs from the original's: no price-difference payments in scope")
    deny_family("exchange_same_size", P.SAME_SIZE, [
        (base_catalog(), "TEE-M", "tee", {"requested_size": "M"}, size_phrase("M", 1)),
        (jeans(), "JEANS-32", "jeans", {"requested_size": "32"}, size_phrase("32", 0)),
        (hoodies(), "HOODIE-S", "hoodie", {"requested_size": "S"}, size_phrase("S", 0)),
    ], "an exchange must change the size")
    # Templates are paired by hand here: only the second one names a size, so only it may bind a SKU.
    different = [
        (P.DIFFERENT_PRODUCT[0], base_catalog(), "TEE-M", "tee", {"requested_product": "JACKET", "_g2": "jacket"}, ""),
        (P.DIFFERENT_PRODUCT[1], hoodies() + jeans(), "HOODIE-M", "hoodie",
         {"requested_sku": "JEANS-32", "_g2": "pair of jeans"}, size_phrase("32", 0)),
        (P.DIFFERENT_PRODUCT[2], base_catalog() + jeans(), "JEANS-30", "jeans", {"requested_product": "TEE", "_g2": "tee"}, ""),
    ]
    for n, (template, catalog, sku, garment, requested, size_text) in enumerate(different):
        _exchange_case(b, "exchange_different_product", n, catalog=catalog, sku=sku, paid=3500, garment=garment,
                       template=template, requested=requested, size_text=size_text, expect=["deny"], confirm=False,
                       note="exchange must stay within the same product_id")
    deny_family("exchange_size_not_offered", P.EXCHANGE_EXPLICIT, [
        (base_catalog(), "TEE-M", "tee", {"requested_size": "XL"}, size_phrase("XL", 1)),
        (jeans(), "JEANS-32", "jeans", {"requested_size": "38"}, size_phrase("38", 1)),
        (hoodies(), "HOODIE-L", "hoodie", {"requested_size": "XXL"}, size_phrase("XXL", 0)),
    ], "requested size is not in the catalog for this product")

    # No size named: 2+ valid -> clarify then answer; exactly 1 -> propose it; 0 -> deny.
    templates = cycle(b.rng("exchange_no_size_multi"), P.EXCHANGE_NO_SIZE)
    answers = cycle(b.rng("exchange_no_size_multi:answer"), P.SIZE_ANSWER)
    for n, (catalog, sku, garment, size) in enumerate([(base_catalog(), "TEE-M", "tee", "S"),
                                                       (hoodies(), "HOODIE-M", "hoodie", "L"),
                                                       (jeans(), "JEANS-32", "jeans", "30")]):
        scheme = SCHEMES[n]
        word = size_word(size)
        answer = next(answers).format(sz=size_phrase(size, n), word=word, Word=word.capitalize())
        _exchange_case(b, "exchange_no_size_multi", n, catalog=catalog, sku=sku, paid=3900, garment=garment,
                       template=next(templates), requested={}, confirm=False, expect=["clarify", "propose", "completed"],
                       extra_turns=[say(answer, "exchange", order_id=scheme.orders[0], line_id=scheme.lines[0],
                                        requested_size=size), yes(P.CONFIRM[n + 3])],
                       note="no size given and 2+ valid sizes: clarify, then the answer is proposed")
    templates = cycle(b.rng("exchange_no_size_single"), P.EXCHANGE_NO_SIZE)
    for n, (catalog, sku, garment) in enumerate([(base_catalog(S=0), "TEE-M", "tee"),
                                                 (base_catalog(), "JACKET-L", "jacket"),
                                                 (sneakers(s8=0), "SNEAKER-9", "sneakers")]):
        _exchange_case(b, "exchange_no_size_single", n, catalog=catalog, sku=sku, paid=4000, garment=garment,
                       template=next(templates), requested={}, expect=["propose", "completed"],
                       note="no size given and exactly one valid size: propose it")
    templates = cycle(b.rng("exchange_no_size_none"), P.EXCHANGE_NO_SIZE)
    for n, (catalog, sku, garment) in enumerate([(base_catalog(), "JACKET-M", "jacket"),
                                                 (base_catalog() + products("SCARF", 2500, {"OS": 6}), "SCARF-OS", "scarf"),
                                                 (base_catalog(S=0, L=0), "TEE-M", "tee")]):
        _exchange_case(b, "exchange_no_size_none", n, catalog=catalog, sku=sku, paid=2500, garment=garment,
                       template=next(templates), requested={}, confirm=False, expect=["deny"],
                       note="no valid replacement size exists at all")

    templates = cycle(b.rng("exchange_relative_size"), P.UP_REQUESTS)
    down = cycle(b.rng("exchange_relative_size:down"), P.DOWN_REQUESTS)
    relative = [(base_catalog(), "TEE-M", "tee", "UP", templates), (jeans(), "JEANS-32", "jeans", "UP", templates),
                (base_catalog(), "TEE-M", "t-shirt", "DOWN", down), (hoodies(), "HOODIE-L", "hoodie", "DOWN", down)]
    for n, (catalog, sku, garment, direction, bank) in enumerate(relative):
        _exchange_case(b, "exchange_relative_size", n, catalog=catalog, sku=sku, paid=4100, garment=garment,
                       template=next(bank), requested={"requested_size": direction}, expect=["propose", "completed"],
                       note=f"'{direction.lower()}' means the next offered size {direction.lower()}; 'too small' is not size S")

    # Original SKU missing from the catalog: an exchange is impossible, a refund is unaffected.
    templates = cycle(b.rng("discontinued_sku"), P.EXCHANGE_EXPLICIT)
    for n in range(2):
        _exchange_case(b, "discontinued_sku", n, catalog=base_catalog(), sku="TEE-M-2019", paid=2800, garment="tee",
                       template=next(templates), requested={"requested_size": "L"}, size_text=size_phrase("L", n),
                       confirm=False, expect=["deny"], note="original SKU not in products: exchange impossible")
    scheme = SCHEMES[2]
    o, l = scheme.orders[0], scheme.lines[0]
    b.add("discontinued_sku", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(date(2026, 10, 26), 6), [line(l, "TEE-M-2019", 2800)])],
          turns=[say(P.REFUND_EXPLICIT[2].format(o=o, l=l), "refund", order_id=o, line_id=l), yes(P.CONFIRM[0])],
          expect=["propose", "completed"], note="refund of a discontinued SKU still works for paid_cents")


# --------------------------------------------------------------------------- families: targets and clarification

def clarification(b: SuiteBuilder) -> None:
    as_of = date(2026, 10, 26)
    vague = cycle(b.rng("clarify_multi_line"), P.ORDER_ONLY_VAGUE)
    by_line = cycle(b.rng("clarify_multi_line:t2"), P.TARGET_BY_LINE)
    for n, (first, second, pick) in enumerate([(("TEE-M", "tee", 3000), ("JACKET-M", "jacket", 9000), 0),
                                               (("HOODIE-S", "hoodie", 5500), ("JEANS-32", "jeans", 7900), 1),
                                               (("TEE-S", "tee", 4000), ("TEE-L", "tee", 4000), 1)]):
        scheme = SCHEMES[n]
        o = scheme.orders[0]
        lines = [line(scheme.lines[0], first[0], first[2]), line(scheme.lines[1], second[0], second[2])]
        target = (first, second)[pick]
        catalog = base_catalog() + hoodies() + jeans()
        cur = next(p["size"] for p in catalog if p["sku"] == target[0])
        b.add("clarify_multi_line", customer=scheme.customer, as_of=as_of, catalog=catalog,
              orders=[order(o, scheme.customer, ago(as_of, 12), lines)],
              turns=[say(next(vague).format(o=fmt_id(o, ("plain", "hash", "lower")[n])), "clarify_needed"),
                     say(next(by_line).format(l=scheme.lines[pick], g=target[1], cur=size_word(cur)), "refund",
                         order_id=o, line_id=scheme.lines[pick]),
                     yes(P.CONFIRM[n])],
              expect=["clarify", "propose", "completed"], note="order named, two lines: ask which one first")

    # Two orders that each contain a matching item.
    desc = cycle(b.rng("clarify_multi_order"), P.DESC_NO_ORDER)
    which = cycle(b.rng("clarify_multi_order:t2"), P.WHICH_ORDER_ANSWER)
    for n, (sku, garment, paid_b) in enumerate([("TEE-M", "tee", 2500), ("TEE-M", "t-shirt", 4000), ("HOODIE-M", "hoodie", 5000)]):
        scheme = SCHEMES[n + 1]
        o1, o2, l = scheme.orders[0], scheme.orders[1], scheme.lines[0]
        b.add("clarify_multi_order", customer=scheme.customer, as_of=as_of, catalog=base_catalog() + hoodies(),
              orders=[order(o1, scheme.customer, ago(as_of, 20), [line(l, sku, 3000)]),
                      order(o2, scheme.customer, ago(as_of, 4), [line(l, sku, paid_b)])],
              turns=[say(next(desc).format(g=garment, cur=size_word("M")), "clarify_needed"),
                     say(next(which).format(o=o2), "refund", order_id=o2, line_id=l),
                     yes(P.CONFIRM[n + 6])],
              expect=["clarify", "propose", "completed"], note="same item in two orders: ask which order; only that line is refunded")

    line_only = cycle(b.rng("clarify_duplicate_line_id"), P.LINE_ONLY)
    which = cycle(b.rng("clarify_duplicate_line_id:t2"), P.WHICH_ORDER_ANSWER)
    for n in range(3):
        scheme = SCHEMES[n]
        o1, o2, l = scheme.orders[0], scheme.orders[1], scheme.lines[0]
        b.add("clarify_duplicate_line_id", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o1, scheme.customer, ago(as_of, 3), [line(l, "TEE-M", 3000)]),
                      order(o2, scheme.customer, ago(as_of, 8), [line(l, "JACKET-M", 9000)])],
              turns=[say(next(line_only).format(l=fmt_id(l, ("plain", "lower", "plain")[n])), "clarify_needed"),
                     say(next(which).format(o=o2), "refund", order_id=o2, line_id=l), yes(P.CONFIRM[n + 1])],
              expect=["clarify", "propose", "completed"], note="line ID exists in two orders")

    unique = [  # lines (sku, paid), target index, garment, size word, template (must carry the disambiguator)
        ([("TEE-M", 3000), ("JACKET-M", 9000)], 1, "jacket", "medium", P.DESC_WITH_ORDER[3]),
        ([("TEE-M", 3000), ("JACKET-M", 9000)], 0, "t-shirt", "medium", P.DESC_WITH_ORDER[1]),
        ([("TEE-S", 4000), ("TEE-M", 4000)], 0, "tee", "small", P.DESC_WITH_ORDER[2]),  # only the size word decides
        ([("HOODIE-M", 5500), ("JEANS-32", 7900)], 1, "jeans", "size 32", P.DESC_WITH_ORDER[0]),
    ]
    for n, (items, pick, garment, word, template) in enumerate(unique):
        scheme = SCHEMES[n % len(SCHEMES)]
        o = scheme.orders[0]
        lines = [line(scheme.lines[i], sku, paid) for i, (sku, paid) in enumerate(items)]
        b.add("description_unique", customer=scheme.customer, as_of=as_of, catalog=base_catalog() + hoodies() + jeans(),
              orders=[order(o, scheme.customer, ago(as_of, 2), lines)],
              turns=[say(template.format(o=fmt_id(o, "plain"), g=garment, cur=word), "refund",
                         order_id=o, line_id=scheme.lines[pick]), yes(P.CONFIRM[n + 2])],
              expect=["propose", "completed"], note="the description matches exactly one line")

    ambiguous = [  # lines, message
        ([("TEE-M", 3000), ("JACKET-M", 9000)], P.AMBIGUOUS_DESC[0].format(cur="medium", o="{o}")),
        ([("TEE-M", 3000), ("TEE-M", 3000)], P.AMBIGUOUS_DESC[2].format(cur="medium")),
        ([("HOODIE-S", 5500), ("HOODIE-L", 5500)], P.AMBIGUOUS_DESC[3].format(g="hoodie")),
    ]
    for n, (items, message) in enumerate(ambiguous):
        scheme = SCHEMES[n]
        o = scheme.orders[0]
        lines = [line(scheme.lines[i], sku, paid) for i, (sku, paid) in enumerate(items)]
        b.add("description_ambiguous", customer=scheme.customer, as_of=as_of, catalog=base_catalog() + hoodies(),
              orders=[order(o, scheme.customer, ago(as_of, 2), lines)],
              turns=[say(message.format(o=o), "clarify_needed")], expect=["clarify"],
              note="the description matches more than one line")
    scheme = SCHEMES[3]
    b.add("description_ambiguous", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(scheme.orders[0], scheme.customer, ago(as_of, 2), [line(scheme.lines[0], "TEE-M", 3000)]),
                  order(scheme.orders[1], scheme.customer, ago(as_of, 9), [line(scheme.lines[0], "TEE-L", 4000)])],
          turns=[say(P.AMBIGUOUS_DESC[1].format(g="tee"), "clarify_needed")], expect=["clarify"],
          note="a tee in each of two orders")

    # Several matches, only one eligible: still clarify (CLARIFICATIONS #6).
    partial = [  # lines with flags, answer, answer target index, expect
        ([("TEE-S", {"final_sale": True}), ("TEE-M", {})], "The medium one.", 1, ["clarify", "propose", "completed"]),
        ([("TEE-S", {"final_sale": True}), ("TEE-M", {})], "the small tee please", 0, ["clarify", "deny"]),
        ([("HOODIE-S", {"unopened": False}), ("HOODIE-L", {})], "I mean the large one.", 1, ["clarify", "propose", "completed"]),
    ]
    first = cycle(b.rng("clarify_partial_eligibility"), P.DESC_WITH_ORDER[:1] + P.AMBIGUOUS_DESC[1:2] + ["Can I return the {g} from {o}? Refund please."])
    for n, (items, answer, pick, expect) in enumerate(partial):
        scheme = SCHEMES[n]
        o = scheme.orders[0]
        garment = "tee" if items[0][0].startswith("TEE") else "hoodie"
        lines = [line(scheme.lines[i], sku, 3500, **flags) for i, (sku, flags) in enumerate(items)]
        turns = [say(next(first).format(g=garment, o=o, cur=""), "clarify_needed"),
                 say(answer, "refund", order_id=o, line_id=scheme.lines[pick])]
        if len(expect) == 3:
            turns.append(yes(P.CONFIRM[n + 4]))
        b.add("clarify_partial_eligibility", customer=scheme.customer, as_of=as_of, catalog=base_catalog() + hoodies(),
              orders=[order(o, scheme.customer, ago(as_of, 2), lines)], turns=turns, expect=expect,
              note="uniqueness is judged over all matching lines, eligible or not")

    no_ids = cycle(b.rng("single_line_no_ids"), P.SINGLE_LINE_NO_IDS)
    for n, (sku, garment) in enumerate([("TEE-M", "t-shirt"), ("HOODIE-S", "hoodie"), ("JEANS-30", "jeans")]):
        scheme = SCHEMES[n + 1]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add("single_line_no_ids", customer=scheme.customer, as_of=as_of, catalog=base_catalog() + hoodies() + jeans(),
              orders=[order(o, scheme.customer, ago(as_of, 6), [line(l, sku, 3300)])],
              turns=[say(next(no_ids).format(g=garment), "refund", order_id=o, line_id=l), yes(P.CONFIRM[n + 7])],
              expect=["propose", "completed"], note="the account has exactly one line and the text names no ID")

    for n, message in enumerate(P.CONTRACTIONS):
        scheme = SCHEMES[n]
        o = scheme.orders[0]
        b.add("contractions_clarify", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, 6), [line(scheme.lines[0], "TEE-M", 3000),
                                                                 line(scheme.lines[1], "TEE-S", 3000)])],
              turns=[say(message, "clarify_needed")], expect=["clarify"],
              note="\"I'm\"/\"it's\" are not sizes M/S; two lines, no unique target")


# --------------------------------------------------------------------------- families: stock faults

def _stock_flow(b: SuiteBuilder, family: str, n: int, *, catalog: list[dict], sku: str, size: str, garment: str,
                faults: list[dict], follow: list[Turn], expect: list[str], note: str, template_bank: list[str] | None = None) -> dict:
    templates = list(template_bank or P.EXCHANGE_EXPLICIT)
    return _exchange_case(b, family, n, catalog=catalog, sku=sku, paid=3000, garment=garment,
                          template=templates[(n * 3 + len(family)) % len(templates)], requested={"requested_size": size},
                          size_text=size_phrase(size, n + 1), confirm=False, faults=faults, extra_turns=follow,
                          expect=expect, note=note)


def stock_faults(b: SuiteBuilder) -> None:
    def change(sku: str, new: int, before: int) -> dict:
        return {"type": "stock_change", "before_turn": before, "sku": sku, "new_stock": new}

    for n, (catalog, sku, size, garment, target) in enumerate([(base_catalog(), "TEE-M", "L", "tee", "TEE-L"),
                                                               (hoodies(), "HOODIE-S", "M", "hoodie", "HOODIE-M"),
                                                               (jeans(), "JEANS-30", "32", "jeans", "JEANS-32")]):
        _stock_flow(b, "stale_stock_to_zero", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[change(target, 0, 2)], follow=[yes(P.CONFIRM[n])], expect=["propose", "deny"],
                    note="replacement sold out between proposal and confirmation: revalidate, deny, no new proposal")
    for n, (catalog, sku, size, target, fault) in enumerate([
            (base_catalog(), "TEE-M", "L", "TEE-L", change("TEE-L", 1, 2)),
            (base_catalog(L=1), "TEE-M", "L", "TEE-L", change("TEE-L", 5, 2)),
            (base_catalog(), "TEE-M", "L", "TEE-L", change("TEE-M", 0, 2))]):
        _stock_flow(b, "stale_stock_positive", n, catalog=catalog, sku=sku, size=size, garment="tee", faults=[fault],
                    follow=[yes(P.CONFIRM[n + 3])], expect=["propose", "completed"],
                    note="stock changed but the replacement is still available; final = new_stock - 1")
    for n, fault in enumerate([change("TEE-M", 7, 2), change("JACKET-L", 4, 2), change("TEE-M", 0, 1)]):
        scheme = SCHEMES[n]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add("stock_change_refund", customer=scheme.customer, as_of=date(2026, 10, 26), catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(date(2026, 10, 26), 11), [line(l, "TEE-M", 3000)])],
              turns=[say(P.REFUND_EXPLICIT[n + 3].format(o=o, l=l), "refund", order_id=o, line_id=l), yes(P.CONFIRM[n + 5])],
              faults=[fault], expect=["propose", "completed"], note="stock is irrelevant to a refund; the event is still recorded")
    for n, (catalog, sku, size, garment, target) in enumerate([(base_catalog(), "TEE-M", "L", "tee", "TEE-L"),
                                                               (sneakers(), "SNEAKER-9", "10", "sneakers", "SNEAKER-10"),
                                                               (hoodies(), "HOODIE-M", "L", "hoodie", "HOODIE-L")]):
        _stock_flow(b, "stock_zero_before_turn1", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[change(target, 0, 1)], follow=[], expect=["deny"],
                    note="the stock change lands before the first turn")
    restock = cycle(b.rng("restock_between_turns"), P.RESTOCK_ASK)
    for n, (catalog, sku, size, garment, target) in enumerate([(base_catalog(), "JACKET-M", "L", "jacket", "JACKET-L"),
                                                               (sneakers(), "SNEAKER-10", "11", "sneakers", "SNEAKER-11"),
                                                               (hoodies(M=0), "HOODIE-S", "M", "hoodie", "HOODIE-M")]):
        scheme = SCHEMES[n % len(SCHEMES)]
        word = size_word(size)
        ask = next(restock).format(word=word, sz=size_phrase(size, n))
        _stock_flow(b, "restock_between_turns", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[change(target, 2, 2)],
                    follow=[say(ask, "exchange", order_id=scheme.orders[0], line_id=scheme.lines[0], requested_size=size),
                            yes(P.CONFIRM[n + 2])],
                    expect=["deny", "propose", "completed"], note="out of stock, restocked, asked again: ends at new_stock - 1")
    alternative = cycle(b.rng("stale_then_alternative"), P.ALT_SIZE_AFTER_DENY)
    for n, (catalog, sku, size, alt, garment, target) in enumerate([
            (base_catalog(), "TEE-M", "L", "S", "tee", "TEE-L"),
            (hoodies(), "HOODIE-M", "L", "S", "hoodie", "HOODIE-L"),
            (jeans(), "JEANS-32", "34", "30", "jeans", "JEANS-34")]):
        scheme = SCHEMES[n % len(SCHEMES)]
        follow = [yes(P.CONFIRM[n]),
                  say(next(alternative).format(sz=size_phrase(alt, n)), "exchange", order_id=scheme.orders[0],
                      line_id=scheme.lines[0], requested_size=alt),
                  yes(P.CONFIRM[n + 6])]
        _stock_flow(b, "stale_then_alternative", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[change(target, 0, 2)], follow=follow, expect=["propose", "deny", "propose", "completed"],
                    note="after the stale deny the customer picks another size; only that exchange commits")
    repeat = cycle(b.rng("stale_then_reconfirm"), P.REPEAT_CONFIRM)
    for n, (catalog, sku, size, garment, target) in enumerate([(base_catalog(), "TEE-M", "L", "tee", "TEE-L"),
                                                               (base_catalog(), "TEE-L", "S", "shirt", "TEE-S"),
                                                               (sneakers(), "SNEAKER-8", "10", "shoes", "SNEAKER-10")]):
        _stock_flow(b, "stale_then_reconfirm", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[change(target, 0, 2)], follow=[yes(P.CONFIRM[n + 1]), yes(next(repeat))],
                    expect=["propose", "deny", "deny"],
                    note="'last' still resolves to the invalidated proposal: deny PROPOSAL_NOT_OPEN")
    for n, (faults, expect) in enumerate([
            ([change("TEE-L", 0, 2), change("TEE-L", 2, 2)], ["propose", "completed"]),
            ([change("TEE-L", 2, 2), change("TEE-L", 0, 2)], ["propose", "deny"]),
            ([change("TEE-L", 1, 2), change("TEE-S", 9, 2)], ["propose", "completed"])]):
        _stock_flow(b, "double_stock_change", n, catalog=base_catalog(), sku="TEE-M", size="L", garment="tee",
                    faults=faults, follow=[yes(P.CONFIRM[n + 8])], expect=expect,
                    note="two faults before the same turn apply in list order")


# --------------------------------------------------------------------------- families: timeout and idempotency

def idempotency(b: SuiteBuilder) -> None:
    timeout = [{"type": "timeout_after_commit"}]
    repeat = cycle(b.rng("timeouts"), P.REPEAT_CONFIRM)
    refund_templates = cycle(b.rng("timeouts:refund"), P.REFUND_EXPLICIT)

    def refund_flow(family: str, n: int, follow: list[Turn], expect: list[str], faults: list[dict], note: str,
                    lines_extra: list[dict] = ()) -> None:
        scheme = SCHEMES[n % len(SCHEMES)]
        as_of = AS_OF_POOL[(n + 2) % len(AS_OF_POOL)]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add(family, customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, 14), [line(l, "TEE-M", 2700)] + list(lines_extra))],
              turns=[say(next(refund_templates).format(o=fmt_id(o, ("plain", "hash", "lower")[n % 3]), l=l), "refund",
                         order_id=o, line_id=l), yes(P.CONFIRM[(n + 4) % len(P.CONFIRM)])] + follow,
              faults=faults, expect=expect, note=note)

    for n in range(3):
        refund_flow("timeout_refund_replay", n, [yes(next(repeat))], ["propose", "completed", "completed"], timeout,
                    "commit persists then times out; reconcile, then replay with zero new effects")
        refund_flow("timeout_two_turns", n, [], ["propose", "completed"], timeout,
                    "only two turns: the reconciled commit must still be reported completed")
        refund_flow("timeout_then_status", n, [say(P.STATUS[n], "status")], ["propose", "completed", "completed"],
                    timeout, "status question after the timeout: report the committed refund")
        refund_flow("duplicate_confirm", n, [yes(next(repeat))], ["propose", "completed", "completed"], [],
                    "confirm twice without any fault: one refund")
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l, l2 = scheme.orders[0], scheme.lines[0], scheme.lines[1]
        refund_flow("duplicate_text_request", n,
                    [say(P.SAME_REQUEST_AGAIN[n].format(o=o, l=l), "refund", order_id=o, line_id=l)],
                    ["propose", "completed", "completed"], [], "same request again without the flag: replay")
        refund_flow("same_request_different_line", n,
                    [say(P.OTHER_LINE_TOO[n].format(o=o, l=l2), "refund", order_id=o, line_id=l2)],
                    ["propose", "completed", "deny"], [], "request_id already completed for another line",
                    lines_extra=[line(l2, "TEE-S", 4000)])
    exchange_bank = cycle(b.rng("timeout_exchange_replay"), P.EXCHANGE_EXPLICIT)
    for n, (catalog, sku, size, garment) in enumerate([(base_catalog(), "TEE-M", "L", "tee"),
                                                       (hoodies(), "HOODIE-M", "S", "hoodie"),
                                                       (jeans(), "JEANS-30", "28", "jeans")]):
        _exchange_case(b, "timeout_exchange_replay", n, catalog=catalog, sku=sku, paid=3000, garment=garment,
                       template=next(exchange_bank), requested={"requested_size": size}, size_text=size_phrase(size, n),
                       faults=timeout, extra_turns=[yes(next(repeat))], expect=["propose", "completed", "completed"],
                       note="replacement stock decremented exactly once despite the timeout and the replay")
    for n, (catalog, sku, size, target, alt, garment) in enumerate([
            (base_catalog(), "TEE-M", "L", "TEE-L", None, "tee"),
            (hoodies(), "HOODIE-S", "M", "HOODIE-M", None, "hoodie"),
            (base_catalog(), "TEE-M", "L", "TEE-L", "S", "t-shirt")]):
        scheme = SCHEMES[n % len(SCHEMES)]
        follow = [yes(P.CONFIRM[n + 2])]
        expect = ["propose", "deny"]
        if alt:  # the fault is still armed and fires on the first commit that actually happens
            follow += [say(P.ALT_SIZE_AFTER_DENY[0].format(sz=size_phrase(alt, 0)), "exchange",
                           order_id=scheme.orders[0], line_id=scheme.lines[0], requested_size=alt), yes(P.CONFIRM[9])]
            expect = ["propose", "deny", "propose", "completed"]
        _stock_flow(b, "timeout_revalidation_fails", n, catalog=catalog, sku=sku, size=size, garment=garment,
                    faults=[{"type": "stock_change", "before_turn": 2, "sku": target, "new_stock": 0}] + timeout,
                    follow=follow, expect=expect, note="revalidation fails before any commit, so the timeout cannot fire there")


# --------------------------------------------------------------------------- families: confirmation semantics

def confirmation(b: SuiteBuilder) -> None:
    as_of = date(2026, 10, 26)

    def base_refund(family: str, n: int, turns_after: list[Turn], expect: list[str], note: str, *,
                    first: Turn | None = None, days: int = 8) -> None:
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        opening = first or say(P.REFUND_EXPLICIT[(n * 4 + len(family)) % len(P.REFUND_EXPLICIT)].format(o=o, l=l),
                               "refund", order_id=o, line_id=l)
        b.add(family, customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, days), [line(l, "TEE-M", 3000), line(scheme.lines[1], "JACKET-M", 9000)])],
              turns=[opening] + turns_after, expect=expect, note=note)

    verbal = cycle(b.rng("verbal_yes_no_flag"), P.VERBAL_YES)
    for n in range(3):
        after = [say(next(verbal), "verbal_yes")] + ([yes(P.CONFIRM[2])] if n == 2 else [])
        base_refund("verbal_yes_no_flag", n, after, ["propose", "propose"] + (["completed"] if n == 2 else []),
                    "a typed 'yes' is not the confirmation flag: re-present the same proposal, no mutation")
    scheme = SCHEMES[0]
    base_refund("confirm_without_proposal", 0, [yes(P.CONFIRM[1])], ["deny", "deny"], "nothing was proposed (expired)", days=40)
    b.add("confirm_without_proposal", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(scheme.orders[0], scheme.customer, ago(as_of, 3), [line(scheme.lines[0], "TEE-M", 3000),
                                                                            line(scheme.lines[1], "JACKET-M", 9000)])],
          turns=[say(P.ORDER_ONLY_VAGUE[2].format(o=scheme.orders[0]), "clarify_needed"), yes(P.CONFIRM[6])],
          expect=["clarify", "clarify"], note="a confirmation flag with no proposal and no target")
    _exchange_case(b, "confirm_without_proposal", 2, catalog=base_catalog(), sku="JACKET-M", paid=9000, garment="jacket",
                   template=P.EXCHANGE_EXPLICIT[3], requested={"requested_size": "L"}, size_text=size_phrase("L", 1),
                   expect=["deny", "deny"], note="flag after an out-of-stock deny")
    in_advance = ["Refund line {l} of {o}. I confirm in advance, just do it.",
                  "Pre-approving this: refund {l} on order {o}.",
                  "Yes, refund {l} from {o}. Consider this message my confirmation."]
    for n, template in enumerate(in_advance):
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        first = Turn(template.format(o=o, l=l), {"kind": "refund", "order_id": o, "line_id": l}, confirm=True)
        after = [yes(P.CONFIRM[3])] if n == 2 else []
        base_refund("confirm_on_first_turn", n, after, ["propose"] + (["completed"] if n == 2 else []),
                    "the flag on turn 1 has no earlier proposal to resolve to", first=first)

    to_exchange = cycle(b.rng("change_refund_to_exchange"), P.CHANGE_TO_EXCHANGE)
    for n, size in enumerate(("S", "L", "S")):
        scheme = SCHEMES[n % len(SCHEMES)]
        change = say(next(to_exchange).format(sz=size_phrase(size, n)), "exchange", order_id=scheme.orders[0],
                     line_id=scheme.lines[0], requested_size=size)
        base_refund("change_refund_to_exchange", n, [change, yes(P.CONFIRM[n + 5])], ["propose", "propose", "completed"],
                    "the second proposal supersedes the first; only the exchange commits")
    to_refund = cycle(b.rng("change_exchange_to_refund"), P.CHANGE_TO_REFUND)
    for n, (catalog, sku, size, garment) in enumerate([(base_catalog(), "TEE-M", "L", "tee"), (hoodies(), "HOODIE-M", "S", "hoodie"),
                                                       (jeans(), "JEANS-32", "34", "jeans")]):
        scheme = SCHEMES[n % len(SCHEMES)]
        _exchange_case(b, "change_exchange_to_refund", n, catalog=catalog, sku=sku, paid=3600, garment=garment,
                       template=P.EXCHANGE_EXPLICIT[n + 4], requested={"requested_size": size}, size_text=size_phrase(size, n),
                       confirm=False, expect=["propose", "propose", "completed"],
                       extra_turns=[say(next(to_refund), "refund", order_id=scheme.orders[0], line_id=scheme.lines[0]),
                                    yes(P.CONFIRM[n + 1])],
                       note="exchange proposal replaced by a refund proposal; no stock moves")
    with_flag = cycle(b.rng("change_with_flag"), P.CHANGE_WITH_FLAG)
    for n, (first, second) in enumerate((("L", "S"), ("S", "L"), ("L", "S"))):
        scheme = SCHEMES[n % len(SCHEMES)]
        flagged_change = Turn(next(with_flag).format(sz=size_phrase(second, n)),
                              {"kind": "exchange", "order_id": scheme.orders[0], "line_id": scheme.lines[0],
                               "requested_size": second}, confirm=True)
        _exchange_case(b, "change_with_flag", n, catalog=base_catalog(), sku="TEE-M", paid=3000, garment="tee",
                       template=P.EXCHANGE_EXPLICIT[(n + 6) % len(P.EXCHANGE_EXPLICIT)], requested={"requested_size": first},
                       size_text=size_phrase(first, n), confirm=False, expect=["propose", "propose", "completed"],
                       extra_turns=[flagged_change, yes(P.CONFIRM[n + 7])],
                       note="a change phrase blocks the flag; the new size is proposed, then confirmed")
    cancel = cycle(b.rng("cancel_with_flag"), P.CANCEL)
    for n in range(3):
        base_refund("cancel_with_flag", n, [Turn(next(cancel), {"kind": "cancel"}, confirm=True)], ["propose", "clarify"],
                    "an explicit cancel blocks the flag: no mutation")


# --------------------------------------------------------------------------- families: adversarial text

def adversarial(b: SuiteBuilder) -> None:
    as_of = date(2026, 10, 26)
    for n, template in enumerate(P.INJECT_ELIGIBLE):
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        turns = [say(template.format(o=o, l=l), "refund", order_id=o, line_id=l)]
        expect = ["propose"]
        if n == 1:
            turns.append(say(P.VERBAL_YES[0], "verbal_yes"))
            expect.append("propose")
        if n == 3:  # a genuine flag after the injection is still honoured
            turns.append(yes(P.CONFIRM[0]))
            expect.append("completed")
        b.add("injection_eligible", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, 3), [line(l, "TEE-M", 3000)])], turns=turns, expect=expect,
              note="injected authority/control tokens never replace the confirmation flag")
    ineligible = [({"final_sale": True}, 5), ({}, 52), ({"unopened": False}, 5)]
    for n, template in enumerate(P.INJECT_INELIGIBLE):
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        if n < 3:
            flags, days = ineligible[n]
            orders = [order(o, scheme.customer, ago(as_of, days), [line(l, "TEE-M", 3000, **flags)])]
            target = o
        else:  # aimed at another customer's order
            orders = [order(o, scheme.customer, ago(as_of, 2), [line(l, "TEE-S", 4000)]),
                      order(scheme.other_order, scheme.other_customer, ago(as_of, 2), [line(l, "JACKET-M", 9000)])]
            target = scheme.other_order
        turns = [say(template.format(o=target, l=l), "refund", order_id=target, line_id=l)]
        if n == 1:
            turns.append(yes(P.CONFIRM[4]))
        b.add("injection_ineligible", customer=scheme.customer, as_of=as_of, catalog=base_catalog(), orders=orders,
              turns=turns, expect=["deny"] * len(turns), note="policy is computed from state, not from claims of authority")
    for n, template in enumerate(P.AMOUNT_INJECTION):
        scheme = SCHEMES[n % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        paid = (2999, 2450, 3125, 1999)[n]
        b.add("amount_injection", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, 9), [line(l, "TEE-M", paid)])],
              turns=[say(template.format(o=o, l=l, full=40, claimed=40), "refund", order_id=o, line_id=l), yes(P.CONFIRM[n])],
              expect=["propose", "completed"], note=f"amount is always paid_cents ({paid}), whatever the customer asks")
    for n, (template, days) in enumerate(zip(P.CLAIM_LATE_BUT_RECENT, (45, 33, 61))):
        scheme = SCHEMES[(n + 1) % len(SCHEMES)]
        o, l = scheme.orders[0], scheme.lines[0]
        b.add("claims_contradict_state", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
              orders=[order(o, scheme.customer, ago(as_of, days), [line(l, "TEE-M", 3000)])],
              turns=[say(template.format(o=o, l=l), "refund", order_id=o, line_id=l)], expect=["deny"],
              note=f"customer says it is recent; delivered_on says {days} days")


# --------------------------------------------------------------------------- families: bigger accounts and conversations

def accounts_and_conversations(b: SuiteBuilder) -> None:
    as_of = date(2026, 10, 26)
    # Several customers in one state, all using the same line IDs.
    people = [("C100", "O100"), ("C200", "O200"), ("C300", "O300")]
    for n, (customer, own_order) in enumerate([("C200", "O200"), ("C300", "O300"), ("C100", "O100")]):
        orders = [order(o, c, ago(as_of, 5 + i), [line("L1", ("TEE-M", "JACKET-M", "TEE-S")[i], (3000, 9000, 4000)[i])])
                  for i, (c, o) in enumerate(people)]
        message = P.REFUND_EXPLICIT[(n + 9) % len(P.REFUND_EXPLICIT)].format(o=own_order, l="L1")
        b.add("multi_customer_state", customer=customer, as_of=as_of, catalog=base_catalog(), orders=orders,
              turns=[say(message, "refund", order_id=own_order, line_id="L1"), yes(P.CONFIRM[n + 2])],
              expect=["propose", "completed"], note="only the authenticated customer's line changes")

    scheme = SCHEMES[0]
    o, o2 = scheme.orders[0], scheme.orders[1]
    two_lines = [line("L1", "TEE-M", 3000), line("L2", "JACKET-M", 9000)]
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(as_of, 6), two_lines)],
          turns=[say("hi, I have a problem with an order", "clarify_needed"),
                 say(f"it's order {o}", "clarify_needed"),
                 say("the jacket. refund it", "refund", order_id=o, line_id="L2"),
                 say(P.VERBAL_YES[3], "verbal_yes"), yes(P.CONFIRM[7]), say(P.STATUS[0], "status")],
          expect=["clarify", "clarify", "propose", "propose", "completed", "completed"], note="six turns")
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(as_of, 6), [line("L1", "TEE-M", 3000)])],
          turns=[say(P.REFUND_EXPLICIT[0].format(o=o, l="L1"), "refund", order_id=o, line_id="L1"),
                 say(P.CHANGE_TO_EXCHANGE[0].format(sz="a large"), "exchange", order_id=o, line_id="L1", requested_size="L"),
                 say("hmm, or maybe small? small please", "exchange", order_id=o, line_id="L1", requested_size="S"),
                 yes(P.CONFIRM[0])],
          expect=["propose", "propose", "propose", "completed"], note="three proposals; only the last commits")
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(as_of, 6), [line("L1", "TEE-M", 3000)])],
          turns=[say(P.EXCHANGE_NO_SIZE[0].format(o=o, l="L1"), "exchange", order_id=o, line_id="L1"),
                 say("medium", "exchange", order_id=o, line_id="L1", requested_size="M"),
                 say("ok, large then", "exchange", order_id=o, line_id="L1", requested_size="L"), yes(P.CONFIRM[1])],
          expect=["clarify", "deny", "propose", "completed"], note="asks for the size it already has, then a valid one")
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(as_of, 40), [line("L1", "TEE-M", 3000)]),
                  order(o2, scheme.customer, ago(as_of, 3), [line("L1", "JACKET-M", 9000)])],
          turns=[say(P.REFUND_EXPLICIT[4].format(o=o, l="L1"), "refund", order_id=o, line_id="L1"),
                 say(f"ok what about the jacket in {o2}? refund that one", "refund", order_id=o2, line_id="L1"),
                 yes(P.CONFIRM[2]), yes(P.REPEAT_CONFIRM[1]), say(P.STATUS[2], "status")],
          expect=["deny", "propose", "completed", "completed", "completed"], note="deny, then another order, then replays")
    scheme = SCHEMES[2]
    o = scheme.orders[0]
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=hoodies(),
          orders=[order(o, scheme.customer, ago(as_of, 12), [line(scheme.lines[0], "HOODIE-M", 4400)])],
          turns=[say(f"i want to exchange my hoodie from {o}, wrong size", "exchange", order_id=o, line_id=scheme.lines[0]),
                 say("XL please", "exchange", order_id=o, line_id=scheme.lines[0], requested_size="XL"),
                 say("fine, a large then", "exchange", order_id=o, line_id=scheme.lines[0], requested_size="L"),
                 say(P.VERBAL_YES[1], "verbal_yes"), yes(P.CONFIRM[4])],
          expect=["clarify", "deny", "propose", "propose", "completed"],
          note="size list, a pricier size refused, a valid size, a typed yes, then the flag")
    scheme = SCHEMES[3]
    o = scheme.orders[0]
    b.add("long_conversation", customer=scheme.customer, as_of=as_of, catalog=base_catalog(),
          orders=[order(o, scheme.customer, ago(as_of, 1), [line(scheme.lines[0], "TEE-M", 3000),
                                                             line(scheme.lines[1], "JACKET-M", 8100)])],
          turns=[say(P.ORDER_ONLY_VAGUE[1].format(o=o), "clarify_needed"),
                 say(f"the jacket, {scheme.lines[1]}", "refund", order_id=o, line_id=scheme.lines[1]),
                 yes(P.CONFIRM[8]), say(P.STATUS[3], "status"), yes(P.REPEAT_CONFIRM[2])],
          faults=[{"type": "timeout_after_commit"}], expect=["clarify", "propose", "completed", "completed", "completed"],
          note="clarify, propose, timeout reconciled, status, replay")

    # A large account: 12 orders x 3 lines, several identical medium tees.
    skus = ["TEE-M", "JACKET-M", "TEE-S", "TEE-M", "TEE-L", "JACKET-L"]
    big = [order(f"O1{i:02d}", "C100", ago(as_of, 2 + i), [line(f"L{k + 1}", skus[(i + k) % len(skus)], 2500 + 100 * k)
                                                              for k in range(3)]) for i in range(12)]
    b.add("large_account", customer="C100", as_of=as_of, catalog=base_catalog(), orders=big,
          turns=[say("Refund line L2 of order O107.", "refund", order_id="O107", line_id="L2"), yes(P.CONFIRM[5])],
          expect=["propose", "completed"], note="explicit target among 36 lines")
    b.add("large_account", customer="C100", as_of=as_of, catalog=base_catalog(), orders=big,
          turns=[say("refund my medium tee", "clarify_needed")], expect=["clarify"], note="many medium tees")
    b.add("large_account", customer="C100", as_of=as_of, catalog=base_catalog(), orders=big,
          turns=[say("Exchange line L3 of order O104 for a large.", "exchange", order_id="O104", line_id="L3", requested_size="L"),
                 yes(P.CONFIRM[9])],
          expect=["propose", "completed"], note="exchange inside a large account")

    id_styles = ["refund line l1 of order #o100", "Order number O100, item L1. Refund please.",
                 "can you refund o-100 / l-1?", "Refund order O100, line 1."]
    for n, message in enumerate(id_styles):
        b.add("id_formatting", customer="C100", as_of=as_of, catalog=base_catalog(),
              orders=[order("O100", "C100", ago(as_of, 4), [line("L1", "TEE-M", 3000)])],
              turns=[say(message, "refund", order_id="O100", line_id="L1"), yes(P.CONFIRM[n + 3])],
              expect=["propose", "completed"], note="IDs are normalised, not matched verbatim")


# --------------------------------------------------------------------------- metamorphic families

METAMORPHIC_SOURCES = ("refund_day30", "refund_expired", "exchange_valid", "exchange_out_of_stock",
                       "exchange_no_size_multi", "clarify_multi_order", "description_unique", "stale_stock_to_zero",
                       "timeout_exchange_replay", "change_refund_to_exchange", "injection_eligible", "foreign_order",
                       "timeout_refund_replay", "verbal_yes_no_flag", "clarify_duplicate_line_id",
                       "exchange_relative_size", "cancel_with_flag", "nonexistent_line")
DATE_SHIFTS = (1, 17, -45, 365, 1096, -400, 59, 3, -1, 730, 29, -200, 4, 366, -30, 90, 1461, -7)


def metamorphic(b: SuiteBuilder) -> None:
    """Re-label transformed copies with the oracle and require identical decisions to the source."""
    sources = [next(i for i, label in enumerate(b.labels) if label["family"] == family) for family in METAMORPHIC_SOURCES]
    for n, index in enumerate(sources):
        case, label = b.cases[index], b.labels[index]
        renamed, intents = rename_ids(case, label["intents"], salt=n + 3, new_case_id=f"v1_metamorphic_rename_{n + 1:02d}")
        b.add_case("metamorphic_rename", renamed, intents, label["expected"]["decisions"], f"IDs renamed from {case['case_id']}")
    for n, index in enumerate(sources):
        case, label = b.cases[index], b.labels[index]
        shifted = shift_dates(case, DATE_SHIFTS[n], f"v1_metamorphic_dateshift_{n + 1:02d}")
        b.add_case("metamorphic_dateshift", shifted, label["intents"], label["expected"]["decisions"],
                   f"all dates shifted {DATE_SHIFTS[n]:+d} days from {case['case_id']}")


FAMILY_BUILDERS: tuple[Callable[[SuiteBuilder], None], ...] = (
    refund_windows, refund_expired, refund_ineligible, not_on_account, exchanges, clarification,
    stock_faults, idempotency, confirmation, adversarial, accounts_and_conversations, metamorphic,
)


def build_suite() -> tuple[dict, dict]:
    builder = SuiteBuilder()
    for build in FAMILY_BUILDERS:
        build(builder)
    cases = {"contract_version": 1, "cases": builder.cases}
    labels = {"labels_version": 1, "generator": "evals/variants/generate.py", "seed": SEED,
              "families": dict(sorted(Counter(label["family"] for label in builder.labels).items())),
              "cases": builder.labels}
    return cases, labels


def dumps(document: dict) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the variant suite (cases + oracle labels).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases, labels = build_suite()
    (out / "variants_v1.json").write_text(dumps(cases))
    (out / "labels_v1.json").write_text(dumps(labels))
    print(f"wrote {len(cases['cases'])} cases in {len(labels['families'])} families to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
