"""30 held-out cases in the terse public style ("Refund line L1 of order O100.").

Same shapes as the 6 public cases, with new IDs, SKUs, numeric sizes, discounted paid_cents,
dates on both sides of the 30-day boundary (month ends, a year end, a leap February), multi-line
and multi-order accounts, other customers' orders in state, and faults on different turns.
Every decision below was decided by hand from CONTRACT.md + CLARIFICATIONS.md defaults.
"""
from __future__ import annotations

from evals.holdout_v3.spec import (
    CONFIRM, STATUS, TIMEOUT, CaseSpec, exchange, exchange_offer, exchanged, flagged, line, order, product,
    refund, refund_offer, refunded, say, sized, state, stock_change, unclear,
)

YES_REFUND = "Yes, confirm the proposed refund."
YES_EXCHANGE = "Yes, confirm the proposed exchange."
REPEAT = "I am repeating my approval because I did not see a clear result."


def tees(s: int = 4, m: int = 2, l: int = 3, price: int = 3200) -> list[dict]:
    return sized("TEE", price, {"S": s, "M": m, "L": l})


def jeans(w30: int = 2, w32: int = 0, w34: int = 3, w36: int | None = None) -> list[dict]:
    """Waist sizes 30-34 share one price; the optional 36 is priced higher (an invalid exchange target)."""
    family = sized("JEANS", 6500, {"30": w30, "32": w32, "34": w34})
    return family + ([product("JEANS-36", "JEANS", "36", 6900, w36)] if w36 is not None else [])


def sneakers(s9: int = 1, s10: int = 2, s11: int = 0) -> list[dict]:
    return sized("SNEAKER", 11000, {"9": s9, "10": s10, "11": s11})


def hoodies(m: int = 3, l: int = 1, xl: int = 0, price: int = 5500) -> list[dict]:
    return sized("HOODIE", price, {"M": m, "L": l, "XL": xl})


def jackets(m: int = 1, l: int = 2) -> list[dict]:
    return sized("JACKET", 9000, {"M": m, "L": l})


# SKUs that do not spell out the size, so only the `size` field can identify the replacement.
OPAQUE_SNEAKERS = [product("SKU-7731", "SNEAKER", "9", 11000, 1), product("SKU-7746", "SNEAKER", "10", 11000, 2),
                   product("SKU-7752", "SNEAKER", "11", 11000, 0)]


CASES: list[CaseSpec] = [
    # ------------------------------------------------------------------ refund window and amount
    CaseSpec(
        "v3_t01", "terse_refund_window",
        "Delivered 2026-10-21, as_of 2026-11-20: exactly 30 days across a 31-day month, eligible. "
        "Refund paid_cents 2599 (discounted), not the 3200 catalog price. Another customer's order holds the same SKU.",
        "C2041",
        state("2026-11-20", tees(5, 3, 2) + jackets(1, 0), [
            order("O5120", "C2041", "2026-10-21", [line("L1", "TEE-L", 2599)]),
            order("O5121", "C9001", "2026-11-01", [line("L1", "TEE-L", 3200)]),
        ]),
        [say("Refund line L1 of my order O5120. It is unopened.", refund("O5120", "L1"),
             "propose", "PROPOSED", refund_offer(1, "O5120", "L1", 2599)),
         flagged("Yes, refund the proposed item and amount.", CONFIRM, "completed", "COMMITTED")],
        effect=refunded("O5120", "L1", 2599),
    ),
    CaseSpec(
        "v3_t02", "terse_refund_window",
        "Delivered 2026-10-31, as_of 2026-12-01: 31 days, expired. T2 carries the flag but no proposal ever "
        "existed, so 'last' resolves to nothing: the request is re-judged and still denied (clarify accepted). "
        "A timeout_after_commit fault is armed but nothing ever commits, so it never fires: no error, no record.",
        "C318",
        state("2026-12-01", jeans(2, 1, 3), [order("O3318", "C318", "2026-10-31", [line("L1", "JEANS-32", 6500)])]),
        [say("Please refund line L1 of order O3318.", refund("O3318", "L1"), "deny", "RETURN_WINDOW_EXPIRED"),
         flagged("Yes, confirm the refund anyway.", CONFIRM, "deny", "RETURN_WINDOW_EXPIRED", alt=("clarify",))],
        faults=[TIMEOUT],
    ),
    CaseSpec(
        "v3_t03", "terse_refund_window",
        "Leap year: delivered 2028-01-31, as_of 2028-03-01 is 30 days (Feb has 29), eligible. Two-line order; "
        "refund L2 only at its discounted paid_cents 4875 (catalog 6500).",
        "C7350",
        state("2028-03-01", tees() + jeans(2, 1, 3), [
            order("O7350", "C7350", "2028-01-31", [line("L1", "TEE-M", 3200), line("L2", "JEANS-30", 4875)]),
            order("O7351", "C9002", "2028-02-20", [line("L1", "JEANS-30", 6500)]),
        ]),
        [say("Refund line L2 of order O7350. It is unopened.", refund("O7350", "L2"),
             "propose", "PROPOSED", refund_offer(1, "O7350", "L2", 4875)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED")],
        effect=refunded("O7350", "L2", 4875),
    ),
    CaseSpec(
        "v3_t04", "terse_refund_window",
        "Delivered on as_of itself: day 0 is inside the inclusive 0-30 window.",
        "C4410",
        state("2026-10-30", sneakers(), [order("O4410", "C4410", "2026-10-30", [line("L1", "SNEAKER-10", 11000)])]),
        [say("Refund line L1 of order O4410. It arrived today and is unopened.", refund("O4410", "L1"),
             "propose", "PROPOSED", refund_offer(1, "O4410", "L1", 11000)),
         flagged("Yes, refund it.", CONFIRM, "completed", "COMMITTED")],
        effect=refunded("O4410", "L1", 11000),
    ),
    # ------------------------------------------------------------------ eligibility refusals
    CaseSpec(
        "v3_t05", "terse_policy_deny",
        "10 days. The message repeats the public 'It is unopened.' claim, but the line record says unopened=false: "
        "the record wins, deny, no state change.",
        "C5502",
        state("2026-11-12", tees(), [order("O5502", "C5502", "2026-11-02", [line("L1", "TEE-S", 3200, opened=True)])]),
        [say("Refund line L1 of order O5502. It is unopened.", refund("O5502", "L1"), "deny", "OPENED")],
    ),
    CaseSpec(
        "v3_t06", "terse_policy_deny",
        "Exchange of a final-sale line (23 days, unopened, DRESS-8 in stock at the same price): eligibility "
        "fails first, so deny. The other line on the order is not final sale and is untouched.",
        "C6603",
        state("2026-11-12", sized("DRESS", 4800, {"4": 1, "6": 2, "8": 2}), [
            order("O6603", "C6603", "2026-10-20",
                  [line("L1", "DRESS-6", 3360, final_sale=True), line("L2", "DRESS-8", 4800)]),
        ]),
        [say("Exchange line L1 of order O6603 for the same dress in size 8.", exchange("O6603", "L1", size="8"),
             "deny", "FINAL_SALE")],
    ),
    CaseSpec(
        "v3_t07", "terse_policy_deny",
        "Line already marked returned=true (records lists start empty, as the contract says): deny.",
        "C7704",
        state("2026-11-05", jackets(), [order("O7704", "C7704", "2026-10-25", [line("L1", "JACKET-M", 9000, returned=True)])]),
        [say("Refund line L1 of order O7704. It is unopened.", refund("O7704", "L1"), "deny", "ALREADY_RETURNED")],
        request_id="R-20261105-0007",
    ),
    CaseSpec(
        "v3_t08", "terse_policy_deny",
        "Order status 'shipped' with delivered_on null: only delivered orders qualify, deny. The customer's "
        "other (delivered) order is irrelevant because the order is named.",
        "C8805",
        state("2026-11-05", hoodies(), [
            order("O8804", "C8805", "2026-10-28", [line("L1", "HOODIE-M", 5500)]),
            order("O8805", "C8805", None, [line("L1", "HOODIE-L", 5500)], status="shipped"),
        ]),
        [say("Refund line L1 of order O8805.", refund("O8805", "L1"), "deny", "NOT_DELIVERED+MISSING_DELIVERY_DATE")],
    ),
    CaseSpec(
        "v3_t09", "terse_ownership",
        "O9101 exists but belongs to C9200; the authenticated customer is C9100: deny exactly like a missing order "
        "(CLARIFICATIONS #3), nothing about the other customer revealed. T2 names the customer's own order but a "
        "line it does not have (O9100 has only L1): no target, so clarify (deny accepted); never guess L1.",
        "C9100",
        state("2026-11-09", tees(), [
            order("O9100", "C9100", "2026-10-30", [line("L1", "TEE-M", 3200)]),
            order("O9101", "C9200", "2026-11-01", [line("L1", "TEE-L", 3200)]),
        ]),
        [say("Refund line L1 of order O9101.", refund("O9101", "L1"), "deny", "ORDER_NOT_FOUND"),
         say("Then refund line L2 of order O9100.", refund("O9100", "L2"), "clarify", "LINE_NOT_FOUND", alt=("deny",))],
    ),
    # ------------------------------------------------------------------ exchanges that succeed
    CaseSpec(
        "v3_t11", "terse_exchange",
        "Year end: delivered 2026-12-05, as_of 2027-01-04 is 30 days. JEANS-30 -> JEANS-34: same product, "
        "same 6500 catalog price (paid 5850 is irrelevant), stock 3 -> 2. Original quarantined, JEANS-30 stock unchanged. "
        "A look-alike product WIDEJEANS also has a 34 at 6500 with more stock: it is a different product_id, untouched.",
        "C1111",
        state("2027-01-04", jeans(2, 0, 3, w36=2) + sized("WIDEJEANS", 6500, {"30": 4, "34": 5}) + tees(), [
            order("O1111", "C1111", "2026-12-05", [line("L1", "JEANS-30", 5850), line("L2", "TEE-M", 3200)]),
            order("O1112", "C9003", "2026-12-20", [line("L1", "JEANS-34", 6500)]),
        ]),
        [say("Exchange line L1 in order O1111 for the same jeans in size 34.", exchange("O1111", "L1", size="34"),
             "propose", "PROPOSED", exchange_offer(1, "O1111", "L1", "JEANS-34")),
         flagged("Yes, confirm this exchange.", CONFIRM, "completed", "COMMITTED")],
        effect=exchanged("O1111", "L1", "JEANS-34"), stock_after={"JEANS-34": 2},
    ),
    CaseSpec(
        "v3_t12", "terse_exchange",
        "Shoe sizes with opaque SKUs (the size is only in the `size` field, never in the SKU text): size 9 "
        "(SKU-7731) -> size 10 (SKU-7746), stock 2 -> 1, 17 days, discounted paid 8800; amount_cents 0. "
        "The customer has a second, older order that is not involved.",
        "C1212",
        state("2026-11-18", OPAQUE_SNEAKERS + tees(), [
            order("O1212", "C1212", "2026-11-01", [line("L1", "SKU-7731", 8800)]),
            order("O1213", "C1212", "2026-10-05", [line("L1", "TEE-M", 3200)]),
        ]),
        [say("Exchange line L1 of order O1212 for the same sneakers in size 10.", exchange("O1212", "L1", size="10"),
             "propose", "PROPOSED", exchange_offer(1, "O1212", "L1", "SKU-7746")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        effect=exchanged("O1212", "L1", "SKU-7746"), stock_after={"SKU-7746": 1},
    ),
    # ------------------------------------------------------------------ impossible exchanges (CLARIFICATIONS #2: deny)
    CaseSpec(
        "v3_t13", "terse_exchange_deny",
        "Requested JEANS-32 has stock 0: deny with no alternative proposal (30 and 34 are mentioned in text only).",
        "C1313",
        state("2026-11-18", jeans(2, 0, 3, w36=2), [order("O1313", "C1313", "2026-11-10", [line("L1", "JEANS-30", 6500)])]),
        [say("Exchange line L1 of order O1313 for the same jeans in size 32.", exchange("O1313", "L1", size="32"),
             "deny", "OUT_OF_STOCK")],
    ),
    CaseSpec(
        "v3_t14", "terse_exchange_deny",
        "JEANS-36 is in stock but its catalog price 6900 differs from JEANS-34's 6500: deny (no price-difference payments).",
        "C1414",
        state("2026-11-18", jeans(2, 0, 3, w36=2), [order("O1414", "C1414", "2026-10-25", [line("L1", "JEANS-34", 6500)])]),
        [say("Exchange line L1 of order O1414 for the same jeans in size 36.", exchange("O1414", "L1", size="36"),
             "deny", "PRICE_MISMATCH")],
    ),
    CaseSpec(
        "v3_t15", "terse_exchange_deny",
        "TEE is offered in S/M/L only; XL does not exist in the catalog: deny.",
        "C1515",
        state("2026-11-18", tees(), [order("O1515", "C1515", "2026-11-15", [line("L1", "TEE-M", 3200)])]),
        [say("Exchange line L1 of order O1515 for the same tee in size XL.", exchange("O1515", "L1", size="XL"),
             "deny", "SIZE_NOT_OFFERED")],
    ),
    CaseSpec(
        "v3_t16", "terse_exchange_deny",
        "Requested size equals the original size (HOODIE-L for HOODIE-L): not a different size, deny.",
        "C1616",
        state("2026-11-22", hoodies(3, 2, 1), [order("O1616", "C1616", "2026-11-01", [line("L1", "HOODIE-L", 5500)])]),
        [say("Exchange line L1 of order O1616 for the same hoodie in size L.", exchange("O1616", "L1", size="L"),
             "deny", "SAME_SIZE")],
    ),
    CaseSpec(
        "v3_t17", "terse_exchange_deny",
        "TEE-M for HOODIE-M: same price 4000 and in stock, but a different product_id: deny.",
        "C1717",
        state("2026-11-22", tees(4, 2, 3, price=4000) + sized("HOODIE", 4000, {"M": 3, "L": 2}), [
            order("O1717", "C1717", "2026-11-12", [line("L1", "TEE-M", 4000)]),
        ]),
        [say("Exchange line L1 of order O1717 for a hoodie in size M.",
             exchange("O1717", "L1", size="M", product_id="HOODIE"), "deny", "DIFFERENT_PRODUCT")],
    ),
    # ------------------------------------------------------------------ exchange with no size named (CLARIFICATIONS #5)
    CaseSpec(
        "v3_t10", "terse_exchange_size_choice",
        "A one-size cap has no other size at all: 'a different size' has zero valid options, so deny.",
        "C1010",
        state("2026-11-09", [product("CAP-OS", "CAP", "OS", 2500, 6)] + tees(), [
            order("O1010", "C1010", "2026-10-28", [line("L1", "CAP-OS", 2500), line("L2", "TEE-M", 3200)]),
        ]),
        [say("Exchange line L1 of order O1010 for a different size.", exchange("O1010", "L1"), "deny", "NO_VALID_SIZE")],
    ),
    CaseSpec(
        "v3_t18", "terse_exchange_size_choice",
        "No size named. JEANS-32 is out of stock and JEANS-36 has a different price, so JEANS-34 is the only valid "
        "size: propose it directly. Stock 2 -> 1.",
        "C1818",
        state("2026-12-10", jeans(1, 0, 2, w36=4), [
            order("O1818", "C1818", "2026-11-20", [line("L1", "JEANS-30", 6500)]),
            order("O1819", "C9004", "2026-11-25", [line("L1", "JEANS-34", 6500)]),
        ]),
        [say("Exchange line L1 of order O1818 for a different size.", exchange("O1818", "L1"),
             "propose", "PROPOSED", exchange_offer(1, "O1818", "L1", "JEANS-34")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        effect=exchanged("O1818", "L1", "JEANS-34"), stock_after={"JEANS-34": 1},
    ),
    CaseSpec(
        "v3_t19", "terse_exchange_size_choice",
        "29 days. No size named and both S and L are valid: clarify listing them. T2 picks S in a follow-up that only "
        "makes sense with the conversation context. TEE-S drops 4 -> 1 before T3 (still positive), so the confirmed "
        "exchange commits against the CURRENT stock: final 0, not the 3 a stale read from T2 would give.",
        "C1919",
        state("2026-12-10", tees(4, 2, 3), [order("O1919", "C1919", "2026-11-11", [line("L1", "TEE-M", 3200)])]),
        [say("Exchange line L1 of order O1919 for a different size.", exchange("O1919", "L1"), "clarify", "SIZE_UNSPECIFIED"),
         say("Size S please, same tee.", exchange("O1919", "L1", size="S"),
             "propose", "PROPOSED", exchange_offer(1, "O1919", "L1", "TEE-S")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        faults=[stock_change(3, "TEE-S", 1)], effect=exchanged("O1919", "L1", "TEE-S"), stock_after={"TEE-S": 0},
    ),
    # ------------------------------------------------------------------ clarification of the target
    CaseSpec(
        "v3_t20", "terse_clarify",
        "Three-line order, no line named: clarify. T2 names L3 (sneakers, paid 9900) and refund. T1/T2 lines untouched.",
        "C2020",
        state("2026-12-10", tees() + jeans(2, 1, 3) + sneakers(), [
            order("O2020", "C2020", "2026-12-01",
                  [line("L1", "TEE-M", 3200), line("L2", "JEANS-32", 6500), line("L3", "SNEAKER-10", 9900)]),
        ]),
        [say("I want to return an item from O2020.", unclear("TARGET_NOT_UNIQUE"), "clarify", "TARGET_NOT_UNIQUE"),
         say("I mean line L3, the size 10 sneakers. Refund it please.", refund("O2020", "L3"),
             "propose", "PROPOSED", refund_offer(1, "O2020", "L3", 9900)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED")],
        effect=refunded("O2020", "L3", 9900),
    ),
    CaseSpec(
        "v3_t21", "terse_clarify",
        "Two own orders both have a line L1, and no order is named: no unique target, clarify. T2 names O2122 "
        "(hoodie, paid 4950). Another customer's order also has HOODIE-M.",
        "C2121",
        state("2026-11-20", tees() + hoodies(), [
            order("O2121", "C2121", "2026-11-01", [line("L1", "TEE-L", 3200)]),
            order("O2122", "C2121", "2026-11-08", [line("L1", "HOODIE-M", 4950)]),
            order("O2123", "C9005", "2026-11-10", [line("L1", "HOODIE-M", 5500)]),
        ]),
        [say("Refund line L1. It is unopened.", unclear("TARGET_NOT_UNIQUE"), "clarify", "TARGET_NOT_UNIQUE"),
         say("It is line L1 of order O2122.", refund("O2122", "L1"),
             "propose", "PROPOSED", refund_offer(1, "O2122", "L1", 4950)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED")],
        effect=refunded("O2122", "L1", 4950),
    ),
    # ------------------------------------------------------------------ stock_change faults
    CaseSpec(
        "v3_t22", "terse_stock_fault",
        "public_05 shape on line L2 of a two-line order: JEANS-34 3 -> 0 before T2, so the confirmed proposal fails "
        "revalidation: deny (CLARIFICATIONS #16). Only the external stock change remains.",
        "C2222",
        state("2026-11-26", tees() + jeans(2, 1, 3), [
            order("O2222", "C2222", "2026-11-10", [line("L1", "TEE-M", 3200), line("L2", "JEANS-32", 6500)]),
        ]),
        [say("Exchange line L2 of O2222 for the same jeans in size 34.", exchange("O2222", "L2", size="34"),
             "propose", "PROPOSED", exchange_offer(1, "O2222", "L2", "JEANS-34")),
         flagged(YES_EXCHANGE, CONFIRM, "deny", "OUT_OF_STOCK")],
        faults=[stock_change(2, "JEANS-34", 0)], stock_after={"JEANS-34": 0}, request_id="REQ-77f1c2",
    ),
    CaseSpec(
        "v3_t23", "terse_stock_fault",
        "29 days. HOODIE-L drops 3 -> 1 before T2: still positive, so the exchange commits and the last unit goes: final 0.",
        "C2323",
        state("2026-11-26", hoodies(3, 3, 0), [order("O2323", "C2323", "2026-10-28", [line("L1", "HOODIE-M", 5500)])]),
        [say("Exchange line L1 of O2323 for the same hoodie in size L.", exchange("O2323", "L1", size="L"),
             "propose", "PROPOSED", exchange_offer(1, "O2323", "L1", "HOODIE-L")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        faults=[stock_change(2, "HOODIE-L", 1)], effect=exchanged("O2323", "L1", "HOODIE-L"), stock_after={"HOODIE-L": 0},
    ),
    CaseSpec(
        "v3_t24", "terse_stock_fault",
        "TEE-L 3 -> 0 BEFORE turn 1, so the first request is already out of stock: deny. T2 asks for S instead: a new "
        "proposal, confirmed in T3. TEE-S 4 -> 3.",
        "C2424",
        state("2026-11-26", tees(4, 2, 3), [order("O2424", "C2424", "2026-11-20", [line("L1", "TEE-M", 3200)])]),
        [say("Exchange line L1 of order O2424 for the same tee in size L.", exchange("O2424", "L1", size="L"),
             "deny", "OUT_OF_STOCK"),
         say("Then exchange it for size S.", exchange("O2424", "L1", size="S"),
             "propose", "PROPOSED", exchange_offer(1, "O2424", "L1", "TEE-S")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        faults=[stock_change(1, "TEE-L", 0)], effect=exchanged("O2424", "L1", "TEE-S"),
        stock_after={"TEE-L": 0, "TEE-S": 3},
    ),
    CaseSpec(
        "v3_t25", "terse_stock_fault",
        "30 days (Nov 3 -> Dec 3). SNEAKER-11 is 0 in the initial state but restocked to 2 before turn 1, so the "
        "exchange is proposed and commits: final 1.",
        "C2525",
        state("2026-12-03", sneakers(1, 2, 0), [order("O2525", "C2525", "2026-11-03", [line("L1", "SNEAKER-10", 11000)])]),
        [say("Exchange line L1 of order O2525 for the same sneakers in size 11.", exchange("O2525", "L1", size="11"),
             "propose", "PROPOSED", exchange_offer(1, "O2525", "L1", "SNEAKER-11")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED")],
        faults=[stock_change(1, "SNEAKER-11", 2)], effect=exchanged("O2525", "L1", "SNEAKER-11"),
        stock_after={"SNEAKER-11": 1},
    ),
    CaseSpec(
        "v3_t26", "terse_stock_fault",
        "Refund while the refunded SKU's own stock is set 4 -> 0 before T2: refunds do not depend on stock and the "
        "returned unit is quarantined, so TEE-S stays 0 (not 1). Refund paid 2800.",
        "C2626",
        state("2026-12-03", tees(4, 2, 3) + jackets(1, 0), [order("O2626", "C2626", "2026-11-25", [line("L1", "TEE-S", 2800)])]),
        [say("Refund line L1 of order O2626. It is unopened.", refund("O2626", "L1"),
             "propose", "PROPOSED", refund_offer(1, "O2626", "L1", 2800)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED")],
        faults=[stock_change(2, "TEE-S", 0)], effect=refunded("O2626", "L1", 2800), stock_after={"TEE-S": 0},
    ),
    # ------------------------------------------------------------------ timeout_after_commit and replay
    CaseSpec(
        "v3_t27", "terse_timeout",
        "public_06 shape with an EXCHANGE: the T2 commit persists then times out; reconciliation finds it (completed; "
        "error accepted, CLARIFICATIONS #1). T3 replays with zero new effects: SNEAKER-9 decremented once, 1 -> 0.",
        "C2727",
        state("2026-11-14", sneakers(1, 2, 0), [order("O2727", "C2727", "2026-10-20", [line("L1", "SNEAKER-10", 11000)])]),
        [say("Exchange line L1 of order O2727 for the same sneakers in size 9.", exchange("O2727", "L1", size="9"),
             "propose", "PROPOSED", exchange_offer(1, "O2727", "L1", "SNEAKER-9")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED_TIMEOUT_RECONCILED", alt=("error",)),
         flagged(REPEAT, CONFIRM, "completed", "REPLAY")],
        faults=[TIMEOUT], effect=exchanged("O2727", "L1", "SNEAKER-9"), stock_after={"SNEAKER-9": 0},
    ),
    CaseSpec(
        "v3_t28", "terse_timeout",
        "The commit (and so the timeout) happens on T3, after a clarification. T4 is an unflagged status question: "
        "completed replay of the same refund (5525), never a second refund.",
        "C2828",
        state("2026-11-14", tees() + jeans(2, 1, 3), [
            order("O2828", "C2828", "2026-11-01", [line("L1", "TEE-M", 3200), line("L2", "JEANS-32", 5525)]),
            order("O2829", "C9006", "2026-11-02", [line("L1", "JEANS-32", 6500)]),
        ]),
        [say("I want to return an item from O2828.", unclear("TARGET_NOT_UNIQUE"), "clarify", "TARGET_NOT_UNIQUE"),
         say("I mean line L2, the size 32 jeans. Refund it please.", refund("O2828", "L2"),
             "propose", "PROPOSED", refund_offer(1, "O2828", "L2", 5525)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED_TIMEOUT_RECONCILED", alt=("error",)),
         say("I did not see a result. Did the refund go through?", STATUS, "completed", "REPLAY")],
        faults=[TIMEOUT], effect=refunded("O2828", "L2", 5525),
    ),
    CaseSpec(
        "v3_t29", "terse_timeout",
        "Both faults at once, at the 30-day boundary (Oct 15 -> Nov 14): TEE-S 4 -> 1 before T2 (still positive), "
        "then the T2 commit times out after persisting. T3 replays. TEE-S ends 0: one decrement only.",
        "C2929",
        state("2026-11-14", tees(4, 2, 3), [order("O2929", "C2929", "2026-10-15", [line("L1", "TEE-M", 3200)])]),
        [say("Exchange line L1 of O2929 for the same tee in size S.", exchange("O2929", "L1", size="S"),
             "propose", "PROPOSED", exchange_offer(1, "O2929", "L1", "TEE-S")),
         flagged(YES_EXCHANGE, CONFIRM, "completed", "COMMITTED_TIMEOUT_RECONCILED", alt=("error",)),
         flagged(REPEAT, CONFIRM, "completed", "REPLAY")],
        faults=[stock_change(2, "TEE-S", 1), TIMEOUT], effect=exchanged("O2929", "L1", "TEE-S"),
        stock_after={"TEE-S": 0}, request_id="req-2929-a",
    ),
    CaseSpec(
        "v3_t30", "terse_replay",
        "After completion the identical request arrives again (no flag) under the same request_id: completed replay "
        "of the original 7650 refund with zero new effects (CLARIFICATIONS #7).",
        "C3030",
        state("2026-11-30", jackets(1, 2), [
            order("O3030", "C3030", "2026-11-02", [line("L1", "JACKET-M", 7650)]),
            order("O3031", "C9007", "2026-11-03", [line("L1", "JACKET-M", 9000)]),
        ]),
        [say("Refund line L1 of order O3030.", refund("O3030", "L1"),
             "propose", "PROPOSED", refund_offer(1, "O3030", "L1", 7650)),
         flagged(YES_REFUND, CONFIRM, "completed", "COMMITTED"),
         say("Refund line L1 of order O3030.", refund("O3030", "L1"), "completed", "REPLAY")],
        effect=refunded("O3030", "L1", 7650),
    ),
]
