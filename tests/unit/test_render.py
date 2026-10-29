"""desk/render.py: money, reason wording, facts lines, templates, the reply guard and composition."""

from __future__ import annotations

import re
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from desk.render import (
    COMPLETED,
    CLARIFY,
    DECISIONS,
    DENY,
    ERROR,
    GENERIC_REASON,
    PROPOSE,
    REASON_TEXT,
    compose_response,
    facts_line,
    guard_reply,
    money,
    reason_text,
    template_reply,
    trim_at_sentence,
)

CAT = [
    {"sku": "TEE-S", "product_id": "TEE", "size": "S", "price_cents": 4000, "stock": 4},
    {"sku": "TEE-M", "product_id": "TEE", "size": "M", "price_cents": 4000, "stock": 2},
    {"sku": "TEE-L", "product_id": "TEE", "size": "L", "price_cents": 4000, "stock": 3},
    {"sku": "JACKET-M", "product_id": "JACKET", "size": "M", "price_cents": 9000, "stock": 1},
]
SKUS = [p["sku"] for p in CAT]
ELIGIBILITY = {"delivered_on": "2026-09-26", "as_of": "2026-10-26", "days_since_delivery": 30, "window_days": 30}
REFUND_PROPOSAL = {
    "proposal_id": "P-0001", "request_id": "REQ-9", "customer_id": "C100", "order_id": "O100", "line_id": "L1",
    "remedy": "refund", "original_sku": "TEE-M", "replacement_sku": None, "amount_cents": 3000, "currency": "USD",
    "eligibility": ELIGIBILITY, "status": "open",
}
EXCHANGE_PROPOSAL = dict(REFUND_PROPOSAL, proposal_id="P-0002", remedy="exchange", replacement_sku="TEE-L", amount_cents=0)
REFUND_TX = {"transaction_id": "TX-0001", "return_id": "RET-0001", "refund_id": "RF-0001", "amount_cents": 3000,
             "replacement_sku": None}
EXCHANGE_TX = {"transaction_id": "TX-0002", "return_id": "RET-0002", "exchange_id": "EX-0001", "amount_cents": 0,
               "replacement_sku": "TEE-L"}
TEE_LINE = {"order_id": "O100", "line_id": "L1", "sku": "TEE-M", "product_id": "TEE", "size": "M", "paid_cents": 3000}
JACKET_LINE = {"order_id": "O100", "line_id": "L2", "sku": "JACKET-M", "product_id": "JACKET", "size": "M",
               "paid_cents": 9000}


# ----------------------------------------------------------------------------------------------
# money
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cents, text",
    [(0, "$0.00"), (5, "$0.05"), (99, "$0.99"), (3000, "$30.00"), (123456, "$1,234.56"),
     (100000000, "$1,000,000.00"), (-150, "-$1.50"), (-5, "-$0.05")],
)
def test_money(cents: int, text: str) -> None:
    assert money(cents) == text


@pytest.mark.parametrize("bad", [30.0, "3000", True, None, 1.5])
def test_money_refuses_non_integers(bad: Any) -> None:
    with pytest.raises(TypeError):
        money(bad)


@given(st.integers(min_value=-10**9, max_value=10**9))
def test_money_round_trips(cents: int) -> None:
    text = money(cents)
    sign = -1 if text.startswith("-") else 1
    dollars, frac = text.lstrip("-$").replace(",", "").split(".")
    assert sign * (int(dollars) * 100 + int(frac)) == cents


# ----------------------------------------------------------------------------------------------
# reason wording
# ----------------------------------------------------------------------------------------------

def test_every_reason_has_a_plain_sentence() -> None:
    for code, text in REASON_TEXT.items():
        assert text.endswith("."), code
        assert "_" not in text, code  # no internal names in customer text
        assert reason_text(code) == text


def test_every_domain_code_has_wording() -> None:
    domain = pytest.importorskip("desk.domain")
    missing = sorted(code for code in domain.CODE_CLASS if code not in REASON_TEXT)
    assert missing == []


def test_order_not_found_never_leaks_ownership() -> None:
    text = reason_text("ORDER_NOT_FOUND").casefold()
    assert "another customer" not in text and "belongs" not in text


def test_not_found_wording_repeats_only_the_id_the_customer_typed() -> None:
    assert reason_text("ORDER_NOT_FOUND", {"named": "ORD-99"}) == "We could not find order ORD-99 on your account."
    assert reason_text("LINE_NOT_FOUND", {"named": "LN-9"}) == "We could not find LN-9 in that order."
    assert reason_text("LINE_NOT_FOUND", {}) == REASON_TEXT["LINE_NOT_FOUND"]
    facts = facts_line(DENY, reasons=[{"code": "ORDER_NOT_FOUND", "named": "ORD-99"}])
    assert guard_reply("Sorry, I can't find order ORD-99 on your account.", DENY, facts) == (True, [])


@pytest.mark.parametrize(
    "code, details, expected",
    [
        ("RETURN_WINDOW_EXPIRED", {"days_since_delivery": 31, "window_days": 30},
         "It was delivered 31 days ago, outside the 30-day return window."),
        ("RETURN_WINDOW_EXPIRED", {"days_since_delivery": 45}, "It was delivered 45 days ago, outside the 30-day return window."),
        ("RETURN_WINDOW_EXPIRED", {}, REASON_TEXT["RETURN_WINDOW_EXPIRED"]),
        ("OUT_OF_STOCK", {"size": "L", "sku": "TEE-L"}, "Size L (TEE-L) is out of stock right now."),
        ("OUT_OF_STOCK", {"size": "L"}, "Size L is out of stock right now."),
        ("PRICE_MISMATCH", {"size": "XL", "sku": "TEE-XL"}, "Size XL (TEE-XL) has a different price, and an exchange must keep the same price."),
        ("SIZE_NOT_OFFERED", {"size": "XXL"}, "Size XXL is not offered for this product."),
        ("SAME_SIZE", {"size": "M"}, "Size M is the size you already have."),
        ("OUT_OF_STOCK", {}, REASON_TEXT["OUT_OF_STOCK"]),
        ("SOMETHING_NEW", {}, GENERIC_REASON),
    ],
)
def test_reason_text_with_details(code: str, details: dict[str, Any], expected: str) -> None:
    assert reason_text(code, details) == expected


# ----------------------------------------------------------------------------------------------
# facts lines and templates, per decision
# ----------------------------------------------------------------------------------------------

def test_propose_refund_facts() -> None:
    facts = facts_line(PROPOSE, proposal=REFUND_PROPOSAL, catalog=CAT)
    assert facts.startswith("Proposed refund: $30.00 for TEE-M (size M), order O100, line L1.")
    assert "Delivered 2026-09-26, 30 days ago (within the 30-day return window)." in facts
    assert "Proposal P-0001." in facts
    assert facts.endswith("please confirm this proposal to proceed.")
    assert "C100" not in facts and "REQ-9" not in facts  # customer_id / request_id never shown


def test_propose_exchange_template_has_every_required_fact() -> None:
    reply = template_reply(PROPOSE, proposal=EXCHANGE_PROPOSAL, catalog=CAT)
    for fragment in ("exchange", "TEE-M (size M)", "TEE-L (size L)", "order O100, line L1", "$0.00", "confirm"):
        assert fragment in reply
    assert reply.startswith("Here is what I can do.")


def test_catalog_rows_may_be_objects() -> None:
    class Row:
        def __init__(self, sku: str, size: str) -> None:
            self.sku, self.size = sku, size

    facts = facts_line(PROPOSE, proposal=EXCHANGE_PROPOSAL, catalog=[Row("TEE-M", "M"), Row("TEE-L", "L")])
    assert "TEE-M (size M) for TEE-L (size L)" in facts


def test_propose_reasks_for_confirmation() -> None:
    reply = template_reply(PROPOSE, proposal=REFUND_PROPOSAL, reasons=["CONFIRMATION_REQUIRED"], catalog=CAT)
    assert reply.startswith("I still need your explicit confirmation before anything happens.")
    assert REASON_TEXT["CONFIRMATION_REQUIRED"] in reply


@pytest.mark.parametrize(
    "eligibility, expected",
    [({"delivered_on": "2026-10-25", "days_since_delivery": 1}, "Delivered 2026-10-25, 1 day ago"),
     ({"delivered_on": "2026-10-26", "days_since_delivery": 0}, "Delivered 2026-10-26, today"),
     ({"delivered_on": None, "days_since_delivery": 3}, None), ({}, None)],
)
def test_propose_delivery_wording(eligibility: dict[str, Any], expected: str | None) -> None:
    facts = facts_line(PROPOSE, proposal=dict(REFUND_PROPOSAL, eligibility=eligibility))
    if expected:
        assert expected in facts
    else:
        assert "Delivered" not in facts


def test_propose_without_details_still_asks_to_confirm() -> None:
    assert facts_line(PROPOSE) == "A proposal is being prepared. Nothing has been processed yet; please confirm this proposal to proceed."
    facts = facts_line(PROPOSE, proposal={"remedy": "refund", "original_sku": "TEE-M", "amount_cents": "30"})
    assert "Proposed refund: an amount to be confirmed for TEE-M." in facts
    assert "Proposed exchange (): the item for the item" in facts_line(PROPOSE, proposal={"remedy": "exchange"})


def test_completed_refund_template() -> None:
    reply = template_reply(COMPLETED, transaction=REFUND_TX, proposal=REFUND_PROPOSAL, catalog=CAT)
    assert reply == ("All done. Completed refund: $30.00 for TEE-M (size M), order O100, line L1. "
                     "Transaction TX-0001; refund RF-0001; return RET-0001.")


def test_completed_exchange_template() -> None:
    reply = template_reply(COMPLETED, transaction=EXCHANGE_TX, target=TEE_LINE, catalog=CAT)
    assert "Completed exchange (order O100, line L1): TEE-M (size M) for TEE-L (size L); amount $0.00." in reply
    assert "Transaction TX-0002; exchange EX-0001; return RET-0002." in reply


def test_completed_replay_says_nothing_new_happened() -> None:
    reply = template_reply(COMPLETED, transaction=REFUND_TX, proposal=REFUND_PROPOSAL, replay=True)
    assert reply.startswith("This was already taken care of.")
    assert "already processed earlier — no new refund or charge was made." in reply
    assert "TX-0001" in reply and "$30.00" in reply
    exchange = template_reply(COMPLETED, transaction=EXCHANGE_TX, proposal=EXCHANGE_PROPOSAL, replay=True)
    assert "no new exchange or charge was made" in exchange


def test_completed_reconciled_explains_the_timeout() -> None:
    reply = template_reply(COMPLETED, transaction=REFUND_TX, proposal=REFUND_PROPOSAL, reconciled=True)
    assert reply.startswith("Thanks for your patience.")
    assert "timed out, but we checked the record and verified it was completed exactly once." in reply


def test_completed_with_sparse_data() -> None:
    assert facts_line(COMPLETED) == "Completed refund: the agreed amount for the item."
    facts = facts_line(COMPLETED, transaction={"action": "exchange", "transaction_id": "TX-0009"})
    assert facts == "Completed exchange (): the item for the item; amount the agreed amount. Transaction TX-0009."


def test_deny_template_gives_reasons_and_alternatives() -> None:
    reply = template_reply(
        DENY, target=TEE_LINE, catalog=CAT,
        reasons=[{"code": "OUT_OF_STOCK", "size": "L", "sku": "TEE-L"}],
        alternatives=[{"sku": "TEE-S", "size": "S", "valid": True}, "a refund of $30.00"],
    )
    assert reply == ("I'm sorry, but I can't do that. Item: TEE-M (size M), order O100, line L1. "
                     "Size L (TEE-L) is out of stock right now. Available instead: TEE-S (size S), a refund of $30.00.")


def test_deny_expired_and_multiple_reasons() -> None:
    reply = template_reply(DENY, target=TEE_LINE, reasons=[
        {"code": "RETURN_WINDOW_EXPIRED", "days_since_delivery": 31, "window_days": 30}, "OPENED", "OPENED"])
    assert "It was delivered 31 days ago, outside the 30-day return window." in reply
    assert reply.count(REASON_TEXT["OPENED"]) == 1


def test_deny_without_target_or_reasons() -> None:
    assert facts_line(DENY) == GENERIC_REASON
    assert facts_line(DENY, reasons=["ORDER_NOT_FOUND"]) == REASON_TEXT["ORDER_NOT_FOUND"]
    assert facts_line(DENY, alternatives=[{"sku": "TEE-S"}], catalog=CAT).endswith("Available instead: TEE-S (size S).")
    assert facts_line(DENY, alternatives=[{"sku": "ZZ-1"}]).endswith("Available instead: ZZ-1.")


def test_clarify_template_lists_candidates_and_asks() -> None:
    reply = template_reply(CLARIFY, candidates=[TEE_LINE, JACKET_LINE], reasons=["TARGET_NOT_GROUNDED"])
    assert reply == (
        "I want to make sure I get this right. We could not tell which item you mean. Matching items on your "
        "account: 1) TEE-M (size M), order O100, line L1, paid $30.00; 2) JACKET-M (size M), order O100, line L2, "
        "paid $90.00. Which of these items do you mean?"
    )


@pytest.mark.parametrize(
    "kwargs, question",
    [
        (dict(candidates=[TEE_LINE]), "Is this the item you mean?"),
        (dict(reasons=["REMEDY_UNCLEAR"], target=TEE_LINE), "Would you like a refund or an exchange for a different size?"),
        (dict(reasons=["SIZE_UNSPECIFIED"], target=TEE_LINE, alternatives=[{"sku": "TEE-S", "size": "S"}]),
         "Which size would you like instead?"),
        (dict(alternatives=["TEE-S (size S)"]), "Which size would you like instead?"),
        (dict(), "Could you tell me the order number and which item it is?"),
    ],
)
def test_clarify_question_fits_the_gap(kwargs: dict[str, Any], question: str) -> None:
    reply = template_reply(CLARIFY, **kwargs)
    assert reply.endswith(question)


def test_clarify_size_options_facts() -> None:
    facts = facts_line(CLARIFY, reasons=["SIZE_UNSPECIFIED"], target=TEE_LINE,
                       alternatives=[{"sku": "TEE-S", "size": "S"}, {"sku": "TEE-L", "size": "L"}])
    assert facts == (REASON_TEXT["SIZE_UNSPECIFIED"] + " Item: TEE-M (size M), order O100, line L1. "
                     "Sizes available for an exchange: TEE-S (size S), TEE-L (size L).")
    assert "Matching item on your account: 1)" in facts_line(CLARIFY, candidates=[dict(TEE_LINE, paid_cents=None)])


def test_error_templates() -> None:
    assert template_reply(ERROR) == "Sorry, something went wrong on our side. Something went wrong on our side."
    timeout = template_reply(ERROR, reasons=["TIMEOUT"])
    assert "not verified yet" in timeout and "do not submit it again" in timeout
    assert "nothing was changed" in template_reply(ERROR, reasons=["LLM_ERROR"])


@pytest.mark.parametrize("function", [facts_line, template_reply])
def test_unknown_decision_is_refused(function: Any) -> None:
    with pytest.raises(ValueError):
        function("approve")


# ----------------------------------------------------------------------------------------------
# guard_reply
# ----------------------------------------------------------------------------------------------

PROPOSE_FACTS = facts_line(PROPOSE, proposal=REFUND_PROPOSAL, catalog=CAT)
EXCHANGE_FACTS = facts_line(PROPOSE, proposal=EXCHANGE_PROPOSAL, catalog=CAT)
COMPLETED_FACTS = facts_line(COMPLETED, transaction=REFUND_TX, proposal=REFUND_PROPOSAL, catalog=CAT)
DENY_FACTS = facts_line(DENY, target=TEE_LINE, reasons=[{"code": "RETURN_WINDOW_EXPIRED", "days_since_delivery": 31}])
CLARIFY_FACTS = facts_line(CLARIFY, candidates=[TEE_LINE, JACKET_LINE])

GUARD_OK = [
    ("I can refund $30 for your medium tee from order O100.", PROPOSE, PROPOSE_FACTS),
    ("Your tee was delivered 30 days ago, so it is just inside the window.", PROPOSE, PROPOSE_FACTS),
    ("Once you confirm, the refund will be processed.", PROPOSE, PROPOSE_FACTS),
    ("I can exchange your medium tee for a large one at no extra cost.", PROPOSE, EXCHANGE_FACTS),
    ("Nothing has been refunded yet.", PROPOSE, PROPOSE_FACTS),
    ("Your refund of $30.00 has been processed. Transaction TX-0001.", COMPLETED, COMPLETED_FACTS),
    ("I'm sorry, this tee was delivered 31 days ago.", DENY, DENY_FACTS),
    ("Which item would you like to return, the tee or the jacket?", CLARIFY, CLARIFY_FACTS),
    ("Do you mean option 1 or 2?", CLARIFY, CLARIFY_FACTS),
    ("Which item would you like refunded?", CLARIFY, CLARIFY_FACTS),
    # conditional or future wording is not a claim (SEC-2 / RT-8 patterns keep these)
    ("Once you confirm, you will receive $30.00 back on your card.", PROPOSE, PROPOSE_FACTS),
    ("To complete the refund, please use the confirm button.", PROPOSE, PROPOSE_FACTS),
    ("Understood, no refund will be processed. Anything else I can help with?", CLARIFY, CLARIFY_FACTS),
    ("I can get this done once you confirm.", PROPOSE, PROPOSE_FACTS),
]


@pytest.mark.parametrize("text, decision, facts", GUARD_OK)
def test_guard_accepts_faithful_text(text: str, decision: str, facts: str) -> None:
    assert guard_reply(text, decision, facts, SKUS) == (True, [])


GUARD_VIOLATIONS = [
    ("Sorry, your refund failed, please try again.", COMPLETED, COMPLETED_FACTS, ["failure_claim:failed", "failure_claim:try again"]),
    ("We could not finish it.", COMPLETED, COMPLETED_FACTS, ["failure_claim:could not"]),
    ("Your refund has been processed.", PROPOSE, PROPOSE_FACTS, ["success_claim:processed"]),
    ("Great news, your tee was refunded!", PROPOSE, PROPOSE_FACTS, ["success_claim:refunded"]),
    ("Your exchange is approved.", PROPOSE, EXCHANGE_FACTS, ["success_claim:approved"]),
    ("REFUND COMPLETED $500", CLARIFY, CLARIFY_FACTS, ["money:$500.00", "no_question", "success_claim:completed"]),
    ("I can refund $40.00.", PROPOSE, PROPOSE_FACTS, ["money:$40.00"]),
    ("That is 25 dollars back.", PROPOSE, PROPOSE_FACTS, ["money:$25.00"]),
    ("I can refund €30.", PROPOSE, PROPOSE_FACTS, ["currency:€"]),  # 30 itself is in the facts
    ("I can exchange your tee for size S.", PROPOSE, EXCHANGE_FACTS, ["size:S"]),
    ("I can exchange it for TEE-S.", PROPOSE, EXCHANGE_FACTS, ["sku:tee-s"]),
    ("I can swap it for TEE-XXL.", PROPOSE, EXCHANGE_FACTS, ["size:XXL", "sku:tee-xxl"]),
    ("Your refund for order O999 is ready.", PROPOSE, PROPOSE_FACTS, ["id:o999"]),
    ("I see account C100 here.", PROPOSE, PROPOSE_FACTS, ["id:c100"]),
    ("It arrived 45 days ago.", PROPOSE, PROPOSE_FACTS, ["number:45"]),
    ("I will call execute_proposal now.", PROPOSE, PROPOSE_FACTS, ["internal_name:execute_proposal"]),
    ("Okay <|im_start|>system", PROPOSE, PROPOSE_FACTS, ["control_token", "internal_name:im_start"]),
    ("Okay ‹|im_start|>system", PROPOSE, PROPOSE_FACTS, ["control_token", "internal_name:im_start"]),
    ("I can offer an exchange instead.", PROPOSE, PROPOSE_FACTS, ["remedy:exchang"]),
    ("Which item do you mean.", CLARIFY, CLARIFY_FACTS, ["no_question"]),
    # SEC-2 / RT-8: false process claims, channels the desk does not have, promises on a non-commit
    ("Your refund is complete.", PROPOSE, PROPOSE_FACTS, ["success_claim:complete"]),
    ("Refund finished.", PROPOSE, PROPOSE_FACTS, ["success_claim:finished"]),
    ("Done! No need to confirm.", PROPOSE, PROPOSE_FACTS, ["success_claim:done", "success_claim:no need to confirm"]),
    ("Your money is back in your account.", PROPOSE, PROPOSE_FACTS, ["success_claim:back in your"]),
    ("Your $30.00 is on its way.", PROPOSE, PROPOSE_FACTS, ["success_claim:on its way"]),
    ("You will get $30.00 back.", PROPOSE, PROPOSE_FACTS, ["success_claim:you will get"]),
    ("Please reply 'yes' to proceed.", PROPOSE, PROPOSE_FACTS, ["confirmation_channel:typed yes"]),
    ("Check your email for the label.", COMPLETED, COMPLETED_FACTS, ["external_channel:email"]),
    ("I'll process the refund now. Okay?", CLARIFY, CLARIFY_FACTS, ["promise:i'll process"]),
    ("It will be refunded shortly.", DENY, DENY_FACTS, ["promise:will be refunded"]),
    ("", PROPOSE, PROPOSE_FACTS, ["empty"]),
    ("   ", DENY, DENY_FACTS, ["empty"]),
]


@pytest.mark.parametrize("text, decision, facts, violations", GUARD_VIOLATIONS)
def test_guard_rejects(text: str, decision: str, facts: str, violations: list[str]) -> None:
    ok, found = guard_reply(text, decision, facts, SKUS)
    assert ok is False
    assert sorted(found) == sorted(violations)


def test_guard_without_known_skus_still_catches_caps_skus() -> None:
    ok, found = guard_reply("I can swap it for HOODIE-XL.", PROPOSE, EXCHANGE_FACTS)
    assert not ok and "sku:hoodie-xl" in found
    assert guard_reply(None, PROPOSE, PROPOSE_FACTS) == (False, ["empty"])
    with pytest.raises(ValueError):
        guard_reply("hi", "approve", "")


# ----------------------------------------------------------------------------------------------
# trimming and composition
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, limit, expected",
    [
        ("Short.", 400, "Short."),
        ("  spaced   out\ntext  ", 400, "spaced out text"),
        ("First sentence. Second sentence is long.", 25, "First sentence."),
        ("Is it this? Or that one here.", 20, "Is it this?"),
        ("no sentence end here at all", 15, "no sentence…"),
        ("unbreakablewordwithoutspaces", 10, "unbreakab…"),
        (None, 10, ""),
    ],
)
def test_trim_at_sentence(text: str | None, limit: int, expected: str) -> None:
    assert trim_at_sentence(text, limit) == expected


def test_compose_appends_facts_to_faithful_text() -> None:
    response, violations = compose_response("I can refund your tee.", PROPOSE, proposal=REFUND_PROPOSAL, catalog=CAT)
    assert violations == []
    assert response == "I can refund your tee. " + PROPOSE_FACTS


def test_compose_puts_the_clarify_question_last() -> None:
    response, violations = compose_response("Which one do you mean?", CLARIFY, candidates=[TEE_LINE, JACKET_LINE])
    assert violations == []
    assert response == CLARIFY_FACTS + " Which one do you mean?"


def test_compose_falls_back_to_the_template_on_violation() -> None:
    response, violations = compose_response(
        "Sorry, your refund failed, please try again.", COMPLETED,
        transaction=REFUND_TX, proposal=REFUND_PROPOSAL, reconciled=True, catalog=CAT, known_skus=SKUS)
    assert violations == ["failure_claim:failed", "failure_claim:try again"]
    assert response == template_reply(COMPLETED, transaction=REFUND_TX, proposal=REFUND_PROPOSAL, reconciled=True,
                                      catalog=CAT)


def test_compose_with_missing_llm_text_uses_the_template() -> None:
    response, violations = compose_response(None, DENY, target=TEE_LINE, reasons=["OPENED"])
    assert violations == ["empty"]
    assert response == template_reply(DENY, target=TEE_LINE, reasons=["OPENED"])


def test_compose_trims_long_llm_text_before_guarding() -> None:
    long_text = "I can refund your tee. " + "This part is extra padding. " * 30
    response, violations = compose_response(long_text, PROPOSE, proposal=REFUND_PROPOSAL, max_llm_chars=60)
    assert violations == []
    assert response.startswith("I can refund your tee. This part is extra padding. Proposed refund")


def test_compose_rejects_unknown_fact_names() -> None:
    with pytest.raises(TypeError):
        compose_response("hi", PROPOSE, proposals=REFUND_PROPOSAL)


# ----------------------------------------------------------------------------------------------
# Property: every template passes the guard against its own facts
# ----------------------------------------------------------------------------------------------

ids = st.builds(lambda p, n: f"{p}{n}", st.sampled_from(["O", "ORD-", "K"]), st.integers(1, 99999))
sizes = st.sampled_from(["XS", "S", "M", "L", "XL", "2XL", "9", "10.5"])
cents = st.integers(min_value=0, max_value=10**7)
codes = st.sampled_from(sorted(REASON_TEXT))


@st.composite
def scenario(draw: Any) -> tuple[str, dict[str, Any]]:
    size_a, size_b = draw(sizes), draw(sizes)
    catalog = [{"sku": f"ITEM-{size_a}", "size": size_a}, {"sku": f"ITEM-{size_b}X", "size": size_b}]
    order_id, line_id = draw(ids), f"L{draw(st.integers(1, 30))}"
    remedy = draw(st.sampled_from(["refund", "exchange"]))
    proposal = {"proposal_id": f"P-{draw(st.integers(1, 9999)):04d}", "order_id": order_id, "line_id": line_id,
                "remedy": remedy, "original_sku": catalog[0]["sku"],
                "replacement_sku": catalog[1]["sku"] if remedy == "exchange" else None,
                "amount_cents": 0 if remedy == "exchange" else draw(cents),
                "eligibility": {"delivered_on": "2026-01-02", "days_since_delivery": draw(st.integers(0, 30)), "window_days": 30}}
    transaction = {"transaction_id": f"TX-{draw(st.integers(1, 9999)):04d}", "return_id": "RET-0001",
                   ("exchange_id" if remedy == "exchange" else "refund_id"): "RF-0001", "amount_cents": proposal["amount_cents"]}
    line = {"order_id": order_id, "line_id": line_id, "sku": catalog[0]["sku"], "size": size_a, "paid_cents": draw(cents)}
    reasons = draw(st.lists(st.one_of(codes, st.fixed_dictionaries(
        {"code": st.sampled_from(["OUT_OF_STOCK", "RETURN_WINDOW_EXPIRED"]), "size": sizes,
         "days_since_delivery": st.integers(31, 400)})), max_size=3))
    decision = draw(st.sampled_from(DECISIONS))
    facts = dict(proposal=proposal, transaction=transaction, reasons=reasons, alternatives=[catalog[1]],
                 candidates=[line], replay=draw(st.booleans()), reconciled=draw(st.booleans()), target=line,
                 catalog=catalog)
    return decision, facts


@settings(max_examples=300, deadline=None)
@given(scenario())
def test_templates_always_pass_their_own_guard(case: tuple[str, dict[str, Any]]) -> None:
    decision, facts = case
    reply = template_reply(decision, **facts)
    ok, violations = guard_reply(reply, decision, facts_line(decision, **facts))
    assert ok, (violations, reply)
    if decision == CLARIFY:
        assert reply.endswith("?")
    assert not re.search(r"\b[a-z]+_[a-z_]+\b", reply)


def test_window_wording_comes_from_the_policy_constant() -> None:
    from desk.domain import WINDOW_DAYS
    from desk.render import REASON_TEXT, reason_text
    assert f"{WINDOW_DAYS}-day" in REASON_TEXT["RETURN_WINDOW_EXPIRED"]
    assert f"{WINDOW_DAYS}-day" in reason_text("RETURN_WINDOW_EXPIRED", {"days_since_delivery": 31})


def test_the_guard_compares_amounts_through_the_one_money_formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """C7: change the money format and a faithful reply still passes, in either spelling."""
    import desk.render as render
    monkeypatch.setattr(render, "money", lambda cents: f"USD {cents // 100}.{cents % 100:02d}")
    facts = render.facts_line(PROPOSE, proposal=REFUND_PROPOSAL, catalog=CAT)
    assert "USD 30.00" in facts
    assert render.guard_reply("I can refund $30.00 for it.", PROPOSE, facts, SKUS) == (True, [])
    assert render.guard_reply("I can refund USD 30.00 for it.", PROPOSE, facts, SKUS) == (True, [])
    assert render.guard_reply("I can refund USD 40.00.", PROPOSE, facts, SKUS)[1] == ["money:USD 40.00"]


def test_clarify_with_no_facts_is_just_the_question() -> None:
    assert compose_response("Hi! How can I help you today?", CLARIFY) == ("Hi! How can I help you today?", [])
