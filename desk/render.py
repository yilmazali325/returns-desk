"""Customer-facing wording: money, reason codes, code-built facts, templates and the reply guard.

The LLM writes one or two friendly sentences. Every fact the customer reads comes from code:
  facts_line()       the authoritative line for a decision: amount, item, IDs, sizes, reasons
  template_reply()   a complete reply built from the facts alone (used when the LLM text fails)
  guard_reply()      rejects LLM text that adds a number, ID, SKU or size the facts do not
                     contain, claims success before a commit, reports failure after one, or
                     (for clarify) asks no question (SPEC §6)
  compose_response() LLM sentence(s) + facts line, or the template on any guard violation

Measured motivation (docs/notes/spec_critique.json): after a commit timeout the model wrote "your
refund failed, please try again" although the money had moved. Another reply named a size the
proposal did not contain. Neither can reach the customer through this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from desk.domain import CLARIFY, COMPLETED, DECISIONS, DENY, ERROR, PROPOSE, WINDOW_DAYS
from desk.grounding import extract_sizes
from desk.sanitize import injection_markers

MAX_LLM_CHARS = 400

CONFIRM_TEXT = "Nothing has changed yet; please confirm this proposal to proceed."


def money(cents: int) -> str:
    """Integer cents -> '$12.50' ('-$1.50', '$1,234.00'). Floats are refused: money is never a float.

    The ONE money formatter: facts, templates, the Resolution report and the UI server use it, and
    the reply guard checks an amount by formatting it with this function (see _money_violations)."""
    if isinstance(cents, bool) or not isinstance(cents, int):
        raise TypeError(f"money() takes integer cents, got {type(cents).__name__}")
    sign = "-" if cents < 0 else ""
    dollars, rest = divmod(abs(cents), 100)
    return f"{sign}${dollars:,}.{rest:02d}"


# ----------------------------------------------------------------------------------------------
# Reason codes in plain language (SPEC §3, §4, §6). ORDER_NOT_FOUND has one wording for "does not
# exist" and "belongs to someone else", so nothing about another customer leaks.
# ----------------------------------------------------------------------------------------------

REASON_TEXT: dict[str, str] = {
    # eligibility (policy.eligibility)
    "ORDER_NOT_FOUND": "We could not find that order on your account.",
    "LINE_NOT_FOUND": "We could not find that item in the order.",
    "NOT_DELIVERED": "This order has not been delivered yet, so it cannot be returned.",
    "MISSING_DELIVERY_DATE": "There is no delivery date on record, so we cannot confirm the return window.",
    "DELIVERY_IN_FUTURE": "The recorded delivery date is in the future, so a return cannot be processed yet.",
    "RETURN_WINDOW_EXPIRED": f"It is outside the {WINDOW_DAYS}-day return window.",
    "OPENED": "The item has been opened; only unopened items can be returned.",
    "FINAL_SALE": "The item was bought as final sale, which cannot be returned.",
    "ALREADY_RETURNED": "This item has already been returned.",
    # exchange options (policy.exchange_options, grounding.resolve_size)
    "SAME_SIZE": "That is the size you already have.",
    "DIFFERENT_PRODUCT": "An exchange can only be for a different size of the same product.",
    "PRICE_MISMATCH": "That size has a different price, and an exchange must keep the same price.",
    "OUT_OF_STOCK": "That size is out of stock right now.",
    "SKU_NOT_FOUND": "This item is no longer in our catalog, so it cannot be exchanged.",
    "SIZE_NOT_OFFERED": "That size is not offered for this product.",
    "NO_EXCHANGE_AVAILABLE": "No other size of this product is available for an exchange right now.",
    # gateway and execution (tools.py)
    "REQUEST_ALREADY_COMPLETED": "This request was already used for a different item, so it cannot be used again.",
    "REMEDY_MISMATCH": "The proposed remedy does not match what you asked for.",
    "REPLACEMENT_NOT_REQUESTED": "That replacement is not the size you asked for.",
    "CONFIRMATION_REQUIRED": "A chat message alone cannot authorize this; please use the explicit confirmation.",
    "CONFIRMATION_CANCELLED": "You asked us not to go ahead, so nothing was processed.",
    "PROPOSAL_NOT_OPEN": "That proposal is no longer open, so it cannot be confirmed.",
    "BINDING_MISMATCH": "The proposal no longer matches the request, so nothing was changed.",
    "TIMEOUT": ("The final step did not answer in time and the result is not verified yet. Please do not "
                "submit it again; we will check its status first."),
    # grounding gates (grounding.check_*)
    "TARGET_NOT_GROUNDED": "We could not tell which item you mean.",
    "TARGET_MISMATCH": "That item does not match the order or line you mentioned.",
    "AMBIGUOUS_TARGET": "We could not tell which item you mean.",
    "NO_DECISION": "We have not prepared a proposal yet; nothing was changed.",
    "SIZE_UNSPECIFIED": "We need to know which size you would like instead.",
    "REMEDY_UNCLEAR": "We are not sure whether you would like a refund or an exchange.",
    # agent protocol slips (never the customer's fault; normally invisible once the turn ends)
    "EXECUTE_LIMIT": "Only one confirmation can be carried out per message.",
    "NO_TARGET": "We could not tell which item you mean.",
    "TOOL_NOT_ALLOWED": "Our assistant made an internal mistake; nothing was changed.",
    "UNKNOWN_TOOL": "Our assistant made an internal mistake; nothing was changed.",
    "DUPLICATE_CALL": "Our assistant made an internal mistake; nothing was changed.",
    # our side
    "INVALID_ARGS": "Our assistant made an internal mistake; nothing was changed.",
    "LLM_ERROR": "Our assistant could not finish this step; nothing was changed.",
    "NOT_EXECUTED": "Your confirmation was not carried out yet; nothing was changed. Please confirm again.",
    "BUDGET_EXHAUSTED": "Our assistant could not finish this step; nothing was changed.",
    "INTERNAL": "Something went wrong on our side.",
}
GENERIC_REASON = "This request cannot be completed as asked."
_EXCHANGE_SIZE_CODES = {"OUT_OF_STOCK", "PRICE_MISMATCH", "SIZE_NOT_OFFERED", "SAME_SIZE"}


def reason_text(code: str, details: Mapping[str, Any] | None = None) -> str:
    """Plain-language sentence for a reason code. Details make it specific when available."""
    details = details or {}
    days = details.get("days_since_delivery")
    if code == "RETURN_WINDOW_EXPIRED" and isinstance(days, int):
        window = details.get("window_days") or WINDOW_DAYS
        return f"It was delivered {days} days ago, outside the {window}-day return window."
    named = details.get("named")  # the id the customer typed, e.g. "ORD-99" (never another customer's data)
    if code == "ORDER_NOT_FOUND" and named:
        return f"We could not find order {named} on your account."
    if code == "LINE_NOT_FOUND" and named:
        return f"We could not find {named} in that order."
    size = details.get("size")
    if code in _EXCHANGE_SIZE_CODES and size:
        sku = details.get("sku")
        subject = f"Size {size}" + (f" ({sku})" if sku else "")
        return {
            "OUT_OF_STOCK": f"{subject} is out of stock right now.",
            "PRICE_MISMATCH": f"{subject} has a different price, and an exchange must keep the same price.",
            "SIZE_NOT_OFFERED": f"{subject} is not offered for this product.",
            "SAME_SIZE": f"{subject} is the size you already have.",
        }[code]
    return REASON_TEXT.get(code, GENERIC_REASON)


def _reason_items(reasons: Iterable[Any] | None) -> list[tuple[str, Mapping[str, Any]]]:
    """Accept reasons as codes ('OPENED') or dicts ({'code': 'OUT_OF_STOCK', 'size': 'L', ...})."""
    items = []
    for reason in reasons or ():
        if isinstance(reason, Mapping):
            items.append((str(reason.get("code", "")), reason))
        else:
            items.append((str(reason), {}))
    return items


# ----------------------------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------------------------

def _size_lookup(catalog: Sequence[Any] | None, *items: Any) -> dict[str, str]:
    """sku -> size from the catalog and from any line summaries we were given."""
    sizes: dict[str, str] = {}
    for entry in list(catalog or ()) + [i for group in items for i in group]:
        sku, size = _get(entry, "sku"), _get(entry, "size")
        if sku and size and sku not in sizes:
            sizes[str(sku)] = str(size)
    return sizes


def _get(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict or from an object with attributes (catalog rows may be models)."""
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _sku_text(sku: Any, sizes: Mapping[str, str]) -> str:
    """'SCARF-L (size L)', or just the SKU when its size is unknown."""
    if not sku:
        return "the item"
    size = sizes.get(str(sku))
    return f"{sku} (size {size})" if size else str(sku)


def _where(order_id: Any, line_id: Any) -> str:
    """'order ORD-7Q, line LN-3' (either part may be missing)."""
    return ", ".join(part for part in (f"order {order_id}" if order_id else "",
                                       f"line {line_id}" if line_id else "") if part)


def _item_text(sku: Any, sizes: Mapping[str, str], order_id: Any, line_id: Any) -> str:
    where = _where(order_id, line_id)
    return _sku_text(sku, sizes) + (f", {where}" if where else "")


def _days_ago(days: int) -> str:
    return "today" if days == 0 else "1 day ago" if days == 1 else f"{days} days ago"


def _delivery_text(eligibility: Mapping[str, Any]) -> str:
    days, delivered = eligibility.get("days_since_delivery"), eligibility.get("delivered_on")
    if not isinstance(days, int) or not delivered:
        return ""
    window = eligibility.get("window_days") or WINDOW_DAYS
    return f"Delivered {delivered}, {_days_ago(days)} (within the {window}-day return window)."


def _alternative_text(option: Any, sizes: Mapping[str, str]) -> str:
    if isinstance(option, str):
        return option
    sku = _get(option, "sku")
    size = _get(option, "size") or sizes.get(str(sku))
    return f"{sku} (size {size})" if size else str(sku)


def _reasons_text(items: list[tuple[str, Mapping[str, Any]]]) -> str:
    return " ".join(_unique_texts(reason_text(code, details) for code, details in items))


def _unique_texts(texts: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for text in texts:
        seen.setdefault(text, None)
    return list(seen)


def _propose_facts(proposal: Mapping[str, Any] | None, sizes: Mapping[str, str],
                   reasons: list[tuple[str, Mapping[str, Any]]]) -> list[str]:
    if not proposal:
        return ["A proposal is being prepared.", CONFIRM_TEXT]
    order_id, line_id, sku = proposal.get("order_id"), proposal.get("line_id"), proposal.get("original_sku")
    amount = proposal.get("amount_cents")
    amount_text = money(amount) if isinstance(amount, int) and not isinstance(amount, bool) else "an amount to be confirmed"
    if proposal.get("remedy") == "exchange":
        replacement = _sku_text(proposal.get("replacement_sku"), sizes)
        offer = (f"Proposed exchange ({_where(order_id, line_id)}): {_sku_text(sku, sizes)} for {replacement}; "
                 f"price difference {amount_text} (same price).")
    else:
        offer = f"Proposed refund: {amount_text} for {_item_text(sku, sizes, order_id, line_id)}."
    proposal_id = proposal.get("proposal_id")
    return [offer, _delivery_text(proposal.get("eligibility") or {}),
            f"Proposal {proposal_id}." if proposal_id else "", _reasons_text(reasons), CONFIRM_TEXT]


def _completed_facts(transaction: Mapping[str, Any] | None, proposal: Mapping[str, Any] | None,
                     target: Mapping[str, Any] | None, sizes: Mapping[str, str],
                     replay: bool, reconciled: bool) -> list[str]:
    tx = dict(transaction or {})
    source = [tx, dict(proposal or {}), dict(target or {})]

    def first(*names: str) -> Any:
        return next((s[n] for s in source for n in names if s.get(n) not in (None, "")), None)

    remedy = first("action", "remedy") or ("exchange" if tx.get("exchange_id") else "refund")
    order_id, line_id, sku = first("order_id"), first("line_id"), first("original_sku", "sku")
    amount = first("amount_cents")
    amount_text = money(amount) if isinstance(amount, int) and not isinstance(amount, bool) else "the agreed amount"
    if remedy == "exchange":
        replacement = _sku_text(first("replacement_sku"), sizes)
        done = (f"Completed exchange ({_where(order_id, line_id)}): {_sku_text(sku, sizes)} for {replacement}; "
                f"amount {amount_text}.")
    else:
        done = f"Completed refund: {amount_text} for {_item_text(sku, sizes, order_id, line_id)}."
    ids = [f"{label} {tx[key]}" for key, label in (("transaction_id", "Transaction"), ("refund_id", "refund"),
                                                    ("exchange_id", "exchange"), ("return_id", "return"))
           if tx.get(key)]
    parts = [done, ("; ".join(ids) + ".") if ids else ""]
    if replay:
        what = "exchange" if remedy == "exchange" else "refund"
        parts.append(f"This request was already processed earlier — no new {what} or charge was made.")
    if reconciled:
        parts.append("The final step timed out, but we checked the record and verified it was completed exactly once.")
    return parts


def _deny_facts(target: Mapping[str, Any] | None, sizes: Mapping[str, str],
                reasons: list[tuple[str, Mapping[str, Any]]], alternatives: Sequence[Any]) -> list[str]:
    parts = [_target_text(target, sizes), _reasons_text(reasons) or GENERIC_REASON]
    if alternatives:
        parts.append("Available instead: " + ", ".join(_alternative_text(a, sizes) for a in alternatives) + ".")
    return parts


def _target_text(target: Mapping[str, Any] | None, sizes: Mapping[str, str]) -> str:
    if not target:
        return ""
    sku = target.get("sku") or target.get("original_sku")
    return f"Item: {_item_text(sku, sizes, target.get('order_id'), target.get('line_id'))}."


def _candidate_text(number: int, line: Mapping[str, Any], sizes: Mapping[str, str]) -> str:
    text = f"{number}) {_item_text(line.get('sku'), sizes, line.get('order_id'), line.get('line_id'))}"
    paid = line.get("paid_cents")
    if isinstance(paid, int) and not isinstance(paid, bool):
        text += f", paid {money(paid)}"
    return text


def _clarify_facts(target: Mapping[str, Any] | None, sizes: Mapping[str, str],
                   reasons: list[tuple[str, Mapping[str, Any]]], alternatives: Sequence[Any],
                   candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    parts = [_reasons_text(reasons), _target_text(target, sizes)]
    if candidates:
        label = "Matching item on your account" if len(candidates) == 1 else "Matching items on your account"
        parts.append(f"{label}: " + "; ".join(_candidate_text(n, c, sizes) for n, c in enumerate(candidates, 1)) + ".")
    if alternatives:
        parts.append("Sizes available for an exchange: " + ", ".join(_alternative_text(a, sizes) for a in alternatives) + ".")
    return parts


def _check_decision(decision: str) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; expected one of {DECISIONS}")


def facts_line(
    decision: str,
    proposal: Mapping[str, Any] | None = None,
    transaction: Mapping[str, Any] | None = None,
    reasons: Iterable[Any] | None = None,
    alternatives: Sequence[Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    replay: bool = False,
    reconciled: bool = False,
    *,
    target: Mapping[str, Any] | None = None,
    catalog: Sequence[Any] | None = None,
) -> str:
    """The code-built facts for a decision. Every number, ID, SKU and size the customer reads.

    target (a line summary or eligibility) names the item for deny/clarify. catalog lets us show
    each SKU's size, which the guard then allows the LLM to mention.
    """
    _check_decision(decision)
    candidates = list(candidates or ())
    alternatives = list(alternatives or ())
    sizes = _size_lookup(catalog, candidates, [target] if target else [])
    items = _reason_items(reasons)
    if decision == PROPOSE:
        parts = _propose_facts(proposal, sizes, items)
    elif decision == COMPLETED:
        parts = _completed_facts(transaction, proposal, target, sizes, replay, reconciled)
    elif decision == DENY:
        parts = _deny_facts(target, sizes, items, alternatives)
    elif decision == CLARIFY:
        parts = _clarify_facts(target, sizes, items, alternatives, candidates)
    else:
        parts = [_reasons_text(items) or "Something went wrong on our side."]
    return " ".join(part for part in parts if part)


# ----------------------------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------------------------

def _clarify_question(codes: set[str], candidates: Sequence[Any], alternatives: Sequence[Any]) -> str:
    if "CONFIRMATION_CANCELLED" in codes:
        return "Is there anything else I can help you with?"
    if "REMEDY_UNCLEAR" in codes:
        return "Would you like a refund or an exchange for a different size?"
    if "SIZE_UNSPECIFIED" in codes or (alternatives and not candidates):
        return "Which size would you like instead?"
    if len(candidates) == 1:
        return "Is this the item you mean?"
    if candidates:
        return "Which of these items do you mean?"
    return "Could you tell me the order number and which item it is?"


def template_reply(
    decision: str,
    proposal: Mapping[str, Any] | None = None,
    transaction: Mapping[str, Any] | None = None,
    reasons: Iterable[Any] | None = None,
    alternatives: Sequence[Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    replay: bool = False,
    reconciled: bool = False,
    *,
    target: Mapping[str, Any] | None = None,
    catalog: Sequence[Any] | None = None,
) -> str:
    """A complete, correct reply from code facts alone. Used whenever the LLM text is unusable."""
    reasons = list(reasons or ())
    facts = facts_line(decision, proposal, transaction, reasons, alternatives, candidates, replay, reconciled,
                       target=target, catalog=catalog)
    codes = {code for code, _ in _reason_items(reasons)}
    if decision == PROPOSE:
        lead = ("I still need your explicit confirmation before anything happens."
                if "CONFIRMATION_REQUIRED" in codes else "Here is what I can do.")
    elif decision == COMPLETED:
        lead = "This was already taken care of." if replay else "Thanks for your patience." if reconciled else "All done."
    elif decision == DENY:
        lead = "I'm sorry, but I can't do that."
    elif decision == CLARIFY:
        question = _clarify_question(codes, list(candidates or ()), list(alternatives or ()))
        return f"I want to make sure I get this right. {facts} {question}"
    else:
        lead = "Sorry, something went wrong on our side."
    return f"{lead} {facts}"


# ----------------------------------------------------------------------------------------------
# Reply guard
# ----------------------------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\d)"
    r"|(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?\s*(?:usd|dollars?|bucks)\b"
    r"|\busd\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\d)",
    re.IGNORECASE,
)
_FOREIGN_CURRENCY_RE = re.compile(r"[€£¥₹]|\b(?:eur|euros?|gbp|pounds?|yen|rupees?)\b", re.IGNORECASE)
_ID_RE = re.compile(r"(?<![\w-])#?[A-Za-z]{1,6}[-_]?\d+(?:[-_]?[A-Za-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
_CAPS_SKU_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z]+_[a-z_]+\b")
_REMEDY_WORD_RE = re.compile(r"\b(refund|exchang)\w*", re.IGNORECASE)
# Claims that the request already happened (SPEC §6; SEC-2, RT-8). Checked with the future/negation
# prefix rule below, so "will be refunded", "to complete" and "nothing has been processed" pass.
_SUCCESS_RE = re.compile(
    r"\b(?:refunded|exchanged|processed|completed?|finished|issued|credited|reimbursed|approved|confirmed"
    r"|successfully|successful|went through|all set|all done|is done|been done|done"
    r"|on (?:its|the) way|been sent|no need to confirm)\b",
    re.IGNORECASE,
)
# Money "back on your card" / "you will receive" are claims unless the sentence makes them depend
# on something still to happen ("once you confirm, you will receive ...").
_RESULT_CLAIM_RE = re.compile(
    r"\bback (?:on|in|to) your\b|\byou(?: will|'ll) (?:receive|get|see)\b|\bhas been (?:sent|returned)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(r"\b(?:once|after|when|if|as soon as|confirm\w*|until|before)\b", re.IGNORECASE)
# The desk has no e-mail, bank, label or courier system, and only the confirm control confirms.
_EXTERNAL_CHANNEL_RE = re.compile(
    r"\b(?:e-?mails?|inbox|bank statement|statement|shipping label|return label|prepaid label|courier|pick-?up)\b",
    re.IGNORECASE,
)
_TYPED_YES_RE = re.compile(r"\b(?:reply|type|say|answer|respond|write|text)\b[^.?!]{0,20}?\byes\b", re.IGNORECASE)
# On clarify/deny/error nothing will happen this turn, so a promise to act is also false.
_PROMISE_RE = re.compile(
    r"\b(?:i will|i'll|we will|we'll) (?:proceed|process|refund|exchange|issue|send)\b"
    r"|\bwill be (?:refunded|exchanged|processed|issued)\b",
    re.IGNORECASE,
)
_FAILURE_RE = re.compile(
    r"\b(?:failed|failure|unable|could not|couldn't|cannot|can't|can not|unsuccessful|declined|rejected"
    r"|denied|did not go through|didn't go through|not processed|not completed|try again|error)\b",
    re.IGNORECASE,
)
# A success word right after one of these talks about the future or denies it ("will be refunded",
# "nothing has been processed"), so it is not a claim of success.
_NOT_A_CLAIM = {"will", "would", "can", "could", "may", "might", "be", "once", "after", "when", "until",
                "before", "if", "not", "no", "never", "nothing", "to"}


def _money_amounts(text: str) -> set[int]:
    """Cents of every money mention ('$30', '$30.00', '30 dollars', 'USD 30')."""
    amounts = set()
    for match in _MONEY_RE.finditer(text):
        whole = match.group(1) or match.group(3) or match.group(5)
        frac = match.group(2) or match.group(4) or match.group(6) or "0"
        amounts.add(int(whole.replace(",", "")) * 100 + int(frac.ljust(2, "0")))
    return amounts


def _money_violations(text: str, facts: str) -> list[str]:
    """An amount in the reply must be one the facts show, compared through money() itself, so a
    change of the currency format can never make the guard reject a faithful reply."""
    return [f"money:{money(c)}" for c in sorted(_money_amounts(text)) if money(c) not in facts]


_NEGATION_RE = re.compile(r"\b(?:no|not|nothing|never|won't|will not)\b", re.IGNORECASE)


def _promises(text: str) -> list[str]:
    """'I'll process your refund' on a turn where nothing happens; 'no refund will be processed' is fine."""
    hits = []
    for match in _PROMISE_RE.finditer(text):
        clause = re.split(r"[.,;:!?]", text[: match.start()])[-1]
        if not _NEGATION_RE.search(clause):
            hits.append(match.group(0).casefold())
    return hits


_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")  # "$30.00" does not end a sentence


def _result_claims(text: str) -> list[str]:
    """'Your money is back on your card' unless the same sentence makes it conditional."""
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(text)] + [len(text)]
    hits = []
    for match in _RESULT_CLAIM_RE.finditer(text):
        start = max([0] + [e for e in ends if e <= match.start()])
        stop = min(e for e in ends if e >= match.end())
        if not _CONDITIONAL_RE.search(text[start:stop]):
            hits.append(match.group(0).casefold())
    return hits


def _id_tokens(text: str) -> set[str]:
    return {re.sub(r"[#_\-]", "", m.group(0)).casefold() for m in _ID_RE.finditer(_MONEY_RE.sub(" ", text))}


def _bare_numbers(text: str) -> set[str]:
    stripped = _ID_RE.sub(" ", _MONEY_RE.sub(" ", text))
    out = set()
    for match in _NUMBER_RE.finditer(stripped):
        whole, _, frac = match.group(0).partition(".")
        whole, frac = whole.lstrip("0") or "0", frac.rstrip("0")
        out.add(f"{whole}.{frac}" if frac else whole)
    return out


def _sku_mentions(text: str, known_skus: Iterable[str]) -> set[str]:
    found = {m.group(0).casefold() for m in _CAPS_SKU_RE.finditer(text)}
    for sku in known_skus:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(sku) + r"(?![A-Za-z0-9])", text, re.IGNORECASE):
            found.add(sku.casefold())
    return found


def _claims(pattern: re.Pattern[str], text: str, allow_future: bool) -> list[str]:
    """Words of a claim class; with allow_future, skip those preceded by will/be/not/once/... ."""
    hits = []
    for match in pattern.finditer(text):
        clause = re.split(r"[.,;:!?]", text[: match.start()])[-1]
        before = set(re.findall(r"[a-z']+", clause.casefold())[-3:])
        if allow_future and before & _NOT_A_CLAIM:
            continue
        hits.append(match.group(0).casefold())
    return hits


def guard_reply(text: str | None, decision: str, facts: str,
                known_skus: Iterable[str] = ()) -> tuple[bool, list[str]]:
    """Check LLM reply text against the code facts. Returns (ok, violations).

    A violation is any number, money amount, ID, SKU or size absent from `facts`, a success claim
    when decision != completed ("is complete", "back on your card", "no need to confirm"), a promise
    to act on clarify/deny/error, a failure claim when decision == completed, a channel the desk
    does not have (e-mail, bank statement, label, courier, "reply yes"), a clarify with no '?',
    control tokens, internal tool names, or (propose/completed) the other remedy's name.
    """
    _check_decision(decision)
    text = (text or "").strip()
    if not text:
        return False, ["empty"]
    known_skus = list(known_skus)
    violations: list[str] = []
    if injection_markers(text) or "‹|" in text:
        violations.append("control_token")
    violations += [f"internal_name:{m}" for m in sorted(set(_SNAKE_CASE_RE.findall(text)))]
    violations += _money_violations(text, facts)
    violations += [f"currency:{m.group(0)}" for m in _FOREIGN_CURRENCY_RE.finditer(text)]
    violations += [f"id:{t}" for t in sorted(_id_tokens(text) - _id_tokens(facts))]
    violations += [f"number:{n}" for n in sorted(_bare_numbers(text) - _bare_numbers(facts))]
    violations += [f"sku:{s}" for s in sorted(_sku_mentions(text, known_skus) - _sku_mentions(facts, known_skus))]
    violations += [f"size:{s}" for s in sorted(set(extract_sizes(text)) - set(extract_sizes(facts)))]
    if decision in (PROPOSE, COMPLETED):
        allowed = {w.casefold() for w in _REMEDY_WORD_RE.findall(facts)}
        violations += [f"remedy:{w}" for w in sorted({w.casefold() for w in _REMEDY_WORD_RE.findall(text)} - allowed)]
    violations += [f"external_channel:{m.group(0).casefold()}" for m in _EXTERNAL_CHANNEL_RE.finditer(text)]
    violations += ["confirmation_channel:typed yes" for _ in _TYPED_YES_RE.finditer(text)]
    if decision != COMPLETED:
        violations += [f"success_claim:{w}" for w in _claims(_SUCCESS_RE, text, allow_future=True)]
        violations += [f"success_claim:{w}" for w in _result_claims(text)]
    else:
        violations += [f"failure_claim:{w}" for w in _claims(_FAILURE_RE, text, allow_future=False)]
    if decision in (CLARIFY, DENY, ERROR):
        violations += [f"promise:{w}" for w in _promises(text)]
    if decision == CLARIFY and "?" not in text:
        violations.append("no_question")
    return not violations, violations


def trim_at_sentence(text: str | None, max_chars: int = MAX_LLM_CHARS) -> str:
    """Collapse whitespace and cut to max_chars at the last sentence end (never mid-word)."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", window)]
    if ends:
        return window[: ends[-1]]
    cut = window.rsplit(" ", 1)[0] if " " in window else window[: max_chars - 1]
    return cut.rstrip(",;:") + "…"


def compose_response(
    llm_text: str | None,
    decision: str,
    *,
    known_skus: Iterable[str] = (),
    max_llm_chars: int = MAX_LLM_CHARS,
    **facts: Any,
) -> tuple[str, list[str]]:
    """Final customer response: LLM sentence(s) + facts line, or the template if the guard fires.

    `facts` are facts_line's keyword arguments (proposal, transaction, reasons, alternatives,
    candidates, replay, reconciled, target, catalog). Returns (response, guard_violations); the
    caller audits a non-empty violation list as a guard override.
    """
    facts_text = facts_line(decision, **facts)
    sentence = trim_at_sentence(llm_text, max_llm_chars)
    ok, violations = guard_reply(sentence, decision, facts_text, known_skus)
    if not ok:
        return template_reply(decision, **facts), violations
    if decision == CLARIFY:  # keep the question last so the reply ends by asking it
        return " ".join(p for p in (facts_text, sentence) if p), []
    return " ".join(p for p in (sentence, facts_text) if p), []
