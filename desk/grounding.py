"""Deterministic language grounding: token extraction and checks of the agent's reading (SPEC §5).

The Resolution agent reads the conversation and decides the target (order + line), the remedy,
the requested size and whether to ask. This module never makes that decision. It does two things:

1. `analyze()` extracts what the customer literally typed: order and line IDs, product words,
   size words and their frame ("my large tee" describes the item, "for a large" asks for one),
   remedy and cancel wording, and a regex reading of the target. The orchestrator uses the facts
   (IDs, cancel/remedy wording) to BLOCK, and the regex reading only to give the agent one round
   of feedback. The test double FakePolicyLLM uses the reading as its "brain".
2. The checks (`check_target`, `check_evidence`, `check_remedy`, `check_requested_size`) verify
   the agent's handoff against those facts and answer ok or a code the agent gets back as
   feedback. Size resolution itself is policy.resolve_size: there is one resolver in the desk.

It is pure: no clock, no randomness, no I/O.

Inputs are plain dicts, or objects with the same attribute names:
    line summary  {order_id, line_id, sku, product_id, size, paid_cents, aliases, delivered_on}
    catalog item  {sku, product_id, size, price_cents, stock}

Each message is read in fixed steps:
    1. normalize: sanitize, casefold, expand contractions ("I'm" -> "i am", so it never gives M),
       and fix one-letter typos of the two strong remedy words ("refnd" -> "refund")
    2. mask money amounts and identifiers: known order/line IDs and SKUs, "line 2" / "order #100",
       ordinals and unknown ID-shaped tokens. Masked text is never read as a size.
    3. mask product words: product ids, SKUs and a generic garment lexicon
    4. tokenize and find size mentions. The words around each mention decide its role: it either
       describes the item the customer has ("my M shirt") or asks for a new one ("for a large",
       "is an L available?")
    5. remedy cue, direction cues (too small -> UP), cancel cue, policy/status question

Later messages refine earlier ones. Each kind of reference (orders, lines, products, original
sizes, requested size, remedy) comes from the most recent message that mentions that kind. So
"sorry, I meant LN-4" replaces LN-3, while a follow-up that only adds a size keeps the earlier order
and line. A weak remedy word ("send it back") never replaces an earlier strong one ("exchange").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from desk.domain import (
    AMBIGUOUS_TARGET,
    EXCHANGE,
    LINE_NOT_FOUND,
    ORDER_NOT_FOUND,
    REFUND,
    REMEDY_MISMATCH,
    REMEDY_UNCLEAR,
    REPLACEMENT_NOT_REQUESTED,
    SIZE_NOT_OFFERED,
    SIZE_UNSPECIFIED,
    TARGET_MISMATCH,
)
from desk.policy import SIZE_WORDS, normalize_id, normalize_size, parse_date, resolve_size
from desk.sanitize import sanitize

# ----------------------------------------------------------------------------------------------
# Vocabulary shared with the orchestrator
# ----------------------------------------------------------------------------------------------

BOTH, NONE = "both", "none"
UP, DOWN, SAME = "UP", "DOWN", "SAME"  # requested-size tokens besides plain labels

# How an accepted handoff target was grounded (audited on set_target).
BY_CUSTOMER_IDS = "customer_ids"  # the order/line IDs the customer typed pin exactly this line
BY_ONLY_LINE = "only_line"  # the account has one line the request can be about
BY_AGENT = "agent"  # the agent cited the customer's words and code verified the citation
BY_AGENT_AND_GROUNDING = "agent+grounding"  # no citation, but code's own reading picks the same line
# The most common citation slip of a small model is a paraphrase ("the large tee"); say how to fix it.
NOT_A_QUOTE = "the evidence is not an exact quote: copy consecutive words of the customer's message exactly as typed"


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict, or from an object with attributes (pydantic model, dataclass)."""
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _unique(values: Iterable[str]) -> list[str]:
    """Drop duplicates, keep first-seen order (results must be deterministic)."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


_norm_id = normalize_id


# ----------------------------------------------------------------------------------------------
# Sizes (the vocabulary is policy.SIZE_WORDS; this section only finds size words in text)
# ----------------------------------------------------------------------------------------------

canonical_size = normalize_size

_MULTIWORD_SIZES = (  # longest first; single tokens are looked up in policy.SIZE_WORDS
    (("extra", "extra", "large"), "XXL"),
    (("extra", "extra", "small"), "XXS"),
    (("extra", "large"), "XL"),
    (("extra", "small"), "XS"),
    (("x", "large"), "XL"),
    (("x", "small"), "XS"),
    (("xx", "large"), "XXL"),
    (("xx", "small"), "XXS"),
    (("double", "xl"), "XXL"),
)
_LETTER_SIZES = {"s": "S", "m": "M", "l": "L"}
# "too large", "a bit small" and "runs small" describe fit, not the size label S/L.
_DEGREE_WORDS = {"too", "so", "very", "really", "bit", "little", "slightly", "way", "quite", "kind", "kinda"}
_FIT_VERBS = {"runs", "run", "running", "fits", "fit", "fitting", "feels", "feel", "comes", "came", "cut", "sized"}
_FIT_SIZE_WORDS = {"small", "large"}
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _size_word(token: str) -> str | None:
    """Ladder label of a size word token ('xl', 'x-large', 'medium'); bare s/m/l are not words."""
    if token in _LETTER_SIZES:
        return None
    return SIZE_WORDS.get(token.replace("-", ""))


def size_aliases(size: Any) -> list[str]:
    """Words a customer may use for a size, for the account card: 'M' -> ['M', 'medium', 'med']."""
    canon = canonical_size(size)
    words = [w for w, label in SIZE_WORDS.items() if label == canon and len(w) > 1 and not w[0].isdigit()]
    return _unique([str(size), canon] + words)


# ----------------------------------------------------------------------------------------------
# Products: a generic garment lexicon, so unseen product ids (HOODIE, JEANS, ...) still ground
# ----------------------------------------------------------------------------------------------

# (substrings of a product_id, words customers use for it). A key found inside a longer key's
# match is ignored, so "SHIRT" inside "SWEATSHIRT" does not add the shirt words.
_GARMENT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("TSHIRT", "TEE"), ("tee", "t-shirt", "tee shirt", "shirt", "top")),
    (("SWEATSHIRT", "HOODIE", "HOODY"), ("hoodie", "hoody", "sweatshirt", "hooded sweatshirt", "pullover")),
    (("POLO",), ("polo", "polo shirt", "shirt", "top")),
    (("SHIRT", "BLOUSE"), ("shirt", "top", "blouse")),
    (("TANKTOP", "TOP", "CAMI"), ("top", "tank top", "tank")),
    (("SWEATER", "JUMPER", "CARDIGAN", "KNIT"), ("sweater", "jumper", "cardigan", "knit")),
    (("JACKET", "COAT", "PARKA", "BLAZER", "ANORAK", "WINDBREAKER"),
     ("jacket", "coat", "parka", "blazer", "anorak", "windbreaker")),
    (("JEAN", "DENIM"), ("jeans", "denim", "pants", "trouser")),
    (("PANT", "TROUSER", "CHINO", "SLACK", "JOGGER", "CAPRI"), ("pants", "trouser", "slacks", "chino", "jogger")),
    (("SHORTS",), ("shorts",)),
    (("SHORTSLEEVE", "LONGSLEEVE"), ()),  # descriptors only: stops "SHORTS" matching inside them
    (("LEGGING",), ("legging",)),
    (("SKIRT",), ("skirt",)),
    (("DRESS", "GOWN"), ("dress", "gown")),
    (("SNEAKER", "TRAINER", "SHOE"), ("shoe", "sneaker", "trainer", "kicks")),
    (("BOOT",), ("boot", "shoe")),
    (("SANDAL",), ("sandal", "shoe")),
    (("SOCK",), ("sock",)),
    (("CAP", "HAT", "BEANIE"), ("cap", "hat", "beanie")),
    (("SCARF", "SCARVES"), ("scarf", "scarves")),
    (("GLOVE", "MITTEN"), ("glove", "mitten")),
    (("BELT",), ("belt",)),
    (("BAG", "BACKPACK", "TOTE"), ("bag", "backpack", "tote")),
    (("VEST", "GILET"), ("vest", "gilet")),
)
_ALL_GARMENT_WORDS = tuple(_unique(w for _, words in _GARMENT_GROUPS for w in words))


def product_aliases(product_id: Any) -> list[str]:
    """Customer words for a product id: 'TEE' -> ['tee', 't-shirt', 'tee shirt', 'shirt', 'top'].

    Works for unseen ids by substring ('MENS-HOODIE' gets the hoodie words). The id itself, as
    words, always comes first ('RAIN-JACKET' -> 'rain jacket').
    """
    if product_id is None:
        return []
    compact = re.sub(r"[^A-Z0-9]", "", str(product_id).upper())
    hits: list[tuple[int, int, int]] = []  # (start, end, group index)
    for group_index, (keys, _words) in enumerate(_GARMENT_GROUPS):
        for key in keys:
            hits.extend((m.start(), m.end(), group_index) for m in re.finditer(key, compact))
    kept = [
        hit for hit in hits
        if not any(o[0] <= hit[0] and hit[1] <= o[1] and o[1] - o[0] > hit[1] - hit[0] for o in hits)
    ]
    aliases = [re.sub(r"[\s_\-]+", " ", str(product_id)).strip().casefold()]
    for _start, _end, group_index in sorted(kept):
        aliases.extend(_GARMENT_GROUPS[group_index][1])
    return _unique(a for a in aliases if a)


def _phrase_pattern(words: list[str]) -> re.Pattern[str]:
    """Whole-word phrase, separators optional ('t-shirt' ~ 'tshirt' ~ 't shirt'), simple plural."""
    body = r"[\s\-_]?".join(re.escape(w) for w in words)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:e?s)?(?![a-z0-9])")


# ----------------------------------------------------------------------------------------------
# Text normalization and fixed cue patterns
# ----------------------------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_IRREGULAR_CONTRACTIONS = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\bcan'?t\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon'?t\b", "will not"),
        (r"\blet's\b", "let us"), (r"\bdont\b", "do not"), (r"\bdoesnt\b", "does not"),
        (r"\bdidnt\b", "did not"), (r"\bisnt\b", "is not"), (r"\bno (?:problem|worries|worry)\b", "fine"),
    )
)
_NOT_CONTRACTION = re.compile(r"(?<=[a-z])n't\b")
_GENERIC_CONTRACTION = re.compile(r"(?<=[a-z0-9])'(s|m|re|ve|ll|d)\b")
_CONTRACTION_SUFFIX = {"s": " is", "m": " am", "re": " are", "ve": " have", "ll": " will", "d": " would"}


def normalize_text(text: str | None) -> str:
    """Sanitize, casefold, expand contractions and fix one-letter typos of the remedy words.

    "I'm sure it's the tee" -> "i am sure it is the tee"; "refnd it plz" -> "refund it plz".
    """
    out = sanitize(text).translate(_APOSTROPHES).casefold()
    for pattern, replacement in _IRREGULAR_CONTRACTIONS:
        out = pattern.sub(replacement, out)
    out = _NOT_CONTRACTION.sub(" not", out)
    out = _GENERIC_CONTRACTION.sub(lambda m: _CONTRACTION_SUFFIX[m.group(1)], out)
    return _WORD_RE.sub(_fix_cue_typo, out)


# Customers type fast ("refnd", "exchnage"). A word one edit away from a strong remedy word counts
# as that word, so a typo does not leave the remedy to the model alone. Only these two long words:
# no common English word is one edit away from either.
_TYPO_TARGETS = ("refund", "exchange")
_WORD_RE = re.compile(r"[a-z]{5,}")


def _fix_cue_typo(match: re.Match[str]) -> str:
    word = match.group(0)
    for target in _TYPO_TARGETS:
        if word != target and not word.startswith(target) and _one_edit_apart(word, target):
            return target
    return word


def _one_edit_apart(a: str, b: str) -> bool:
    """True when one insertion, deletion, substitution or swap of adjacent letters turns a into b."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diffs) == 1:
            return True
        return len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = next((k for k, (x, y) in enumerate(zip(short, long_)) if x != y), len(short))
    return short[i:] == long_[i + 1:]


_STRONG_EXCHANGE_RE = re.compile(
    r"\b(?:exchang\w*|swap\w*|switch\w*|trade|replace\w*|(?:different|another|other) size"
    r"|size (?:up|down)|(?:up|down) (?:a|one) size|change (?:it|this|that|them|the size))\b"
)
_WEAK_EXCHANGE_RE = re.compile(r"\b(?:bigger|larger|smaller)\b")
_STRONG_REFUND_RE = re.compile(r"\b(?:refund\w*|money back|reimburs\w*|pay me back)\b")
_WEAK_REFUND_RE = re.compile(r"\b(?:return(?:s|ed|ing)?|(?:send|ship|take|give) (?:it |this |them |that )?back)\b")
_UP_RE = re.compile(
    r"\b(?:too (?:small|tight|short|snug|narrow)|bigger|larger|looser|size up|up (?:a|one) size"
    r"|(?:a|one) size up|next size up|(?:runs?|running|fits?|fitting) (?:a (?:bit|little) )?small)\b"
)
_DOWN_RE = re.compile(
    r"\b(?:too (?:big|large|loose|baggy|long|wide)|smaller|tighter|size down|down (?:a|one) size"
    r"|(?:a|one) size down|next size down|(?:runs?|running|fits?|fitting) (?:a (?:bit|little) )?(?:large|big))\b"
)
# A cue is ignored when the words just before it negate it: "not a refund", "instead of an exchange".
_NEGATED_BEFORE = re.compile(
    r"(?:\b(?:not|no|never|without)(?:\s+(?:want|need|asking for|looking for|interested in))?"
    r"|\binstead of|\bthan|\bforget(?:\s+about)?|\bskip|\bdrop)(?:\s+(?:a|an|the|any|my))?\s*$"
)
# Undecided wording: "should I return it or swap it?" mentions both remedies on purpose.
_HEDGE_RE = re.compile(r"\b(?:or|either|maybe|perhaps|not sure|unsure|undecided|which (?:one|is better)|whichever)\b")
# "in the same size" / "the same one again": the size the customer already has.
_SAME_SIZE_RE = re.compile(r"\bsame (?:size|one)\b")
# Questions about rules or status are not requests (RT-5): code never pushes them toward a proposal.
# Hypothetical wording always marks one; rule/status words only inside a question.
_HYPOTHETICAL_RE = re.compile(r"\b(?:just asking|hypothetical\w*|if i (?:wanted|were|was|decide\w*)|in general"
                              r"|what happens if)\b")
_RULE_OR_STATUS_RE = re.compile(r"\b(?:polic(?:y|ies)|how long|how many days|do you (?:offer|accept|allow)|status"
                                r"|where is my|(?:go|gone|went) through)\b")
# Money mentions are masked before sizes are read ("$32" is never size 32) and kept for evidence.
_MONEY_MENTION_RE = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{1,2}))?|\b(\d[\d,]*)(?:\.(\d{1,2}))?\s*(?:dollars?|bucks|usd)\b")

# --- cancel wording (CLARIFICATIONS #9): an explicit cancel verb that is not negated, or a message
# that OPENS with no/stop/wait/hold on and never comes round to consenting. "Please don't cancel",
# "there is no need to hold off", "sorry it took a while" and "I'll take that one instead" are not
# cancels; "No, cancel that", "Stop, no thanks" and "never mind" are.
_CANCEL_VERB_RE = re.compile(
    r"\bcancel\w*|\bchanged? my mind\b|\bnever ?mind\b|\bforget (?:it|about it|that)\b|\bkeep it\b"
    r"|\bcall (?:it|this|that) off\b|\b(?:do|did) not (?:proceed|confirm|process|do it|do this|go ahead)\b"
    r"|\bdo not want (?:it|this|that|to (?:go ahead|proceed|continue|do (?:it|this)))\b"
)
_CANCEL_NEGATED_BEFORE = re.compile(r"\b(?:not|never|no need to|no reason to|without)\s+(?:\w+\s+){0,2}$")
_HESITATION_START_RE = re.compile(r"^\W*(?:no|nope|nah|stop|wait|hold on|hold off|hang on)\b")
_CONSENT_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|ok|okay|confirm\w*|go ahead|proceed|do it|that is right|that is correct"
    r"|correct|sounds good|please do|approve\w*|agreed?|fine)\b"
)

# Words that, before a size or product, mark it as the one the customer WANTS (exchange context).
_STRONG_TRIGGERS = {"for", "to", "into"}
_WEAK_TRIGGERS = {"need", "want", "get", "like", "prefer", "take", "make", "receive"}
# Words that may sit between a trigger and the size without breaking the link: "for the same scarf in size L".
_FILLERS = {
    "a", "an", "the", "same", "one", "ones", "size", "sized", "sizes", "it", "this", "that", "them",
    "these", "those", "new", "different", "another", "other", "please", "or", "me", "us", "just",
    "pair", "pairs",
}
# "go with" introduces a choice ("let's go with the size 10"); a bare "with" usually does not.
_CHOICE_VERBS_BEFORE_WITH = {"go", "going", "goes"}
# "have" asks for something only in a request frame ("let me have the small", "can I have a large",
# "do you have an XL"); "I have the medium" describes the item, so a bare "have" is not a trigger.
_REQUEST_BEFORE_HAVE = {("let", "me"), ("can", "i"), ("could", "i"), ("may", "i"), ("i", "will"), ("i", "would"),
                        ("do", "you")}
_SEND_VERBS = {"send", "ship", "give", "mail"}  # "ship me an L" asks for one; "ship the L back" does not
_ASKING_ABOUT = {"what", "how"}  # "how about an L?" offers to take it
# Words right after a size that turn it into a request or an offer: "the M instead", "is an L in
# stock?", "the S is fine".
_OFFER_AFTER = (("instead",), ("please", "instead"), ("rather", "than"), ("available",), ("in", "stock"),
                ("back", "in", "stock"), ("is", "fine"), ("would", "be", "fine"), ("works",), ("would", "work"),
                ("is", "ok"), ("is", "okay"))
_CLAUSE_BREAKS = {".", ",", ";", ":", "!", "?", "(", ")", "and", "but", "because", "so", "then",
                  "also", "although", "though", "since", "while"}
_DETERMINERS = {"my", "the", "this", "that", "our", "his", "her", "their", "your"}
_POSSESSIVES = {"my", "our", "mine"}
# Words keep inner hyphens and dots, so "x-large", "2xl" and "10.5" are single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*|[.,;:!?()]")

# Identifier patterns that need no account knowledge.
_LINE_NUMBER_RE = re.compile(r"\b(?:line|item)(?:\s*(?:no\.?|number|num|nr\.?))?\s*#?\s*(\d+)(?![\w.])")
_ORDER_NUMBER_RE = re.compile(
    r"\border(?:\s*(?:no\.?|number|num|nr\.?|id))?\s*#?\s*(\d+)(?![\w.])|(?<![\w#])#\s*(\d+)(?![\w.])"
)
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
             "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
_ORDINAL_RE = re.compile(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(?:item|line|product|article)\b")
# letters + digits (+ up to 2 letters), e.g. ORD-99, LN-30, ord55, ORD-7Q
_ID_SHAPED_RE = re.compile(r"(?<![a-z0-9])#?([a-z]{1,4})[-_]?(\d{1,8})([a-z]{0,2})(?![a-z0-9])")
_NOT_ID_PREFIXES = {"size", "sz", "us", "eu", "uk", "no", "nr", "num", "item", "line", "x"}
# Prefixes of the desk's own record ids (P-0001, TX-0001, ...): naming one is not naming an order.
_RECORD_PREFIXES = {"p", "tx", "ret", "rf", "ex", "q", "ev", "req"}
_ORDER_CONTEXT_RE = re.compile(r"\b(?:order|ord|purchase)\b(?:\W+(?:no|number|num|nr|id))?\W*$")
_LINE_CONTEXT_RE = re.compile(r"\b(?:line|item)\b(?:\W+(?:no|number|num|nr))?\W*$")
_OTHER_CONTEXT_RE = re.compile(
    r"\b(?:customer|account|client|member|user|request|proposal|transaction|ticket|case|reference|ref"
    r"|tracking|confirmation|id)\b(?:\W+(?:id|no|number|num|is|was))*\W*$"
)
_ID_TOKEN = "zidref"  # stands in for every masked identifier; not a size, cue or filler word
_MONEY_TOKEN = "zmoney"  # stands in for every masked money amount
_PRODUCT_TOKEN = "zprod"  # + letters; one per product mention


def _product_token(index: int) -> str:
    """Placeholder word for product mention #index. Letters only, so it never looks like an ID."""
    return _PRODUCT_TOKEN + "".join(chr(ord("a") + int(d)) for d in str(index))


def _trailing_number(identifier: str) -> int | None:
    digits = re.findall(r"\d+", identifier)
    return int(digits[-1]) if digits else None


def _letter_prefix(identifier: str) -> str:
    match = re.match(r"[a-z]+", _norm_id(identifier))
    return match.group(0) if match else ""


def _id_pattern(identifier: str) -> re.Pattern[str] | None:
    """Whole-token pattern for a known ID, tolerant of how customers type it.

    'ORD7' matches 'ord7', '#ORD7', 'ORD-7', 'ord 7'; never 'ORD71'. Between a letter run and a
    digit run the separator is optional. Between two letter runs ('SCARF-M') it is required, so the
    plural 'scarfs' is not read as SKU 'SCARF-S'. All-digit IDs, single-word IDs and 1-char IDs
    return None: they are too easy to hit by accident ('1', 'a', 'tee'). All-digit IDs are matched
    only after a context word ('line 1', 'order #100'), and product ids go through the alias lexicon.
    """
    # Letter runs and digit runs, e.g. "SCARF-M" -> ["scarf", "m"], "ORD7" -> ["ord", "7"].
    pieces = re.findall(r"[a-z]+|\d+(?:\.\d+)*", str(identifier).casefold())
    joined = "".join(pieces)
    if len(joined) < 2 or joined.isdigit() or (joined.isalpha() and len(pieces) == 1):
        return None
    body = re.escape(pieces[0])
    for previous, piece in zip(pieces, pieces[1:]):
        # Adjacent runs of the same kind were split by a separator in the original, so require one.
        switches_kind = previous[-1].isdigit() != piece[0].isdigit()
        body += (r"[\s#_\-]*" if switches_kind else r"[\s#_\-]+") + re.escape(piece)
    return re.compile(r"(?<![a-z0-9])#?" + body + r"(?![a-z0-9])")


def _money_cents(match: re.Match[str]) -> int:
    whole = (match.group(1) or match.group(3)).replace(",", "")
    frac = match.group(2) or match.group(4) or "0"
    return int(whole) * 100 + int(frac.ljust(2, "0"))


# ----------------------------------------------------------------------------------------------
# Account view: the customer's lines and the catalog, prepared once per analyze() call
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _KnownId:
    kind: str  # "order" | "line" | "sku"
    value: str  # as written in state
    pattern: re.Pattern[str]


@dataclass
class _Account:
    lines: list[dict[str, Any]]
    order_ids: list[str]
    known_ids: list[_KnownId]
    sku_info: dict[str, tuple[str | None, str | None]]  # sku -> (product_id, size)
    phrases: list[tuple[re.Pattern[str], tuple[str, ...]]]  # longest first; () = garment not on account
    order_prefixes: set[str]
    line_prefixes: set[str]
    numeric_sizes: set[str]  # numeric sizes the catalog offers ("32", "10.5")
    colours: dict[str, str]  # sku -> colour word, when the catalog carries one

    def lines_of_order(self, order_id: str) -> list[dict[str, Any]]:
        return [line for line in self.lines if line["order_id"] == order_id]


def _line_view(line: Any, catalog_by_sku: dict[str, Any]) -> dict[str, Any]:
    """Plain-dict copy of a line summary; product_id/size filled from the catalog when missing."""
    sku = _field(line, "sku")
    product = catalog_by_sku.get(sku)
    product_id = _field(line, "product_id") or (_field(product, "product_id") if product is not None else None)
    size = _field(line, "size") or (_field(product, "size") if product is not None else None)
    return {
        "order_id": str(_field(line, "order_id")),
        "line_id": str(_field(line, "line_id")),
        "sku": sku,
        "product_id": product_id,
        "size": size,
        "paid_cents": _field(line, "paid_cents"),
        "aliases": list(_field(line, "aliases") or []),
        "delivered_on": _field(line, "delivered_on"),
    }


def _build_account(lines: Sequence[Any], catalog: Sequence[Any]) -> _Account:
    catalog_by_sku = {_field(p, "sku"): p for p in catalog}
    views = [_line_view(line, catalog_by_sku) for line in lines]
    order_ids = _unique(v["order_id"] for v in views)
    line_ids = _unique(v["line_id"] for v in views)

    sku_info: dict[str, tuple[str | None, str | None]] = {}
    for product in catalog:
        sku_info[_field(product, "sku")] = (_field(product, "product_id"), _field(product, "size"))
    for view in views:
        sku_info.setdefault(view["sku"], (view["product_id"], view["size"]))

    known: list[_KnownId] = []
    for kind, values in (("order", order_ids), ("line", line_ids), ("sku", list(sku_info))):
        for value in values:
            pattern = _id_pattern(value) if value else None
            if pattern is not None:
                known.append(_KnownId(kind, value, pattern))
    known.sort(key=lambda k: -len(_norm_id(k.value)))  # longest first: no partial overlaps

    product_words: dict[str, list[str]] = {}
    for product_id in _unique(p for p in [v["product_id"] for v in views] +
                              [info[0] for info in sku_info.values()] if p):
        product_words[product_id] = product_aliases(product_id)
    for view in views:
        if view["product_id"]:
            product_words[view["product_id"]].extend(a.casefold() for a in view["aliases"])
    sizes = {canonical_size(size) for _pid, size in sku_info.values() if size is not None}
    colours = {_field(p, "sku"): str(_field(p, "color") or _field(p, "colour")).casefold()
               for p in catalog if _field(p, "color") or _field(p, "colour")}
    return _Account(
        lines=views,
        order_ids=order_ids,
        known_ids=known,
        sku_info=sku_info,
        phrases=_phrase_table(product_words),
        order_prefixes={_letter_prefix(o) for o in order_ids} - {""},
        line_prefixes={_letter_prefix(i) for i in line_ids} - {""},
        numeric_sizes={s for s in sizes if _NUMBER_RE.match(s)},
        colours=colours,
    )


def _phrase_table(product_words: dict[str, list[str]]) -> list[tuple[re.Pattern[str], tuple[str, ...]]]:
    """One pattern per distinct phrase with the products it names, longest phrase first."""
    owners: dict[str, list[str]] = {}
    spelled: dict[str, list[str]] = {}
    for product_id, phrases in product_words.items():
        for phrase in phrases:
            words = [w for w in re.split(r"[\s\-_]+", phrase.casefold()) if w]
            if not words:
                continue
            key = "".join(words)
            owners.setdefault(key, [])
            if product_id not in owners[key]:
                owners[key].append(product_id)
            if len(words) > len(spelled.get(key, [])):
                spelled[key] = words
    for phrase in _ALL_GARMENT_WORDS:  # garment words with no product on the account still count
        words = re.split(r"[\s\-_]+", phrase)
        key = "".join(words)
        owners.setdefault(key, [])
        if len(words) > len(spelled.get(key, [])):
            spelled[key] = words
    ordered = sorted(owners, key=lambda k: (-len(k), k))
    return [(_phrase_pattern(spelled[key]), tuple(owners[key])) for key in ordered]


# ----------------------------------------------------------------------------------------------
# Reading one message
# ----------------------------------------------------------------------------------------------

@dataclass
class _MessageRefs:
    """Everything one message mentions, already resolved against the account."""

    orders: list[str] = field(default_factory=list)
    unknown_orders: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    unknown_lines: list[str] = field(default_factory=list)
    other_ids: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    unmatched_products: list[str] = field(default_factory=list)
    requested_products: list[str] = field(default_factory=list)
    requested_words: list[str] = field(default_factory=list)  # asked-for garment words owned by no line
    original_sizes: list[str] = field(default_factory=list)
    hard_sizes: bool = False  # an original size describes the item ("my size L tee"): a hard filter
    requested_sizes: list[str] = field(default_factory=list)
    direction: str | None = None
    all_sizes: list[str] = field(default_factory=list)
    remedy: str = NONE
    strong_remedy: str = NONE  # the remedy stated with strong words only (refund / exchange / both)
    remedy_words: list[str] = field(default_factory=list)
    cancel: bool = False
    inquiry: bool = False
    amounts: list[int] = field(default_factory=list)  # money mentioned, in cents
    notes: list[str] = field(default_factory=list)

    @property
    def references(self) -> bool:
        """Does the message point at an item (an ID, a product word or a size)?"""
        return bool(self.orders or self.unknown_orders or self.lines or self.unknown_lines or self.products
                    or self.unmatched_products or self.all_sizes)


@dataclass
class _ProductMention:
    product_ids: tuple[str, ...]  # empty: a garment word for nothing on the account
    surface: str
    size: str | None = None  # set when the mention was a SKU (it names a size too)


@dataclass
class _SizeMention:
    size: str
    index: int  # token index
    trigger: str | None  # "strong" | "weak" | "in" | None (see _trigger_scan)
    before_garment: bool
    describes: bool  # "my large", "the size L tee": a description of the item the customer has


class _MessageReader:
    """Reads one customer message against the account. One instance per message."""

    def __init__(self, account: _Account, scope_orders: list[str], prior_remedy: str) -> None:
        self.account = account
        self.scope_orders = scope_orders  # orders in effect from earlier messages (for ordinals)
        self.prior_remedy = prior_remedy
        self.refs = _MessageRefs()
        self.products: list[_ProductMention] = []

    def read(self, raw: str) -> _MessageRefs:
        text = normalize_text(raw)
        text = self._mask_money(text)
        text = self._mask_known_ids(text)
        text = self._mask_numbered_refs(text)
        text = self._mask_ordinals(text)
        text = self._mask_unknown_ids(text)
        id_masked = text  # identifiers gone, product words still readable
        tokens = _TOKEN_RE.findall(self._mask_products(text))
        sizes = self._find_sizes(tokens)
        self.refs.remedy = self._remedy_cue(id_masked, sizes)
        context = self.refs.remedy if self.refs.remedy != NONE else self.prior_remedy
        self._assign_roles(tokens, sizes, context)
        if context != REFUND:
            self.refs.direction = self._direction(id_masked)
        if context in (EXCHANGE, BOTH) and not self.refs.requested_sizes and not self.refs.direction \
                and _SAME_SIZE_RE.search(id_masked):
            self.refs.requested_sizes = [SAME]
        self.refs.cancel = cancel_cue(id_masked)
        self.refs.inquiry = bool(_HYPOTHETICAL_RE.search(id_masked)) or (
            "?" in id_masked and bool(_RULE_OR_STATUS_RE.search(id_masked)))
        return self.refs

    # -- step 2: money and identifiers -----------------------------------------------------

    def _mask_money(self, text: str) -> str:
        def hit(match: re.Match[str]) -> str:
            self.refs.amounts.append(_money_cents(match))
            return f" {_MONEY_TOKEN} "

        return _MONEY_MENTION_RE.sub(hit, text)

    def _mask_known_ids(self, text: str) -> str:
        for known in self.account.known_ids:
            text = known.pattern.sub(lambda m, k=known: self._known_id_hit(k), text)
        return text

    def _known_id_hit(self, known: _KnownId) -> str:
        if known.kind == "order":
            self.refs.orders = _unique(self.refs.orders + [known.value])
            return f" {_ID_TOKEN} "
        if known.kind == "line":
            self.refs.lines = _unique(self.refs.lines + [known.value])
            return f" {_ID_TOKEN} "
        product_id, size = self.account.sku_info[known.value]
        mention = _ProductMention((product_id,) if product_id else (), known.value,
                                  canonical_size(size) if size else None)
        return self._add_product(mention)

    def _mask_numbered_refs(self, text: str) -> str:
        """'line 2' / 'item #2' / 'order 100' / '#100': match by the ID's trailing number."""
        def line_hit(match: re.Match[str]) -> str:
            number = int(match.group(1))
            hits = _unique(v["line_id"] for v in self.account.lines if _trailing_number(v["line_id"]) == number)
            if hits:
                self.refs.lines = _unique(self.refs.lines + hits)
            else:
                self.refs.unknown_lines = _unique(self.refs.unknown_lines + [f"line {number}"])
            return f" {_ID_TOKEN} "

        def order_hit(match: re.Match[str]) -> str:
            digits = match.group(1) or match.group(2)
            hits = [o for o in self.account.order_ids if _trailing_number(o) == int(digits)]
            if hits:
                self.refs.orders = _unique(self.refs.orders + hits)
            elif len(digits) >= 3:  # "order 2 more" is not an order number
                self.refs.unknown_orders = _unique(self.refs.unknown_orders + [f"#{digits}"])
            else:
                return match.group(0)
            return f" {_ID_TOKEN} "

        return _ORDER_NUMBER_RE.sub(order_hit, _LINE_NUMBER_RE.sub(line_hit, text))

    def _mask_ordinals(self, text: str) -> str:
        """'the second item' = position 2 within the one order in scope (named here, earlier, or only)."""
        def hit(match: re.Match[str]) -> str:
            position = _ORDINALS[match.group(1)]
            scope = self.refs.orders or self.scope_orders
            if not scope and len(self.account.order_ids) == 1:
                scope = self.account.order_ids
            order_lines = self.account.lines_of_order(scope[0]) if len(scope) == 1 else []
            if 0 < position <= len(order_lines):
                self.refs.lines = _unique(self.refs.lines + [order_lines[position - 1]["line_id"]])
                if not self.refs.orders:
                    self.refs.orders = [scope[0]]
            else:
                self.refs.unknown_lines = _unique(self.refs.unknown_lines + [match.group(0)])
            return f" {_ID_TOKEN} "

        return _ORDINAL_RE.sub(hit, text)

    def _mask_unknown_ids(self, text: str) -> str:
        """ID-shaped tokens that are not on the account: classify as order, line or other."""
        pieces: list[str] = []
        cursor = 0
        for match in _ID_SHAPED_RE.finditer(text):
            if match.group(1) in _NOT_ID_PREFIXES:
                continue
            shown = match.group(0).lstrip("#").upper()
            kind = self._classify_unknown(text[: match.start()], match.group(1))
            bucket = {"order": self.refs.unknown_orders, "line": self.refs.unknown_lines,
                      "other": self.refs.other_ids}[kind]
            if shown not in bucket:
                bucket.append(shown)
            pieces.append(text[cursor: match.start()] + f" {_ID_TOKEN} ")
            cursor = match.end()
        return "".join(pieces) + text[cursor:]

    def _classify_unknown(self, before: str, prefix: str) -> str:
        if _ORDER_CONTEXT_RE.search(before):
            return "order"
        if _LINE_CONTEXT_RE.search(before):
            return "line"
        if _OTHER_CONTEXT_RE.search(before):
            return "other"
        if prefix in self.account.order_prefixes:
            return "order"
        if prefix in self.account.line_prefixes:
            return "line"
        if prefix in _RECORD_PREFIXES:
            return "other"
        # "my money back for LN-9 in order ORD-7Q": once the message names one of the customer's
        # orders, a second unrecognized identifier is a line of it (people name the order once).
        if self.refs.orders:
            return "line"
        # An unrecognized identifier is most likely an order number. Treating it as one is the
        # safe choice: it blocks grounding instead of being silently ignored.
        return "order"

    # -- step 3: products ------------------------------------------------------------------

    def _add_product(self, mention: _ProductMention) -> str:
        self.products.append(mention)
        return f" {_product_token(len(self.products) - 1)} "

    def _mask_products(self, text: str) -> str:
        for pattern, product_ids in self.account.phrases:
            text = pattern.sub(lambda m, ids=product_ids: self._add_product(_ProductMention(ids, m.group(0))), text)
        return text

    def _product_at(self, token: str) -> _ProductMention | None:
        if not token.startswith(_PRODUCT_TOKEN):
            return None
        index = int("".join(str(ord(c) - ord("a")) for c in token[len(_PRODUCT_TOKEN):]))
        return self.products[index]

    # -- step 4: sizes and their roles -----------------------------------------------------

    def _find_sizes(self, tokens: list[str]) -> list[_SizeMention]:
        found: list[_SizeMention] = []
        index = 0
        while index < len(tokens):
            size, width = self._size_at(tokens, index)
            if size is None:
                product = self._product_at(tokens[index])
                size, width = (product.size, 1) if product is not None else (None, 1)  # a SKU names its size
            if size is not None:
                after = index + width
                before_garment = after < len(tokens) and self._is_garment(tokens[after])
                before = _word_before(tokens, index, skip={"size", "sz"})
                describes = before in _POSSESSIVES or (before in _DETERMINERS and before_garment)
                found.append(_SizeMention(size, index, self._trigger(tokens, index, after), before_garment, describes))
            index += width
        return found

    def _trigger(self, tokens: list[str], start: int, after: int) -> str | None:
        """What introduces a size: the words before it, or an offer frame right after it.

        "size S instead", "an M rather than that" and "is an L available?" all ask for the size named.
        """
        following = tokens[after: after + 3]
        if any(tuple(following[: len(frame)]) == frame for frame in _OFFER_AFTER):
            return "strong"
        return _trigger_scan(tokens, start, self._is_garment)[0]

    def _size_at(self, tokens: list[str], index: int) -> tuple[str | None, int]:
        """(canonical size, token width) if a size mention starts at tokens[index]."""
        for words, size in _MULTIWORD_SIZES:
            if tuple(tokens[index: index + len(words)]) == words:
                return size, len(words)
        token = tokens[index]
        previous = tokens[index - 1] if index > 0 else ""
        word = _size_word(token)
        if word is not None:
            if token in _FIT_SIZE_WORDS and (previous in _DEGREE_WORDS or previous in _FIT_VERBS):
                return None, 1
            return word, 1
        if token in _LETTER_SIZES and self._letter_is_size(tokens, index):
            return _LETTER_SIZES[token], 1
        if _NUMBER_RE.match(token):
            if _word_before(tokens, index) in {"size", "sz", "waist"}:
                return canonical_size(token), 1
            # "for a 32" asks for size 32 only when the catalog offers a size 32 (it is not a count).
            if canonical_size(token) in self.account.numeric_sizes and _word_before(
                    tokens, index, skip={"a", "an", "the"}) in _STRONG_TRIGGERS:
                return canonical_size(token), 1
        return None, 1

    def _letter_is_size(self, tokens: list[str], index: int) -> bool:
        """Bare s/m/l only in the SPEC patterns: 'size M', 'in (a) M', 'an M', 'M <product>', 'for (an) L'."""
        if _word_before(tokens, index) in {"size", "sz"}:
            return True
        if index + 1 < len(tokens) and self._is_garment(tokens[index + 1]):
            return True
        before = index - 1
        if before >= 0 and tokens[before] == "an":
            return True
        if before >= 0 and tokens[before] in {"a", "the"}:
            before -= 1
        return before >= 0 and tokens[before] in _STRONG_TRIGGERS | _WEAK_TRIGGERS | {"in"}

    def _is_garment(self, token: str) -> bool:
        return token.startswith(_PRODUCT_TOKEN)

    def _bare_request(self, tokens: list[str], index: int) -> bool:
        """In an exchange conversation, a size standing alone ('Large.', 'size L please') is the one wanted.

        It describes the item instead when its clause names a product ('the size M tee') or a
        determiner introduces it ('my medium', 'the size M one').
        """
        start, end = index, index + 1
        while start > 0 and tokens[start - 1] not in _CLAUSE_BREAKS:
            start -= 1
        while end < len(tokens) and tokens[end] not in _CLAUSE_BREAKS:
            end += 1
        if any(self._is_garment(t) for t in tokens[start:end]):
            return False
        return _word_before(tokens, index, skip={"size", "sz"}) not in _DETERMINERS

    def _assign_roles(self, tokens: list[str], sizes: list[_SizeMention], context: str) -> None:
        """Split mentions into 'the item I have' (original) and 'the one I want' (requested)."""
        self.refs.all_sizes = _unique(s.size for s in sizes)
        if context == REFUND:
            requested: list[_SizeMention] = []
        else:  # a for/to/into size outranks a weaker 'in a large' / 'I need a large' / bare 'Large'
            weak = [s for s in sizes if s.trigger in ("weak", "in")
                    or (s.trigger is None and context in (EXCHANGE, BOTH) and self._bare_request(tokens, s.index))]
            requested = [s for s in sizes if s.trigger == "strong"] or weak
        self.refs.requested_sizes = _unique(s.size for s in requested)
        originals = [s for s in sizes if s not in requested]
        self.refs.original_sizes = _unique(s.size for s in originals)
        self.refs.hard_sizes = any(s.describes for s in originals)

        for index, token in enumerate(tokens):
            product = self._product_at(token)
            if product is None:
                continue
            trigger, says_same = (None, False) if context == REFUND else _trigger_scan(tokens, index, self._is_garment)
            if not product.product_ids:
                self.refs.unmatched_products = _unique(self.refs.unmatched_products + [product.surface])
                if trigger is not None:  # "for the jacket" when no jacket is owned: still the product asked for
                    self.refs.requested_words = _unique(self.refs.requested_words + [product.surface])
                continue
            if trigger is not None:
                self.refs.requested_products = _unique(self.refs.requested_products + list(product.product_ids))
            if trigger is None or says_same:
                self.refs.products = _unique(self.refs.products + list(product.product_ids))

    # -- step 5: cues ----------------------------------------------------------------------

    def _remedy_cue(self, text: str, sizes: list[_SizeMention]) -> str:
        strong_exchange = _live_cues(_STRONG_EXCHANGE_RE, text)
        weak_exchange = _live_cues(_WEAK_EXCHANGE_RE, text)
        # "for a large" / "and get a large" ask for a size; "refund for the medium tee" only describes it,
        # and a bare "in a small" ("I may buy one in a small some day") is too weak to mean an exchange.
        if any(s.trigger in ("strong", "weak") and not s.before_garment for s in sizes):
            weak_exchange.append("<size> wanted")
        strong_refund = _live_cues(_STRONG_REFUND_RE, text)
        weak_refund = _live_cues(_WEAK_REFUND_RE, text)
        self.refs.remedy_words = strong_exchange + weak_exchange + strong_refund + weak_refund
        self.refs.strong_remedy = (BOTH if strong_exchange and strong_refund else EXCHANGE if strong_exchange
                                   else REFUND if strong_refund else NONE)
        any_exchange, any_refund = strong_exchange or weak_exchange, strong_refund or weak_refund
        if (strong_exchange and strong_refund) or (any_exchange and any_refund and _HEDGE_RE.search(text)):
            return BOTH  # CLARIFICATIONS #4: both remedies, and the customer has not chosen
        if strong_exchange:
            return EXCHANGE
        if strong_refund:
            return REFUND
        if weak_exchange:  # "return it for a bigger one" is an exchange
            return EXCHANGE
        return REFUND if weak_refund else NONE

    def _direction(self, text: str) -> str | None:
        up, down = _live_cues(_UP_RE, text), _live_cues(_DOWN_RE, text)
        if up and down:
            self.refs.notes.append("conflicting size directions ignored")
            return None
        return UP if up else DOWN if down else None


def cancel_cue(text: str) -> bool:
    """Does this (normalized) message cancel? Per clause and negation-aware (CLARIFICATIONS #9)."""
    for match in _CANCEL_VERB_RE.finditer(text):
        clause = re.split(r"[.,;:!?]", text[: match.start()])[-1]
        if not _CANCEL_NEGATED_BEFORE.search(clause):
            return True
    opening = _HESITATION_START_RE.search(text)
    # "Hmm, wait... yes, go ahead." hesitates, then consents: not a cancel.
    return bool(opening) and not _CONSENT_RE.search(text[opening.end():])


def _live_cues(pattern: re.Pattern[str], text: str) -> list[str]:
    """Matches of a cue pattern that are not negated within their clause."""
    live = []
    for match in pattern.finditer(text):
        clause = re.split(r"[.,;:!?]", text[: match.start()])[-1]
        if not _NEGATED_BEFORE.search(clause):
            live.append(match.group(0))
    return live


def _word_before(tokens: list[str], index: int, skip: set[str] = frozenset()) -> str:
    """The previous token, skipping a ':' ('size: M') and any words in `skip`."""
    before = index - 1
    while before >= 0 and (tokens[before] == ":" or tokens[before] in skip):
        before -= 1
    return tokens[before] if before >= 0 else ""


def _trigger_scan(tokens: list[str], index: int, is_garment: Any) -> tuple[str | None, bool]:
    """Walk back from a mention over filler words to find what introduces it.

    Returns (trigger strength, whether 'same' was passed). 'for/to/into' and 'what/how about' are
    strong; 'need/want/get/...', 'send me', 'go with' and a request-frame 'have' ("can I have", "do
    you have") are weak; a bare 'in' gives "in" (it asks for a size in an exchange conversation but
    is no remedy cue), unless it follows a product word ("the tee in size M" describes the item).
    "a pair of" is skipped. Stops at a clause break or any other word.
    """
    weak = False
    says_same = False
    cursor = index - 1
    while cursor >= 0:
        token = tokens[cursor]
        previous = tokens[cursor - 1] if cursor > 0 else ""
        if token in _CLAUSE_BREAKS:
            break
        if token in _STRONG_TRIGGERS or (token == "about" and previous in _ASKING_ABOUT):
            return "strong", says_same
        if token in _WEAK_TRIGGERS or (token == "me" and previous in _SEND_VERBS):
            return "weak", says_same
        if token == "in":
            if not (cursor > 0 and is_garment(previous)):
                weak = True
        elif token == "with" and previous in _CHOICE_VERBS_BEFORE_WITH:
            return "weak", says_same
        elif token == "have" and tuple(tokens[max(0, cursor - 2): cursor]) in _REQUEST_BEFORE_HAVE:
            return "weak", says_same
        elif token == "of" and previous in {"pair", "pairs"}:
            pass  # "for a pair of boots" reads like "for boots"
        elif token == "same":
            says_same = True
        elif not (token in _FILLERS or is_garment(token) or _size_word(token)
                  or token in _LETTER_SIZES or _NUMBER_RE.match(token)):
            break
        cursor -= 1
    return ("in" if weak else None), says_same


# ----------------------------------------------------------------------------------------------
# Combining messages
# ----------------------------------------------------------------------------------------------

@dataclass
class _Conversation:
    """References in effect after reading messages in order (latest mention of each kind wins)."""

    orders: list[str] = field(default_factory=list)
    unknown_orders: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    unknown_lines: list[str] = field(default_factory=list)
    other_ids: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    unmatched_products: list[str] = field(default_factory=list)
    requested_products: list[str] = field(default_factory=list)
    requested_words: list[str] = field(default_factory=list)
    original_sizes: list[str] = field(default_factory=list)
    hard_sizes: bool = False
    requested: list[str] = field(default_factory=list)  # explicit sizes, or [UP] / [DOWN] / [SAME]
    mentioned_sizes: list[str] = field(default_factory=list)
    remedy: str = NONE
    strong_remedy: str = NONE  # strong wording behind `remedy` (NONE when it came from weak words)
    stated_remedy: str = NONE  # what the latest message that talks about the remedy states strongly
    remedy_words: list[str] = field(default_factory=list)

    def absorb(self, refs: _MessageRefs) -> None:
        if refs.orders or refs.unknown_orders:
            self.orders, self.unknown_orders = refs.orders, refs.unknown_orders
        if refs.lines or refs.unknown_lines:
            self.lines, self.unknown_lines = refs.lines, refs.unknown_lines
        if refs.products or refs.unmatched_products:
            self.products, self.unmatched_products = refs.products, refs.unmatched_products
        if refs.requested_products or refs.requested_words:
            self.requested_products, self.requested_words = refs.requested_products, refs.requested_words
        if refs.original_sizes:
            self.original_sizes, self.hard_sizes = refs.original_sizes, refs.hard_sizes
        if refs.requested_sizes:
            self.requested = refs.requested_sizes
        elif refs.direction:
            self.requested = [refs.direction]
        if refs.remedy != NONE:
            self.stated_remedy = BOTH if refs.remedy == BOTH else refs.strong_remedy
            # "I'll post the old one back" (weak refund words) must not turn an earlier "exchange" into
            # a refund: sending the item back is part of every exchange (review NEW-2).
            weak_refund = refs.remedy == REFUND and refs.strong_remedy == NONE
            if not (weak_refund and self.strong_remedy == EXCHANGE):
                self.remedy, self.remedy_words = refs.remedy, refs.remedy_words
                self.strong_remedy = refs.strong_remedy if refs.strong_remedy != NONE else self.strong_remedy
        self.other_ids = _unique(self.other_ids + refs.other_ids)
        self.mentioned_sizes = _unique(self.mentioned_sizes + refs.all_sizes)


@dataclass
class Analysis:
    """What the conversation grounds. Built only by analyze(); read by the orchestrator and gates."""

    order_refs: list[str] = field(default_factory=list)
    unknown_order_refs: list[str] = field(default_factory=list)
    line_refs: list[str] = field(default_factory=list)
    unknown_line_refs: list[str] = field(default_factory=list)
    other_id_refs: list[str] = field(default_factory=list)  # e.g. a proposal id; never blocks
    product_refs: list[str] = field(default_factory=list)  # the item the customer has
    unmatched_product_words: list[str] = field(default_factory=list)  # garment words for nothing owned
    requested_products: list[str] = field(default_factory=list)  # "for a jacket"
    requested_product_words: list[str] = field(default_factory=list)  # "for the jacket", no jacket owned
    requested_size: str | None = None  # canonical size, UP, DOWN or SAME; None if absent or ambiguous
    requested_sizes: list[str] = field(default_factory=list)
    original_size_hints: list[str] = field(default_factory=list)
    hard_size_hints: bool = False  # the hints describe the item ("my size L tee"), so they filter hard
    mentioned_sizes: list[str] = field(default_factory=list)
    remedy_cue: str = NONE
    stated_remedy: str = NONE  # strong wording of the latest message that names a remedy (for the gate)
    remedy_words: list[str] = field(default_factory=list)
    cancel_cue: bool = False  # in the LAST message only
    last_remedy_cue: str = NONE  # LAST message only: lets a confirm turn spot "yes, but refund it"
    last_strong_remedy: str = NONE  # LAST message only, strong words only
    last_requested_size: str | None = None
    last_sizes: list[str] = field(default_factory=list)  # every size the LAST message mentions
    last_references: bool = False  # the LAST message names an ID, a product or a size
    inquiry: bool = False  # the LAST message asks about rules or status, not for a remedy
    candidates: list[dict[str, Any]] = field(default_factory=list)
    unique_target: tuple[str, str] | None = None
    not_found_code: str | None = None
    filters_used: list[str] = field(default_factory=list)
    explain: list[str] = field(default_factory=list)
    account_keys: list[tuple[str, str]] = field(default_factory=list, repr=False)
    id_candidates: list[dict[str, Any]] = field(default_factory=list, repr=False)  # lines the typed IDs allow
    texts: list[str] = field(default_factory=list, repr=False)  # normalized customer messages, for citations
    as_of: str | None = field(default=None, repr=False)
    account: _Account | None = field(default=None, repr=False)

    @property
    def candidate_keys(self) -> list[tuple[str, str]]:
        return [(c["order_id"], c["line_id"]) for c in self.candidates]

    @property
    def id_candidate_keys(self) -> list[tuple[str, str]]:
        return [(c["order_id"], c["line_id"]) for c in self.id_candidates]

    @property
    def requested_product(self) -> str | None:
        return self.requested_products[0] if len(self.requested_products) == 1 else None

    def find_line(self, order_id: Any, line_id: Any) -> tuple[str, str] | None:
        """The account's (order_id, line_id) for loosely typed IDs ('o-100', 'l1'), else None."""
        wanted = (_norm_id(order_id), _norm_id(line_id))
        return next((k for k in self.account_keys if (_norm_id(k[0]), _norm_id(k[1])) == wanted), None)

    def has_order(self, order_id: Any) -> bool:
        return any(_norm_id(k[0]) == _norm_id(order_id) for k in self.account_keys)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary for the audit log (candidates reduced to their keys)."""
        return {
            "order_refs": self.order_refs, "unknown_order_refs": self.unknown_order_refs,
            "line_refs": self.line_refs, "unknown_line_refs": self.unknown_line_refs,
            "other_id_refs": self.other_id_refs, "product_refs": self.product_refs,
            "unmatched_product_words": self.unmatched_product_words,
            "requested_products": self.requested_products, "requested_size": self.requested_size,
            "requested_sizes": self.requested_sizes, "original_size_hints": self.original_size_hints,
            "mentioned_sizes": self.mentioned_sizes, "remedy_cue": self.remedy_cue,
            "stated_remedy": self.stated_remedy, "remedy_words": self.remedy_words, "cancel_cue": self.cancel_cue,
            "last_remedy_cue": self.last_remedy_cue, "last_requested_size": self.last_requested_size,
            "inquiry": self.inquiry,
            "candidates": [f"{o}/{line}" for o, line in self.candidate_keys],
            "id_candidates": [f"{o}/{line}" for o, line in self.id_candidate_keys],
            "unique_target": list(self.unique_target) if self.unique_target else None,
            "not_found_code": self.not_found_code, "filters_used": self.filters_used,
            "explain": self.explain,
        }


def analyze(messages: Sequence[str], lines: Sequence[Any], catalog: Sequence[Any],
            as_of: str | None = None) -> Analysis:
    """Ground every customer message of the conversation so far against the customer's own lines.

    `lines` must be the authenticated customer's lines only: other customers' orders are then
    "unknown", which yields the same ORDER_NOT_FOUND as an order that does not exist. `as_of`
    (the business date) lets a citation like "the one from last month" be checked.
    """
    account = _build_account(lines, catalog)
    conversation = _Conversation()
    explain: list[str] = []
    last = _MessageRefs()
    for number, raw in enumerate(messages, start=1):
        reader = _MessageReader(account, scope_orders=conversation.orders, prior_remedy=conversation.remedy)
        last = reader.read(raw)
        conversation.absorb(last)
        explain.extend(_describe_message(number, last))

    candidates, filters_used, filter_notes = _filter_candidates(account, conversation)
    not_found = _not_found_code(account, conversation)
    unique = _unique_target(candidates, not_found)
    explain.extend(filter_notes)
    explain.append(_describe_outcome(candidates, unique, not_found, conversation))

    requested = conversation.requested
    return Analysis(
        order_refs=conversation.orders,
        unknown_order_refs=conversation.unknown_orders,
        line_refs=conversation.lines,
        unknown_line_refs=conversation.unknown_lines,
        other_id_refs=conversation.other_ids,
        product_refs=conversation.products,
        unmatched_product_words=conversation.unmatched_products,
        requested_products=conversation.requested_products,
        requested_product_words=conversation.requested_words,
        requested_size=requested[0] if len(requested) == 1 else None,
        requested_sizes=requested,
        original_size_hints=conversation.original_sizes,
        hard_size_hints=conversation.hard_sizes,
        mentioned_sizes=conversation.mentioned_sizes,
        remedy_cue=conversation.remedy,
        stated_remedy=conversation.stated_remedy,
        remedy_words=conversation.remedy_words,
        cancel_cue=last.cancel,
        last_remedy_cue=last.remedy,
        last_strong_remedy=last.strong_remedy,
        last_requested_size=(last.requested_sizes[0] if len(last.requested_sizes) == 1
                             else None if last.requested_sizes else last.direction),
        last_sizes=last.all_sizes,
        last_references=last.references,
        inquiry=last.inquiry,
        candidates=candidates,
        unique_target=unique,
        not_found_code=not_found,
        filters_used=filters_used,
        explain=explain,
        account_keys=[(v["order_id"], v["line_id"]) for v in account.lines],
        id_candidates=_id_candidates(account, conversation),
        texts=[_quote_form(m) for m in messages],
        as_of=as_of,
        account=account,
    )


def _id_candidates(account: _Account, conv: _Conversation) -> list[dict[str, Any]]:
    """The lines the IDs the customer typed allow (all lines when no ID was typed). Facts only."""
    return [line for line in account.lines
            if (not conv.orders or line["order_id"] in conv.orders) and (not conv.lines or line["line_id"] in conv.lines)]


def _filter_candidates(account: _Account, conv: _Conversation) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Narrow the customer's lines by order, then line, then product, then original size.

    A filter runs only when the text references something of its kind. A filter that removes
    every line is kept when it is a hard fact or a description of the item ("my size L tee"): an
    item the customer does not own must not be guessed. A loose size mention that would remove
    every line ("is the large back?") is ignored and explained instead.
    """
    candidates = list(account.lines)
    used: list[str] = []
    notes: list[str] = []
    # A named line ID is stronger evidence than a description: "refund LN-3, still sealed in the bag"
    # must not lose LN-3 because "bag" matches nothing owned. So once a line is named, a product or
    # size filter that would remove every remaining candidate is ignored (and explained).
    soft = bool(conv.lines)

    def apply(name: str, label: str, keep: Any, soft: bool = False) -> None:
        nonlocal candidates
        kept = [line for line in candidates if keep(line)]
        if soft and not kept:
            notes.append(f"filter {name} {label} ignored: it matches none of the remaining line(s)")
            return
        candidates = kept
        used.append(name)
        notes.append(f"filter {name} {label}: {len(candidates)} candidate(s)")

    if conv.orders:
        apply("order", "/".join(conv.orders), lambda line: line["order_id"] in conv.orders)
    if conv.lines:
        apply("line", "/".join(conv.lines), lambda line: line["line_id"] in conv.lines)
    products = conv.products
    if not products and not conv.unmatched_products and conv.requested_products:
        # "exchange it for a large tee": an exchange stays within one product, so the item is a tee
        products = conv.requested_products
        notes.append("no original product named; using the requested product")
    if products or conv.unmatched_products:
        label = "/".join(products + [repr(w) for w in conv.unmatched_products])
        apply("product", label, lambda line: line["product_id"] in products, soft)
    if conv.original_sizes:
        apply("size", "/".join(conv.original_sizes),
              lambda line: line["size"] is not None and canonical_size(line["size"]) in conv.original_sizes,
              soft or not conv.hard_sizes)
    return candidates, used, notes


def _not_found_code(account: _Account, conv: _Conversation) -> str | None:
    if conv.unknown_orders:
        return ORDER_NOT_FOUND
    if conv.unknown_lines:
        return LINE_NOT_FOUND
    if conv.lines:
        in_scope = [line for line in account.lines
                    if line["line_id"] in conv.lines and (not conv.orders or line["order_id"] in conv.orders)]
        if not in_scope:  # e.g. "LN-5 of ORD-7Q" where LN-5 belongs to another of the customer's orders
            return LINE_NOT_FOUND
    return None


def _unique_target(candidates: list[dict[str, Any]], not_found: str | None) -> tuple[str, str] | None:
    """Exactly one candidate left and nothing named that does not exist.

    This covers both SPEC §5 cases. Either at least one filter narrowed the lines to one, or no
    filter ran at all, which leaves exactly one candidate only when the account has a single line
    and the text names no order or line.
    """
    if not_found or len(candidates) != 1:
        return None
    return candidates[0]["order_id"], candidates[0]["line_id"]


def _describe_message(number: int, refs: _MessageRefs) -> list[str]:
    parts = []
    for label, values in (
        ("orders", refs.orders), ("unknown orders", refs.unknown_orders), ("lines", refs.lines),
        ("unknown lines", refs.unknown_lines), ("other ids", refs.other_ids), ("products", refs.products),
        ("unmatched product words", refs.unmatched_products), ("requested products", refs.requested_products),
        ("original sizes", refs.original_sizes), ("requested sizes", refs.requested_sizes),
    ):
        if values:
            parts.append(f"{label} {', '.join(values)}")
    if refs.direction:
        parts.append(f"direction {refs.direction}")
    if refs.remedy != NONE:
        parts.append(f"remedy {refs.remedy} ({', '.join(refs.remedy_words)})")
    if refs.cancel:
        parts.append("cancel cue")
    if refs.inquiry:
        parts.append("question about rules or status")
    parts.extend(refs.notes)
    return [f"message {number}: " + ("; ".join(parts) if parts else "no references")]


def _describe_outcome(candidates: list[dict[str, Any]], unique: tuple[str, str] | None,
                      not_found: str | None, conv: _Conversation) -> str:
    if unique:
        return f"unique target {unique[0]}/{unique[1]}"
    if not_found:
        named = conv.unknown_orders if not_found == ORDER_NOT_FOUND else conv.unknown_lines or conv.lines
        return f"no unique target: {not_found} ({', '.join(named)})"
    listed = ", ".join(f"{c['order_id']}/{c['line_id']}" for c in candidates) or "none"
    return f"no unique target: {len(candidates)} candidate(s) ({listed})"


# ----------------------------------------------------------------------------------------------
# Checks over the agent's handoff: facts can block, they never choose
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetCheck:
    """The verdict on the agent's (order, line): accepted with how it was grounded, or a code."""

    ok: bool
    code: str | None = None
    key: tuple[str, str] | None = None
    grounded_by: str | None = None
    evidence: str | None = None
    reason: str | None = None  # why a citation was not accepted (agent feedback)


def check_target(analysis: Analysis, order_id: Any, line_id: Any, evidence: Any = None) -> TargetCheck:
    """Is the agent's chosen line consistent with the facts, and does someone's reading pick it?

    Hard facts: the line is on the account; every order/line ID the customer typed agrees with it.
    Then: when the typed IDs (or the account) leave one line, the agent's target is accepted.
    With 2+ lines left the agent must cite the customer's words that pick its line (evidence);
    without a citation it is accepted only if code's own reading picks the same line.
    """
    key = analysis.find_line(order_id, line_id)
    if key is None:
        return TargetCheck(False, LINE_NOT_FOUND if analysis.has_order(order_id) else ORDER_NOT_FOUND)
    if analysis.not_found_code:
        return TargetCheck(False, analysis.not_found_code)
    if (analysis.order_refs and key[0] not in analysis.order_refs) or (
            analysis.line_refs and key[1] not in analysis.line_refs):
        return TargetCheck(False, TARGET_MISMATCH, reason="it contradicts the order or line id the customer typed")
    plausible = analysis.id_candidate_keys
    if len(plausible) == 1:
        if analysis.order_refs or analysis.line_refs:
            return TargetCheck(True, key=key, grounded_by=BY_CUSTOMER_IDS)
        contradiction = _description_contradiction(analysis, key)
        if contradiction:
            return TargetCheck(False, TARGET_MISMATCH, reason=contradiction)
        return TargetCheck(True, key=key, grounded_by=BY_ONLY_LINE)
    quote = str(evidence).strip() if evidence else ""
    reason = "the handoff cites no customer words that pick this line"
    if quote:
        reason = check_evidence(analysis, key, quote)
        if reason is None:
            return TargetCheck(True, key=key, grounded_by=BY_AGENT, evidence=quote)
    if analysis.unique_target == key:
        return TargetCheck(True, key=key, grounded_by=BY_AGENT_AND_GROUNDING, evidence=quote or None)
    return TargetCheck(False, AMBIGUOUS_TARGET, reason=reason)


def _description_contradiction(analysis: Analysis, key: tuple[str, str]) -> str | None:
    """On a one-line account, a possessive size or a product the customer does not own is a fact
    about the item they mean ("my size L tee" when they own an M tee, "the hoodie" when they own
    only a tee, even if the catalog sells hoodies): do not accept a guess."""
    line = next(v for v in analysis.id_candidates if (v["order_id"], v["line_id"]) == key)
    if analysis.hard_size_hints and line["size"] is not None and \
            canonical_size(line["size"]) not in analysis.original_size_hints:
        return f"the customer described their item as size {'/'.join(analysis.original_size_hints)}"
    if analysis.product_refs and line["product_id"] not in analysis.product_refs:
        return f"the customer described a {'/'.join(analysis.product_refs)}, which is not this line's product"
    if analysis.unmatched_product_words and not analysis.product_refs:
        return f"the customer described a {'/'.join(analysis.unmatched_product_words)}, which is not on the account"
    return None


def check_evidence(analysis: Analysis, key: tuple[str, str], evidence: str) -> str | None:
    """Verify the agent's citation for picking `key` among the lines the typed IDs allow.

    Accepted iff the words are a verbatim part of the customer's messages, not inside a hedge
    ("the scarf or the cap", "not sure which"), and every attribute they name (product word,
    size, SKU, price, colour, delivery month/date or recency, order or line id) fits the chosen
    line while together they rule out every other candidate. Returns None when accepted, else the
    reason (fed back to the agent).
    """
    quote = _quote_form(evidence)
    if len(quote) < 2:
        return "the evidence is empty"
    sentences = [_sentence_around(text, m.start(), m.end()) for text in analysis.texts
                 for m in re.finditer(re.escape(quote), text)]
    if not sentences:
        return NOT_A_QUOTE
    if all(_HEDGE_RE.search(sentence) for sentence in sentences):
        return "the customer's words hedge between items ('or', 'not sure', 'which one')"
    candidates = analysis.id_candidates
    matches = _attribute_matches(analysis, quote, candidates)
    if not matches:
        return "the evidence names nothing that tells the lines apart"
    chosen = next(c for c in candidates if (c["order_id"], c["line_id"]) == key)
    for kind, fitting in matches.items():
        if not any(c is chosen for c in fitting):
            return f"the evidence describes another line ({kind})"
    survivors = [c for c in candidates if all(any(c is f for f in fitting) for fitting in matches.values())]
    if len(survivors) != 1:
        return "the evidence fits more than one line"
    return None


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence(s) of `text` that contain text[start:end]."""
    left = max(text.rfind(mark, 0, start) for mark in ".!?;")
    rights = [i for i in (text.find(mark, end) for mark in ".!?;") if i >= 0]
    return text[left + 1: min(rights) if rights else len(text)]


def _quote_form(text: Any) -> str:
    """Text as compared for citations: sanitized, casefolded, straight quotes, single spaces."""
    flat = " ".join(sanitize(str(text)).translate(_APOSTROPHES).casefold().split())
    return flat.strip(" \"'“”‘’.,;:!?")


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")
_NEWEST_RE = re.compile(r"\b(?:latest|newest|most recent|recent|last order|last one|newer)\b")
_OLDEST_RE = re.compile(r"\b(?:oldest|first order|first one|earlier|earliest|older)\b")


def _attribute_matches(analysis: Analysis, quote: str, candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """For each attribute kind the quote mentions, the candidate lines that fit it."""
    assert analysis.account is not None
    refs = _MessageReader(analysis.account, scope_orders=[], prior_remedy=NONE).read(quote)
    found: dict[str, list[dict[str, Any]]] = {}
    if refs.orders:
        found["order"] = [c for c in candidates if c["order_id"] in refs.orders]
    if refs.lines:
        found["line"] = [c for c in candidates if c["line_id"] in refs.lines]
    products = refs.products or refs.requested_products  # "for the cardigan" names the item too
    if products or refs.unmatched_products:
        found["product"] = [c for c in candidates if c["product_id"] in products]
    if refs.original_sizes:
        found["size"] = [c for c in candidates if c["size"] is not None
                         and canonical_size(c["size"]) in refs.original_sizes]
    if refs.amounts:
        found["price"] = [c for c in candidates if c["paid_cents"] in refs.amounts]
    colours = {analysis.account.colours.get(c["sku"]) for c in candidates} - {None}
    named_colours = {colour for colour in colours if re.search(rf"\b{re.escape(colour)}\b", quote)}
    if named_colours:
        found["colour"] = [c for c in candidates if analysis.account.colours.get(c["sku"]) in named_colours]
    dated = _date_matches(analysis.as_of, quote, candidates)
    if dated is not None:
        found["delivery"] = dated
    return found


def _date_matches(as_of: str | None, quote: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Lines whose delivery fits a month name, an ISO date, 'last/this month' or 'latest/oldest'."""
    days = {id(c): parse_date(c.get("delivered_on")) for c in candidates}
    known = [c for c in candidates if days[id(c)] is not None]
    if not known:
        return None
    if _NEWEST_RE.search(quote) or _OLDEST_RE.search(quote):
        pick = max if _NEWEST_RE.search(quote) else min
        edge = pick(days[id(c)] for c in known)
        return [c for c in known if days[id(c)] == edge]
    today = parse_date(as_of)
    wanted: set[tuple[int, int]] = set()
    if today is not None and re.search(r"\blast month\b", quote):
        wanted.add((today.year - (today.month == 1), (today.month - 2) % 12 + 1))
    if today is not None and re.search(r"\bthis month\b", quote):
        wanted.add((today.year, today.month))
    months = {i + 1 for i, name in enumerate(_MONTHS) if re.search(_month_pattern(name), quote)}
    iso = {d for d in (parse_date(m) for m in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", quote)) if d is not None}
    if not (wanted or months or iso):
        return None
    return [c for c in known if (days[id(c)].year, days[id(c)].month) in wanted or days[id(c)].month in months
            or days[id(c)] in iso]


def _month_pattern(name: str) -> str:
    # "may" is also a verb ("may I return it"): only a preposition before it makes it the month.
    if name == "may":
        return r"\b(?:in|on|from|since|during|early|mid|late) may\b"
    return rf"\b(?:{name}|{name[:3]})\b"


def check_remedy(analysis: Analysis, remedy: str) -> tuple[bool, str | None]:
    """Block the agent's remedy only on STRONG contradicting wording (refund / money back /
    reimburse vs exchange / swap / different size) in the latest message that talks about the
    remedy, or when that message names both remedies undecided. Weak words never flip a remedy."""
    if analysis.stated_remedy == BOTH:
        return False, REMEDY_UNCLEAR
    if analysis.stated_remedy == NONE:
        return True, None
    wanted = EXCHANGE if EXCHANGE in str(remedy).casefold() else REFUND if REFUND in str(remedy).casefold() else None
    return (True, None) if wanted == analysis.stated_remedy else (False, REMEDY_MISMATCH)


def check_requested_size(analysis: Analysis, requested: Any, original_sku: str,
                         catalog: Sequence[Any], requested_product: Any = None) -> tuple[bool, str | None]:
    """Accept the agent's requested size unless the customer's own size words contradict it.

    The agent reads sizes code cannot (numerics like "32s", "talla L", "mediun"). Its size is
    blocked when the customer literally asked for a different size (or for two sizes), and when
    the customer wrote no size at all that it could come from: a size nobody typed is a guess,
    and "UNSPECIFIED" lets Operations answer with the possible sizes instead (CLARIFICATIONS #5).
    """
    if _asks_for_named_other_product(analysis, requested_product, original_sku, catalog):
        return True, None  # another product is refused whatever the size (DIFFERENT_PRODUCT)
    asked_sku, asked_code = resolve_size(_arg(requested), original_sku, list(catalog))
    said = analysis.requested_sizes
    if len(said) > 1:  # "for an L or an XL": the customer must choose
        return False, SIZE_UNSPECIFIED
    if said:
        said_sku, _said_code = resolve_size(said[0], original_sku, list(catalog))
        if said_sku is not None:
            return (True, None) if asked_sku == said_sku else (False, REPLACEMENT_NOT_REQUESTED)
        # The customer's size is not offered: any reading that is not offered either ends in the
        # same refusal, so it may go on to Operations; an offered size would be a different request.
        both_unavailable = asked_sku is None and asked_code is not None
        return (True, None) if both_unavailable else (False, REPLACEMENT_NOT_REQUESTED)
    # The customer named no size: a concrete size must come from their words. Their own item's size
    # is never a request here ("the same size" would have been read above).
    if asked_sku is not None and (asked_sku == original_sku or not _size_written(analysis, asked_sku, catalog)):
        return False, SIZE_UNSPECIFIED
    if asked_code == SIZE_NOT_OFFERED and not _typed_by_customer(analysis, _arg(requested)):
        return False, SIZE_UNSPECIFIED
    return True, None


def newer_size_in_latest_words(last: Analysis, requested: Any, original_sku: str, catalog: Sequence[Any]) -> bool:
    """The latest message writes a size code reads no request from ("Ugh. What about a 30 then?" after
    the 34 ran out) and the agent proposes exactly that size. Newer words beat an older request, so
    the size read from earlier messages does not block it; the size must still be written in the
    latest message, never the agent's guess, and never the customer's own item's size."""
    if last.requested_sizes:
        return False
    asked_sku, _code = resolve_size(_arg(requested), original_sku, list(catalog))
    return asked_sku is not None and asked_sku != original_sku and _size_written(last, asked_sku, catalog)


def changes_the_open_offer(last: Analysis, requested: Any, original_sku: str, offered_sku: str,
                           catalog: Sequence[Any]) -> bool:
    """The customer's latest message moves an open exchange offer to another size written there
    ("yes, but make it an 11 instead, the 10 might still be tight"). The offered size is what they
    are changing away from, so it does not compete with the new size; the new size must still be
    the customer's own words (in the latest message), never the agent's guess."""
    asked_sku, _code = resolve_size(_arg(requested), original_sku, list(catalog))
    return asked_sku is not None and asked_sku not in (offered_sku, original_sku) and _size_written(last, asked_sku, catalog)


def _arg(value: Any) -> str:
    return "" if value is None else str(value)


def _asks_for_named_other_product(analysis: Analysis, requested_product: Any, original_sku: str,
                                  catalog: Sequence[Any]) -> bool:
    """The agent says the customer wants another catalog product, and the customer did ask for that
    product ("for a pair of jeans in size 32" on a hoodie). Then no size can make it a valid exchange."""
    wanted = _arg(requested_product).strip().casefold()
    named = next((p for p in catalog if wanted in (str(_field(p, "product_id")).casefold(),
                                                   str(_field(p, "sku")).casefold())), None)
    original = next((p for p in catalog if _field(p, "sku") == original_sku), None)
    return (named is not None and original is not None
            and _field(named, "product_id") != _field(original, "product_id")
            and _field(named, "product_id") in analysis.requested_products)


def _typed_by_customer(analysis: Analysis, word: str) -> bool:
    """Did the customer type this size word themselves (whole word)?"""
    token = _quote_form(word)
    return bool(token) and any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", t) for t in analysis.texts)


_SIZE_TYPO_MIN = 5  # one-typo matching only for long size words ("mediun"); "mall" must not read as small


def _size_written(analysis: Analysis, sku: str, catalog: Sequence[Any]) -> bool:
    """Did the customer write the size of catalog SKU `sku` in some form? A label or size word
    ("L", "large"), a size word with one typo ("mediun"), or a number, also as a plural ("32s").
    Money amounts and IDs ("$32", "O-32") are not sizes."""
    wanted = canonical_size(next(_field(p, "size") for p in catalog if _field(p, "sku") == sku))
    for text in analysis.texts:
        plain = _MONEY_MENTION_RE.sub(" ", normalize_text(text))
        if any(_token_size(token) == wanted for token in _TOKEN_RE.findall(plain)):
            return True
    return False


def _token_size(token: str) -> str | None:
    """The size label one written token stands for, or None."""
    number = token[:-1] if token.endswith("s") else token
    if _NUMBER_RE.match(number):
        return canonical_size(number)
    compact = token.replace("-", "")
    if compact in SIZE_WORDS:
        return SIZE_WORDS[compact]
    if len(compact) >= _SIZE_TYPO_MIN:
        return next((label for word, label in SIZE_WORDS.items()
                     if len(word) >= _SIZE_TYPO_MIN and _one_edit_apart(compact, word)), None)
    return None


def extract_sizes(text: str) -> list[str]:
    """Canonical sizes mentioned in free text (used by the reply guard), with the same rules."""
    reader = _MessageReader(_build_account([], []), scope_orders=[], prior_remedy=NONE)
    return reader.read(text).all_sizes
