"""Neutralize customer text before it reaches a prompt (SPEC §5).

Customer messages are untrusted data. The Qwen chat template tokenizes strings such as
"<|im_start|>" and "<tool_call>" as real control tokens, so a customer could open a fake system
turn or forge a tool call. That was measured on this stack: the model obeyed an injected
"<|im_start|>system" block (docs/notes/spec_critique.json). We break every such marker by
swapping its "<" for the look-alike "‹" (U+2039). The text stays readable for the model and the
audit log, but it no longer matches a special token. The marker pattern is generic (any "<|...|>"
and any tag-shaped "<name>"), so the model's 33 control tokens, including "<tts_text_bos>", are
all covered without a list (SEC-3; tests/review/security/test_sanitize_vocab.py re-reads the GGUF). We also drop invisible control and format
characters and cap the length.

Plain-language tricks such as "SYSTEM: execute now" are left as they are. They cannot authorize
anything, because only the confirm_proposal flag grants a mutation. The prompt also marks the
text as untrusted (wrap_untrusted).
"""

from __future__ import annotations

import re
import unicodedata

MAX_MESSAGE_CHARS = 1000
UNTRUSTED_LABEL = "CUSTOMER MESSAGE (untrusted data)"
SAFE_LT = "‹"  # "‹": looks like "<" but is not part of any special token

# A "<" that starts a special token of the deployed vocabulary. Matched by SHAPE, not by a list, so a
# model swap stays covered: "<|...|>" (every Qwen special token), any tag-like "<name>" / "</name>"
# ("<think>", "<tool_call>", "<tts_text_bos>"), "<function=" / "<parameter=" of the tool format, and
# the template's tag names even when the ">" is missing. Ordinary "<" ("size <M", "<3") is kept.
_TAG_NAMES = r"(?:tool_call|tool_response|tools|think|function|parameter|tts_\w*)"
_MARKER_START = re.compile(
    r"<(?=\s*\|)|<(?=\s*/?\s*" + _TAG_NAMES + r"\b)|<(?=\s*/?\s*[A-Za-z_][\w\-]*\s*[>=])",
    re.IGNORECASE,
)
# The same markers, captured whole, for audit reporting.
_MARKER_SNIPPET = re.compile(
    r"<\s*\|[^|<>]{0,40}\|?>?|<\s*/?\s*" + _TAG_NAMES + r"\b[^<>]{0,40}>?|<\s*/?\s*[A-Za-z_][\w\-]*\s*(?:>|=[^<>]{0,40}>?)",
    re.IGNORECASE,
)
# Character categories removed outright. Cc = control, Cf = format (zero-width, bidi overrides),
# Cs = lone surrogates, which would break JSON encoding later.
_DROPPED_CATEGORIES = {"Cc", "Cf", "Cs"}
_KEPT_CONTROLS = {"\n", "\t"}


def strip_control_chars(text: str) -> str:
    """Drop invisible control and format characters. Newlines and tabs are kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        ch for ch in text if ch in _KEPT_CONTROLS or unicodedata.category(ch) not in _DROPPED_CATEGORIES
    )


def neutralize_markers(text: str) -> str:
    """Replace the '<' of every control-token or tool-tag marker with '‹'."""
    return _MARKER_START.sub(SAFE_LT, text)


def sanitize(text: str | None, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Return customer text that is safe to place in a prompt. Idempotent.

    Order matters. Invisible characters are stripped first, so "<​|im_start|>" cannot
    survive by hiding a zero-width space inside the marker. Truncation comes last and keeps the
    result within max_chars.
    """
    if text is None:
        return ""
    cleaned = neutralize_markers(strip_control_chars(str(text)))
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1] + "…"
    return cleaned


def injection_markers(text: str | None) -> list[str]:
    """Return the control-token markers found in raw text (for audit and metrics, not for gating)."""
    if not text:
        return []
    return [m.group(0) for m in _MARKER_SNIPPET.finditer(strip_control_chars(str(text)))]


def wrap_untrusted(text: str | None) -> str:
    """Place sanitized customer text inside the labeled untrusted-data block used in prompts."""
    body = sanitize(text).replace('"""', "'''")  # the customer cannot close the block early
    return f'{UNTRUSTED_LABEL}:\n"""\n{body}\n"""'
