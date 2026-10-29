"""Nothing in the shipped code may know the public cases (CONTRACT: "Never hardcode IDs, phrases,
counts or expected outcomes"). Graders run unseen variants, so a fixture ID or phrase in desk/,
server/ or adapter.py would be an answer lookup.

Three checks:
1. Source: no case id, customer id, request id, order id or SKU of a public case, and no public
   customer message (or a distinctive part of one), anywhere in the shipped Python and UI files.
2. Prompts: the static prompts and desk notes also avoid the public line ids and product ids.
3. Runtime: running every public case, no prompt ever contains the customer_id or request_id
   (the gateway scopes identity; models never see it).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from desk import orchestrator
from desk.agents import prompts
from desk.llm import FakeLLM
from desk.profiles import get_profile
from desk.session import run_case

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]
SHIPPED = sorted([*ROOT.joinpath("desk").rglob("*.py"), ROOT / "adapter.py",
                  *[p for p in ROOT.joinpath("server").rglob("*") if p.suffix in (".py", ".js", ".html", ".css")]])


def public_tokens() -> dict[str, set[str]]:
    tokens: dict[str, set[str]] = {"ids": set(), "lines": set(), "products": set(), "messages": set()}
    for case in CASES:
        state = case["initial_state"]
        tokens["ids"] |= {case["case_id"], case["customer_id"], case["request_id"]}
        tokens["ids"] |= {o["order_id"] for o in state["orders"]} | {p["sku"] for p in state["products"]}
        tokens["lines"] |= {line["line_id"] for o in state["orders"] for line in o["lines"]}
        tokens["products"] |= {p["product_id"] for p in state["products"]}
        tokens["messages"] |= {t["message"] for t in case["turns"]}
    return tokens


TOKENS = public_tokens()


def fragments(message: str) -> list[str]:
    """The message itself and every run of 6 consecutive words (a distinctive phrase)."""
    words = message.split()
    return [message] + [" ".join(words[i:i + 6]) for i in range(max(0, len(words) - 5))]


def whole_token(token: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(token) + r"(?![A-Za-z0-9_-])")


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: str(p.relative_to(ROOT)))
def test_shipped_files_contain_no_public_ids_or_phrases(path: Path) -> None:
    text = path.read_text()
    leaked = [t for t in sorted(TOKENS["ids"]) if whole_token(t).search(text)]
    folded = text.casefold()
    leaked += [f for m in sorted(TOKENS["messages"]) for f in fragments(m) if f.casefold() in folded]
    assert not leaked, f"{path.name} mentions public fixtures: {leaked}"


PROMPT_TEXTS = {
    "RESOLUTION_SYSTEM": prompts.RESOLUTION_SYSTEM,
    "OPERATIONS_SYSTEM_STRUCTURED": prompts.OPERATIONS_SYSTEM_STRUCTURED,
    "OPERATIONS_SYSTEM_FREE_TEXT": prompts.OPERATIONS_SYSTEM_FREE_TEXT,
    "NO_CARD_TEXT": prompts.NO_CARD_TEXT,
    "CONFIRMATION_WITHHELD": prompts.CONFIRMATION_WITHHELD,
    "CONFIRMATION_DONE": prompts.CONFIRMATION_DONE,
    "REPLAY_NOTE": orchestrator.REPLAY_NOTE,
    "ALREADY_COMPLETED_NOTE": orchestrator.ALREADY_COMPLETED_NOTE,
    "CANCEL_NOTE": orchestrator.CANCEL_NOTE,
}


@pytest.mark.parametrize("name", sorted(PROMPT_TEXTS))
def test_static_prompts_use_only_invented_examples(name: str) -> None:
    text = PROMPT_TEXTS[name]
    found = [t for t in sorted(TOKENS["ids"] | TOKENS["lines"]) if whole_token(t).search(text)]
    found += [p for p in sorted(TOKENS["products"]) if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)]
    assert not found, f"{name} contains fixture tokens {found}"


def test_examples_in_the_prompt_are_the_invented_ones() -> None:
    assert "ORD-7Q" in prompts.RESOLUTION_SYSTEM and "LN-3" in prompts.RESOLUTION_SYSTEM and "SCARF" in prompts.RESOLUTION_SYSTEM


@pytest.mark.parametrize("profile_name", ["F", "B"])
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_prompts_never_carry_customer_or_request_ids(case: dict, profile_name: str) -> None:
    profile = get_profile(profile_name)

    def answer(request):  # a model that only asks questions; Operations reports immediately
        if request.role == "resolution":
            return ("reply_to_customer", {"message": "Could you tell me more?"})
        return ("report_to_resolution", {"outcome": "needs_info", **({} if profile.structured_report else {"summary": "ok"})})

    llm = FakeLLM(answer)
    run_case(case, llm, profile)
    assert llm.requests, "the case made LLM calls"
    for request in llm.requests:
        blob = json.dumps(request.messages)
        assert case["customer_id"] not in blob and case["request_id"] not in blob
