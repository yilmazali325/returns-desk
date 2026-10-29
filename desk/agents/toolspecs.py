"""Tool schemas for the two agents (SPEC §6), as module constants.

Why constants: every LLM step of a role must send byte-identical tool definitions, otherwise the
prompt cache is lost (measured in docs/notes/llm_infra.md: changing the tool list drops the cached
prefix to 0 tokens). What an agent may call *at this step* is never expressed by editing these
lists; it is passed separately as `allowed` (enforced by the decoder in constrained mode and by
code in native mode).

Design choices that come from measurements on this stack (docs/notes/spec_critique.json):
- IDs are free strings, never enums. An enum of the account's lines makes the decoder pick a
  valid-looking wrong line for an off-account request; code validates IDs after generation.
- `handoff_to_operations` is a tagged union, one branch per intent, with only the fields that
  intent needs. Operations tools take no IDs at all: they act on the target the orchestrator
  grounded and injected.
- `report_to_resolution` comes in two shapes so the baseline profile can keep a free-text summary
  while the final profile lets code build the report (the LLM only names the outcome).
"""

from __future__ import annotations

from typing import Any

from desk.llm import ToolSpec

STRING = {"type": "string"}


def _object(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


# --- Resolution (customer-facing) -------------------------------------------------------------

FIND_ORDER_LINES = "find_order_lines"
HANDOFF = "handoff_to_operations"
REPLY = "reply_to_customer"

PROPOSE_REFUND = "propose_refund"
PROPOSE_EXCHANGE = "propose_exchange"
CHECK_STATUS = "check_status"
EXECUTE_CONFIRMED = "execute_confirmed"
HANDOFF_INTENTS = (PROPOSE_REFUND, PROPOSE_EXCHANGE, CHECK_STATUS, EXECUTE_CONFIRMED)

FIND_ORDER_LINES_SPEC = ToolSpec(
    FIND_ORDER_LINES,
    "List the authenticated customer's order lines (order id, line id, SKU, product, size).",
    _object({"order_ref": {"type": "string", "description": 'An order id, or "ALL".'}}, ("order_ref",)),
)

_SIZE_HELP = 'The size the customer wants, e.g. "L"; "UP" or "DOWN" for one size bigger or smaller; "UNSPECIFIED".'
_PRODUCT_HELP = '"SAME", or the product_id the customer asks for instead.'
# Optional citation (SPEC §6.0): the customer's exact words that pick this line when the ids they
# typed leave more than one line. Code accepts the target iff the quote is theirs and fits only it.
EVIDENCE = {"type": "string", "description": "The customer's exact words that pick this line, when several lines could fit."}

HANDOFF_SPEC = ToolSpec(
    HANDOFF,
    ("Hand the request to the back-office Operations agent. Pick exactly one intent: propose_refund or "
     "propose_exchange (also to re-check stock or re-present a proposal), check_status (only to learn whether a "
     "confirmed request already went through), execute_confirmed (only after the confirm control)."),
    {
        "type": "object",
        "oneOf": [
            _object({"intent": {"const": PROPOSE_REFUND}, "order_id": STRING, "line_id": STRING, "evidence": EVIDENCE},
                    ("intent", "order_id", "line_id")),
            _object({"intent": {"const": PROPOSE_EXCHANGE}, "order_id": STRING, "line_id": STRING,
                     "requested_size": {"type": "string", "description": _SIZE_HELP},
                     "requested_product": {"type": "string", "description": _PRODUCT_HELP},
                     "evidence": EVIDENCE},
                    ("intent", "order_id", "line_id", "requested_size")),
            _object({"intent": {"const": CHECK_STATUS}}, ("intent",)),
            _object({"intent": {"const": EXECUTE_CONFIRMED}}, ("intent",)),
        ],
    },
)

REPLY_SPEC = ToolSpec(
    REPLY,
    "Send your message to the customer. This ends your turn.",
    _object({"message": STRING}, ("message",)),
    terminal=True,
)

RESOLUTION_TOOLS: tuple[ToolSpec, ...] = (FIND_ORDER_LINES_SPEC, HANDOFF_SPEC, REPLY_SPEC)

# --- Operations (back office) -----------------------------------------------------------------

LIST_EXCHANGE_OPTIONS = "list_exchange_options"
CREATE_PROPOSAL = "create_proposal"
EXECUTE_PROPOSAL = "execute_proposal"
GET_REQUEST_STATUS = "get_request_status"
REPORT = "report_to_resolution"

OUTCOMES = ("proposed", "denied", "completed", "needs_info", "failed")

LIST_EXCHANGE_OPTIONS_SPEC = ToolSpec(
    LIST_EXCHANGE_OPTIONS, "List the other sizes of the briefed line's product with price, stock and validity.",
    _object({}))
CREATE_PROPOSAL_SPEC = ToolSpec(
    CREATE_PROPOSAL,
    "Create the proposal for the briefed line. The amount is computed by the tool.",
    _object({"remedy": {"type": "string", "enum": ["refund", "exchange"]},
             "replacement_sku": {"type": "string", "description": "Exchange only: the SKU of the requested size."}},
            ("remedy",)),
)
EXECUTE_PROPOSAL_SPEC = ToolSpec(
    EXECUTE_PROPOSAL, "Execute the proposal the customer explicitly confirmed this turn. At most once.", _object({}))
GET_REQUEST_STATUS_SPEC = ToolSpec(
    GET_REQUEST_STATUS, "Look up whether this customer request has already been completed.", _object({}))

_OUTCOME = {"type": "string", "enum": list(OUTCOMES)}
REPORT_STRUCTURED_SPEC = ToolSpec(
    REPORT, "Report the outcome to the Resolution agent. This ends your work.",
    _object({"outcome": _OUTCOME}, ("outcome",)), terminal=True)
REPORT_FREE_TEXT_SPEC = ToolSpec(
    REPORT, "Report the outcome to the Resolution agent. This ends your work.",
    _object({"outcome": _OUTCOME,
             "summary": {"type": "string", "description": "One or two sentences for the Resolution agent."}},
            ("outcome", "summary")),
    terminal=True)

# No separate eligibility tool: create_proposal runs the eligibility rules and is the call whose
# answer decides. With a check_eligibility tool the model called it first on every handoff (8% of
# all LLM time on 241 variant cases) and then often reported "denied" without the deciding call
# (58 times). See docs/notes/tuning_log.md, iteration 2. The gateway's check_eligibility and
# check_proposal are read-only calls for the orchestrator's blocking checks, not agent tools.
_OPERATIONS_BASE = (LIST_EXCHANGE_OPTIONS_SPEC, CREATE_PROPOSAL_SPEC, EXECUTE_PROPOSAL_SPEC, GET_REQUEST_STATUS_SPEC)
OPERATIONS_TOOLS_STRUCTURED: tuple[ToolSpec, ...] = (*_OPERATIONS_BASE, REPORT_STRUCTURED_SPEC)
OPERATIONS_TOOLS_FREE_TEXT: tuple[ToolSpec, ...] = (*_OPERATIONS_BASE, REPORT_FREE_TEXT_SPEC)


def operations_tools(structured_report: bool) -> tuple[ToolSpec, ...]:
    """The Operations tool list for a profile (one fixed tuple per profile, never rebuilt)."""
    return OPERATIONS_TOOLS_STRUCTURED if structured_report else OPERATIONS_TOOLS_FREE_TEXT
