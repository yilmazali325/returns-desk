"""Store tests: byte-identical export, atomic commits and rollback, primary-key duplicate
guards, deterministic IDs, record shapes, and reopening a store file."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from desk.domain import (
    ALREADY_RETURNED,
    EXCHANGE,
    LINE_NOT_FOUND,
    OPEN,
    OUT_OF_STOCK,
    REFUND,
    REQUEST_ALREADY_COMPLETED,
    SUPERSEDED,
    Proposal,
    binding_hash,
    make_binding,
)
from desk.store import CommitRefused, Store, common_fields, record_return

PUBLIC = json.loads((Path(__file__).resolve().parents[2] / "candidate_assets" / "public_cases.json").read_text())["cases"]


def _proposal(remedy: str = REFUND, replacement: str | None = None, *, pid: str = "P-0001", amount: int = 4200,
              request_id: str = "REQ-T1", line_id: str = "X9", status: str = OPEN) -> Proposal:  # fmt: skip
    binding = make_binding(customer_id="C7", order_id="ORD-55", line_id=line_id, remedy=remedy,
                           amount_cents=amount, replacement_sku=replacement, request_id=request_id)  # fmt: skip
    return Proposal(
        proposal_id=pid, request_id=request_id, customer_id="C7", order_id="ORD-55", line_id=line_id,
        remedy=remedy, original_sku="HOODIE-M", replacement_sku=replacement, amount_cents=amount, currency="USD",
        eligibility={"delivered_on": "2026-03-01", "as_of": "2026-03-15", "days_since_delivery": 14, "window_days": 30},
        binding_hash=binding_hash(binding), status=status, created_turn=1,
    )  # fmt: skip


def _commit(store: Store, proposal: Proposal) -> dict:
    with store.transaction() as tx:
        return record_return(tx, proposal, store.as_of)


# --- exact-shape export --------------------------------------------------------------------


@pytest.mark.parametrize("case", PUBLIC, ids=[c["case_id"] for c in PUBLIC])
def test_public_initial_states_round_trip_byte_identically(case):
    store = Store(case["initial_state"])
    assert json.dumps(store.export()) == json.dumps(case["initial_state"])
    assert json.dumps(store.export(), indent=2, ensure_ascii=False) == json.dumps(case["initial_state"], indent=2, ensure_ascii=False)


def test_unknown_fields_and_key_order_survive_load_and_commit(state):
    state["x_top"] = {"nested": [1, 2.5, None, True]}
    state["orders"][0]["x_note"] = "gift"
    state["orders"][0]["lines"][0]["x_flag"] = None
    state["products"][0] = {"x_first": 1, **state["products"][0]}
    store = Store(state)
    assert json.dumps(store.export()) == json.dumps(state)

    _commit(store, _proposal())
    final = store.export()
    assert list(final) == list(state)  # exactly the input keys, in input order
    assert final["x_top"] == state["x_top"] and final["orders"][0]["x_note"] == "gift"
    assert list(final["orders"][0]["lines"][0]) == list(state["orders"][0]["lines"][0])
    assert list(final["products"][0]) == list(state["products"][0])


def test_store_neither_aliases_its_input_nor_hands_out_its_document(state):
    store = Store(state)
    state["products"][0]["stock"] = 999
    exported = store.export()
    assert exported["products"][0]["stock"] == 2
    exported["products"][0]["stock"] = 777
    assert store.export()["products"][0]["stock"] == 2


def test_invalid_as_of_is_rejected_at_load(state):
    state["as_of"] = "26/10/2026"
    with pytest.raises(ValueError, match="as_of"):
        Store(state)


# --- seeding -----------------------------------------------------------------------------


def test_seed_writes_version_1_and_mirrors_the_input(state):
    state["orders"][0]["lines"][1]["returned"] = True
    state["orders"][0]["lines"][0]["returned"] = None  # not exactly False -> treated as returned
    state["idempotency"] = {
        "REQ-OLD": {"status": "completed", "order_id": "ORD-55"},
        "REQ-PENDING": {"status": "pending"},
        "REQ-JUNK": "not a record",
    }
    state["returns"] = [{"return_id": "RET-0001"}, {"return_id": "RET-0002"}]
    store = Store(state)
    assert store.version() == 1
    assert sorted(store.returned_lines()) == [("C7", "ORD-55", "X10"), ("C7", "ORD-55", "X9")]
    assert store.idempotency_record("REQ-OLD") == {"status": "completed", "order_id": "ORD-55"}
    assert store.idempotency_record("REQ-PENDING") is None
    with store.transaction() as tx:
        assert tx.next_id("RET") == "RET-0003"  # continues after existing records
        assert tx.next_id("TX") == "TX-0004"  # three idempotency entries in the input
        assert tx.next_id("P") == "P-0001"


def test_seed_tolerates_a_non_dict_idempotency(state):
    state["idempotency"] = []
    assert Store(state).idempotency_record("anything") is None


# --- transactions -------------------------------------------------------------------------


def test_exception_inside_a_transaction_rolls_everything_back(state, lookup):
    store = Store(state)
    with pytest.raises(RuntimeError):
        with store.transaction() as tx:
            tx.next_id("P")
            tx.set_stock("HOODIE-L", 0)
            tx.put_proposal(_proposal())
            tx.append_record("returns", {"return_id": "RET-X", "transaction_id": "TX-X", "amount_cents": 0})
            raise RuntimeError("boom")
    assert store.export() == state and store.version() == 1
    assert store.proposals() == [] and store.ledger() == []
    with store.transaction() as tx:
        assert tx.next_id("P") == "P-0001"  # the rolled-back ID was not burned


def test_commit_without_document_change_keeps_the_version(state):
    store = Store(state)
    with store.transaction() as tx:
        tx.put_proposal(_proposal())
    assert store.version() == 1 and [p.proposal_id for p in store.proposals()] == ["P-0001"]


def test_ids_are_deterministic_per_prefix(state):
    store = Store(state)
    with store.transaction() as tx:
        ids = [tx.next_id("P"), tx.next_id("P"), tx.next_id("EV"), tx.next_id("P")]
    assert ids == ["P-0001", "P-0002", "EV-0001", "P-0003"]


# --- record shapes --------------------------------------------------------------------------


def test_refund_commit_writes_every_record_with_common_fields(state, lookup, common_fields):
    store = Store(state)
    record = _commit(store, _proposal())
    final = store.export()
    assert record == final["idempotency"]["REQ-T1"]
    assert record["result"] == {
        "transaction_id": "TX-0001", "return_id": "RET-0001", "refund_id": "RF-0001", "quarantine_id": "Q-0001",
        "amount_cents": 4200, "replacement_sku": None,
    }  # fmt: skip
    assert (record["status"], record["action"]) == ("completed", REFUND)
    for collection in ("returns", "refunds", "quarantine"):
        (item,) = final[collection]
        assert all(item[name] == record[name] for name in common_fields), collection
    assert final["returns"][0]["remedy"] == REFUND and final["returns"][0]["disposition"] == "quarantine"
    assert final["refunds"][0]["amount_cents"] == 4200  # paid, not the 5000 catalog price
    assert final["quarantine"][0]["quantity"] == 1 and final["quarantine"][0]["reason"] == "customer_return"
    assert final["exchanges"] == []
    assert record["business_date"] == "2026-03-15" and record["sku"] == "HOODIE-M" and record["replacement_sku"] is None
    assert lookup.line(final, "ORD-55", "X9")["returned"] is True
    assert [p["stock"] for p in final["products"]] == [p["stock"] for p in state["products"]]  # quarantine never restocks
    assert store.version() == 2 and ("C7", "ORD-55", "X9") in store.returned_lines()


def test_exchange_commit_takes_one_replacement_unit_and_never_restocks_the_original(state, lookup, common_fields):
    store = Store(state)
    record = _commit(store, _proposal(EXCHANGE, "HOODIE-L", amount=0))
    final = store.export()
    assert lookup.product(final, "HOODIE-L")["stock"] == 2  # 3 - 1
    assert lookup.product(final, "HOODIE-M")["stock"] == 1  # original unchanged
    (exchange,) = final["exchanges"]
    assert exchange["original_sku"] == "HOODIE-M" and exchange["replacement_sku"] == "HOODIE-L"
    assert all(exchange[name] == record[name] for name in common_fields)
    assert record["amount_cents"] == 0 and record["result"]["exchange_id"] == "EX-0001"
    assert final["refunds"] == [] and len(final["quarantine"]) == 1 and final["returns"][0]["remedy"] == EXCHANGE
    stock_rows = [row for row in store.ledger() if row["type"] == "stock"]
    assert stock_rows == [{"seq": stock_rows[0]["seq"], "transaction_id": "TX-0001", "type": "stock",
                           "sku": "HOODIE-L", "old_stock": 3, "new_stock": 2}]  # fmt: skip


def test_transaction_effects_list_every_business_change(state):
    store = Store(state)
    with store.transaction() as tx:
        record_return(tx, _proposal(EXCHANGE, "HOODIE-L", amount=0), store.as_of)
        effects = list(tx.effects)
    assert [e["type"] for e in effects] == ["line_returned", "record", "stock", "record", "record", "idempotency"]
    assert [e.get("collection") for e in effects if e["type"] == "record"] == ["returns", "exchanges", "quarantine"]
    assert len(store.ledger()) == len(effects)


def test_common_fields_order():
    fields = common_fields(_proposal(), "TX-9", "2026-03-15")
    assert list(fields)[:3] == ["transaction_id", "request_id", "proposal_id"] and fields["status"] == "completed"


def test_missing_collections_in_a_malformed_input_do_not_crash_a_commit(state):
    for key in ("returns", "refunds", "quarantine", "idempotency"):
        del state[key]
    store = Store(state)
    _commit(store, _proposal())
    assert len(store.export()["refunds"]) == 1


# --- primary-key and stock guards ------------------------------------------------------------


def test_returned_lines_primary_key_blocks_a_second_return_and_rolls_back(state):
    store = Store(state)
    _commit(store, _proposal())
    before = store.export()
    with pytest.raises(CommitRefused) as refused:
        _commit(store, _proposal(EXCHANGE, "HOODIE-L", pid="P-0002", amount=0, request_id="REQ-T2"))
    assert refused.value.code == ALREADY_RETURNED
    assert store.export() == before and store.idempotency_record("REQ-T2") is None


def test_idempotency_primary_key_blocks_a_second_transaction_for_the_request(state):
    store = Store(state)
    _commit(store, _proposal())
    before = store.export()
    with pytest.raises(CommitRefused) as refused:
        _commit(store, _proposal(line_id="X10", pid="P-0002", amount=1500))
    assert refused.value.code == REQUEST_ALREADY_COMPLETED
    assert store.export() == before


@pytest.mark.parametrize("stock", [0, -2, None])
def test_decrement_never_takes_stock_below_zero(state, lookup, stock):
    lookup.product(state, "HOODIE-L")["stock"] = stock
    store = Store(state)
    with pytest.raises(CommitRefused) as refused:
        _commit(store, _proposal(EXCHANGE, "HOODIE-L", amount=0))
    assert refused.value.code == OUT_OF_STOCK and store.export() == state


def test_decrement_of_unknown_sku_is_refused(state):
    store = Store(state)
    with pytest.raises(CommitRefused, match="OUT_OF_STOCK"):
        _commit(store, _proposal(EXCHANGE, "HOODIE-9XL", amount=0))


def test_marking_a_line_that_is_not_on_the_account_is_refused(state):
    store = Store(state)
    with pytest.raises(CommitRefused) as refused:
        _commit(store, _proposal(line_id="X404"))
    assert refused.value.code == LINE_NOT_FOUND and refused.value.message.endswith("not on this account")
    assert store.returned_lines() == []


# --- proposals, events, stock ------------------------------------------------------------------


def test_proposal_upsert_keeps_creation_order_and_tracks_open_ones(state):
    store = Store(state)
    with store.transaction() as tx:
        tx.put_proposal(_proposal(pid="P-0001"))
        tx.put_proposal(_proposal(pid="P-0002", request_id="REQ-OTHER"))
    with store.transaction() as tx:
        assert tx.open_proposal("REQ-T1").proposal_id == "P-0001"
        tx.put_proposal(tx.proposal("P-0001").with_status(SUPERSEDED))
        assert tx.open_proposal("REQ-T1") is None
    assert [p.proposal_id for p in store.proposals()] == ["P-0001", "P-0002"]
    assert store.proposal("P-0001").status == SUPERSEDED and store.proposal("P-9") is None
    assert store.open_proposal("REQ-OTHER").proposal_id == "P-0002" and store.open_proposal("REQ-T1") is None


def test_set_stock_is_plain_assignment(state, lookup):
    store = Store(state)
    with store.transaction() as tx:
        assert tx.set_stock("HOODIE-L", -4) == (True, 3)
        assert tx.set_stock("NOPE", 1) == (False, None)
        tx.add_event({"event_id": "EV-0001", "type": "stock_change"})
    assert lookup.product(store.export(), "HOODIE-L")["stock"] == -4
    assert store.events() == [{"event_id": "EV-0001", "type": "stock_change"}]


def test_as_of_property(state):
    assert Store(state).as_of == "2026-03-15"


# --- durability -------------------------------------------------------------------------------


def test_reopening_a_store_file_restores_everything(state, tmp_path):
    path = str(tmp_path / "session.sqlite")
    store = Store(state, path)
    with store.transaction() as tx:
        tx.put_proposal(_proposal())
    _commit(store, _proposal())
    with store.transaction() as tx:
        tx.add_event({"event_id": tx.next_id("EV"), "type": "stock_change"})
    expected = store.export()
    store.close()

    reopened = Store.open(path)
    assert json.dumps(reopened.export()) == json.dumps(expected)
    assert reopened.idempotency_record("REQ-T1")["transaction_id"] == "TX-0001"
    assert [p.proposal_id for p in reopened.proposals()] == ["P-0001"]
    assert reopened.events()[0]["event_id"] == "EV-0001" and reopened.version() == 2
    with reopened.transaction() as tx:
        assert tx.next_id("TX") == "TX-0002"
    reopened.close()


def test_a_session_file_cannot_be_initialised_twice(state, tmp_path):
    path = str(tmp_path / "session.sqlite")
    Store(state, path).close()
    with pytest.raises(ValueError, match="already holds a session"):
        Store(copy.deepcopy(state), path)


def test_reopening_an_empty_file_fails(tmp_path):
    with pytest.raises(ValueError, match="no store to reopen"):
        Store.open(str(tmp_path / "empty.sqlite"))
