"""Tests for length-prefixed HMAC idempotency keys and the 3-state / 5-state side-effect protocol."""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from llmlab.durable.idempotency import (
    EffectIdentity,
    ProviderCapability,
    UnknownSideEffectOutcome,
    _frame,
    execute_idempotent_effect,
    idempotency_key,
)
from tests.conftest import VALID_GRAPH_DIGEST, seed_genesis_checkpoint

TEST_SECRET = b"super-secret-32-byte-hmac-key-01"


def test_frame_eliminates_concatenation_collisions() -> None:
    """Verify length-prefixed framing distinguishes boundary shifts in adjacent fields."""
    assert _frame(b"node_a", b"b_step") != _frame(b"node_", b"ab_step")

    state_hash = hashlib.sha256(b'{"amount":100}').digest()
    id_1 = EffectIdentity(
        workflow_id="wf-1",
        graph_digest=VALID_GRAPH_DIGEST,
        node_id="pay_a",
        logical_attempt=1,
        state_hash=state_hash,
    )
    id_2 = EffectIdentity(
        workflow_id="wf-1",
        graph_digest=VALID_GRAPH_DIGEST,
        node_id="pay_a",
        logical_attempt=2,
        state_hash=state_hash,
    )
    assert idempotency_key(TEST_SECRET, id_1) == idempotency_key(TEST_SECRET, id_1)
    assert idempotency_key(TEST_SECRET, id_1) != idempotency_key(TEST_SECRET, id_2)


async def test_crash_between_intent_and_committed_on_idempotent_provider(
    db_engine: AsyncEngine,
) -> None:
    """Simulate a socket crash after Phase 1 INTENT commit and verify deduplicated recovery."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        ckpt_id = await seed_genesis_checkpoint(conn, workflow_id)

    ident = EffectIdentity(
        workflow_id=str(workflow_id),
        graph_digest=VALID_GRAPH_DIGEST,
        node_id="charge_card",
        logical_attempt=1,
        state_hash=hashlib.sha256(b'{"cents":5000}').digest(),
    )

    provider_ledger: dict[str, dict[str, Any]] = {}
    network_invocations = 0

    async def crashing_provider(key: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal network_invocations
        network_invocations += 1
        # Provider executes the charge on the remote server before the connection drops
        provider_ledger[key] = {"charge_id": "ch_999", "cents": payload["cents"]}
        raise ConnectionResetError("Simulated TCP reset before HTTP response reached worker.")

    with pytest.raises(ConnectionResetError):
        await execute_idempotent_effect(
            engine=db_engine,
            secret=TEST_SECRET,
            ident=ident,
            checkpoint_id=ckpt_id,
            write_index=0,
            branch_id="main",
            channel_name="payment_result",
            request_payload={"cents": 5000},
            capability=ProviderCapability.IDEMPOTENT_BY_KEY,
            provider_call=crashing_provider,
        )

    # Verify the database durably captured the Ambiguous state via write_state = 'INTENT'
    async with db_engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT write_state, recorded_response FROM checkpoint_writes WHERE checkpoint_id = :c"),
                {"c": str(ckpt_id)},
            )
        ).mappings().one()
        assert row["write_state"] == "INTENT"
        assert row["recorded_response"] is None

    async def healthy_idempotent_provider(key: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal network_invocations
        network_invocations += 1
        if key in provider_ledger:
            return provider_ledger[key]
        provider_ledger[key] = {"charge_id": "ch_duplicate_bug", "cents": payload["cents"]}
        return provider_ledger[key]

    # Recovery run 1: transitions INTENT -> COMMITTED using provider's idempotency cache
    recovered_result = await execute_idempotent_effect(
        engine=db_engine,
        secret=TEST_SECRET,
        ident=ident,
        checkpoint_id=ckpt_id,
        write_index=0,
        branch_id="main",
        channel_name="payment_result",
        request_payload={"cents": 5000},
        capability=ProviderCapability.IDEMPOTENT_BY_KEY,
        provider_call=healthy_idempotent_provider,
    )
    assert recovered_result == {"charge_id": "ch_999", "cents": 5000}
    assert network_invocations == 2

    # Recovery run 2 (replay): row is now COMMITTED, so zero network calls occur
    replayed_result = await execute_idempotent_effect(
        engine=db_engine,
        secret=TEST_SECRET,
        ident=ident,
        checkpoint_id=ckpt_id,
        write_index=0,
        branch_id="main",
        channel_name="payment_result",
        request_payload={"cents": 5000},
        capability=ProviderCapability.IDEMPOTENT_BY_KEY,
        provider_call=healthy_idempotent_provider,
    )
    assert replayed_result == {"charge_id": "ch_999", "cents": 5000}
    assert network_invocations == 2


async def test_status_queryable_reconciles_ambiguous_intent_without_re_POSTing(
    db_engine: AsyncEngine,
) -> None:
    """Verify STATUS_QUERYABLE providers poll status_Reconciler instead of re-invoking POST."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        ckpt_id = await seed_genesis_checkpoint(conn, workflow_id)

    ident = EffectIdentity(
        workflow_id=str(workflow_id),
        graph_digest=VALID_GRAPH_DIGEST,
        node_id="create_ticket",
        logical_attempt=1,
        state_hash=hashlib.sha256(b'{"title":"Outage"}').digest(),
    )

    post_calls = 0

    async def failing_first_post(key: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal post_calls
        post_calls += 1
        raise TimeoutError("Gateway timed out after ticket creation.")

    with pytest.raises(TimeoutError):
        await execute_idempotent_effect(
            engine=db_engine,
            secret=TEST_SECRET,
            ident=ident,
            checkpoint_id=ckpt_id,
            write_index=0,
            branch_id="main",
            channel_name="ticket",
            request_payload={"title": "Outage"},
            capability=ProviderCapability.STATUS_QUERYABLE,
            provider_call=failing_first_post,
        )

    async def reconciler_query(key: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        return {"ticket_id": "INC-4470", "client_ref": key, "title": payload["title"]}

    result = await execute_idempotent_effect(
        engine=db_engine,
        secret=TEST_SECRET,
        ident=ident,
        checkpoint_id=ckpt_id,
        write_index=0,
        branch_id="main",
        channel_name="ticket",
        request_payload={"title": "Outage"},
        capability=ProviderCapability.STATUS_QUERYABLE,
        provider_call=failing_first_post,
        status_Reconciler=reconciler_query,
    )

    assert result["ticket_id"] == "INC-4470"
    assert post_calls == 1


async def test_non_idempotent_provider_raises_unknown_outcome_on_preexisting_intent(
    db_engine: AsyncEngine,
) -> None:
    """Verify NON_IDEMPOTENT providers never auto-retry an uncommitted INTENT row."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        ckpt_id = await seed_genesis_checkpoint(conn, workflow_id)

    ident = EffectIdentity(
        workflow_id=str(workflow_id),
        graph_digest=VALID_GRAPH_DIGEST,
        node_id="send_wire",
        logical_attempt=1,
        state_hash=hashlib.sha256(b'{"wire":10000}').digest(),
    )

    calls = 0

    async def wire_call(key: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("Socket hung up mid-response.")

    with pytest.raises(RuntimeError):
        await execute_idempotent_effect(
            engine=db_engine,
            secret=TEST_SECRET,
            ident=ident,
            checkpoint_id=ckpt_id,
            write_index=0,
            branch_id="main",
            channel_name="wire_status",
            request_payload={"wire": 10000},
            capability=ProviderCapability.NON_IDEMPOTENT,
            provider_call=wire_call,
        )

    with pytest.raises(UnknownSideEffectOutcome) as exc_info:
        await execute_idempotent_effect(
            engine=db_engine,
            secret=TEST_SECRET,
            ident=ident,
            checkpoint_id=ckpt_id,
            write_index=0,
            branch_id="main",
            channel_name="wire_status",
            request_payload={"wire": 10000},
            capability=ProviderCapability.NON_IDEMPOTENT,
            provider_call=wire_call,
        )

    assert exc_info.value.intent_payload == {"wire": 10000}
    assert calls == 1
