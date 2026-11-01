"""Deterministic HMAC idempotency keys and two-phase side-effect reconciliation."""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

IDEMPOTENCY_DOMAIN: Final[bytes] = b"llmlab/idempotency-key/v1"


class ProviderCapability(str, Enum):
    """Classification of an external provider's deduplication and query semantics."""

    IDEMPOTENT_BY_KEY = "IDEMPOTENT_BY_KEY"
    STATUS_QUERYABLE = "STATUS_QUERYABLE"
    NON_IDEMPOTENT = "NON_IDEMPOTENT"


class UnknownSideEffectOutcome(Exception):
    """Raised when an uncommitted INTENT is found for a provider that cannot deduplicate or query."""

    def __init__(self, side_effect_key: str, intent_payload: dict[str, Any]) -> None:
        super().__init__(
            f"Side effect {side_effect_key} is in an Ambiguous (INTENT) state on a NON_IDEMPOTENT "
            "provider; automatic retry is forbidden. Route to human or Saga reconciliation."
        )
        self.side_effect_key = side_effect_key
        self.intent_payload = intent_payload


UnknownOutcomeError = UnknownSideEffectOutcome


class AmbiguousEffectReconciliationError(Exception):
    """Raised when a STATUS_QUERYABLE provider cannot resolve an in-flight INTENT record."""


@dataclass(frozen=True, slots=True)
class EffectIdentity:
    """Canonical coordinate identifying a single logical external side effect."""

    workflow_id: str
    graph_digest: str
    node_id: str
    logical_attempt: int
    state_hash: bytes


def _frame(*parts: bytes) -> bytes:
    """Length-prefix each byte field (4-byte big-endian) to prevent concatenation collisions."""
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def idempotency_key(secret: bytes, ident: EffectIdentity) -> str:
    """Compute κ = HMAC-SHA256(secret, domain ‖ workflow ‖ graph ‖ node ‖ attempt ‖ H(S_t))."""
    if len(ident.state_hash) != 32:
        raise ValueError("EffectIdentity.state_hash must be a 32-byte SHA-256 digest.")
    framed_preimage = _frame(
        IDEMPOTENCY_DOMAIN,
        ident.workflow_id.encode("utf-8"),
        ident.graph_digest.encode("utf-8"),
        ident.node_id.encode("utf-8"),
        str(ident.logical_attempt).encode("utf-8"),
        ident.state_hash,
    )
    return hmac.new(secret, framed_preimage, hashlib.sha256).hexdigest()


RECORD_INTENT_SQL = text(
    """
    INSERT INTO checkpoint_writes (
        checkpoint_id,
        write_index,
        workflow_id,
        branch_id,
        node_id,
        channel_name,
        write_state,
        write_payload,
        side_effect_key,
        recorded_response
    )
    VALUES (
        CAST(:checkpoint_id AS uuid),
        CAST(:write_index AS integer),
        CAST(:workflow_id AS uuid),
        :branch_id,
        :node_id,
        :channel_name,
        'INTENT',
        CAST(:write_payload AS jsonb),
        :side_effect_key,
        NULL
    )
    ON CONFLICT (workflow_id, side_effect_key) WHERE side_effect_key IS NOT NULL
    DO NOTHING;
    """
)

READ_EFFECT_ROW_SQL = text(
    """
    SELECT checkpoint_id,
           write_index,
           write_state,
           write_payload,
           recorded_response
      FROM checkpoint_writes
     WHERE workflow_id     = CAST(:workflow_id AS uuid)
       AND side_effect_key = :side_effect_key
     LIMIT 1;
    """
)

CONFIRM_EFFECT_SQL = text(
    """
    UPDATE checkpoint_writes
       SET write_state       = 'COMMITTED',
           recorded_response = CAST(:recorded_response AS jsonb),
           recorded_at       = now()
     WHERE workflow_id     = CAST(:workflow_id AS uuid)
       AND side_effect_key = :side_effect_key
       AND write_state     = 'INTENT'
    RETURNING checkpoint_id, write_index;
    """
)


async def _confirm_effect(
    conn: AsyncConnection,
    workflow_id: str,
    key: str,
    response_payload: dict[str, Any],
) -> None:
    """Transition an INTENT row to COMMITTED and persist the provider response."""
    await conn.execute(
        CONFIRM_EFFECT_SQL,
        {
            "workflow_id": workflow_id,
            "side_effect_key": key,
            "recorded_response": json.dumps(response_payload, sort_keys=True),
        },
    )


async def execute_idempotent_effect(
    engine: AsyncEngine,
    secret: bytes,
    ident: EffectIdentity,
    checkpoint_id: uuid.UUID,
    write_index: int,
    branch_id: str,
    channel_name: str,
    request_payload: dict[str, Any],
    capability: ProviderCapability,
    provider_call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    status_Reconciler: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]] | None = None,
) -> dict[str, Any]:
    """Execute an external side effect across the 3-state physical / 5-state logical lifecycle.

    Mapping between physical storage (`checkpoint_writes.write_state`) and the 5-state lifecycle:
      1. Attempted  -> Phase 1 transaction inserts `write_state = 'INTENT'` (if not already present).
      2. Dispatched -> Phase 2 invokes `provider_call(key, request_payload)` outside any DB transaction.
      3. Confirmed  -> Phase 3 updates `write_state = 'COMMITTED'` with `recorded_response`.
      4. Ambiguous  -> On recovery, discovering a pre-existing `write_state = 'INTENT'` proves a prior
                       worker entered Phase 1 and may have dispatched Phase 2 before crashing.
      5. Reconciled -> Resolved via `status_Reconciler`, idempotent replay (`IDEMPOTENT_BY_KEY`), or
                       halting for human/Saga intervention (`UnknownSideEffectOutcome`).
    """
    key = idempotency_key(secret, ident)

    # Phase 1: Durable INTENT registration or prior-state discovery (short transaction)
    is_preexisting_intent = False
    async with engine.begin() as conn:
        existing = (
            await conn.execute(
                READ_EFFECT_ROW_SQL,
                {"workflow_id": ident.workflow_id, "side_effect_key": key},
            )
        ).mappings().first()

        if existing is not None:
            if existing["write_state"] == "COMMITTED":
                raw_resp = existing["recorded_response"]
                parsed_resp: dict[str, Any] = (
                    json.loads(raw_resp) if isinstance(raw_resp, str) else dict(raw_resp)
                )
                return parsed_resp
            is_preexisting_intent = True
        else:
            await conn.execute(
                RECORD_INTENT_SQL,
                {
                    "checkpoint_id": str(checkpoint_id),
                    "write_index": write_index,
                    "workflow_id": ident.workflow_id,
                    "branch_id": branch_id,
                    "node_id": ident.node_id,
                    "channel_name": channel_name,
                    "write_payload": json.dumps(request_payload, sort_keys=True),
                    "side_effect_key": key,
                },
            )

    # Handle the Ambiguous state if an uncommitted INTENT row already existed prior to this run
    if is_preexisting_intent:
        if capability == ProviderCapability.NON_IDEMPOTENT:
            raise UnknownSideEffectOutcome(side_effect_key=key, intent_payload=request_payload)

        if capability == ProviderCapability.STATUS_QUERYABLE:
            if status_Reconciler is None:
                raise AmbiguousEffectReconciliationError(
                    f"Provider for key {key} is STATUS_QUERYABLE, but no status_Reconciler was supplied."
                )
            observed_response = await status_Reconciler(key, request_payload)
            if observed_response is not None:
                async with engine.begin() as conn:
                    await _confirm_effect(conn, ident.workflow_id, key, observed_response)
                return observed_response

    # Phase 2: Dispatched (outside DB transaction; connection returned to pool)
    response = await provider_call(key, request_payload)

    # Phase 3: Confirmed (short transaction)
    async with engine.begin() as conn:
        await _confirm_effect(conn, ident.workflow_id, key, response)

    return response
