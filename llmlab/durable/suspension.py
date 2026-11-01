"""Zero-compute workflow suspension, HMAC resumption tokens, and expiry sweeping."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from llmlab.durable.idempotency import _frame

RESUMPTION_DOMAIN: Final[bytes] = b"llmlab/resumption-token/v1"


class SuspensionError(Exception):
    """Base error for workflow suspension and resumption operations."""


class InvalidResumptionToken(SuspensionError):
    """Raised when a resumption token is malformed, forged, already redeemed, or version-mismatched."""


class SuspensionExpired(SuspensionError):
    """Raised when a valid resumption token is presented after its expiration timestamp."""


@dataclass(frozen=True, slots=True)
class IssuedResumptionToken:
    """Pair of raw bearer token (sent to human reviewer) and its SHA-256 digest (stored in DB)."""

    raw_token: str
    token_digest: bytes
    suspended_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ResumedWorkflowState:
    """Result returned when a suspended workflow is atomically resumed via CAS."""

    checkpoint_id: uuid.UUID
    workflow_id: uuid.UUID
    branch_id: str
    version: int
    fencing_token: int
    state_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExpiredSuspensionRecord:
    """Metadata of a suspended workflow reclaimed by the expiration sweeper."""

    checkpoint_id: uuid.UUID
    workflow_id: uuid.UUID
    branch_id: str
    version: int
    fencing_token: int
    state_payload: dict[str, Any]


def issue_resumption_token(
    secret: bytes,
    workflow_id: uuid.UUID,
    branch_id: str,
    step: int,
    version: int,
    audience: str,
    suspended_at: datetime,
    expires_at: datetime,
    nonce: bytes | None = None,
) -> IssuedResumptionToken:
    """Generate ρ = HMAC-SHA256(K, domain ‖ workflow ‖ branch ‖ step ‖ v ‖ aud ‖ τ_susp ‖ τ_exp ‖ n)."""
    if expires_at <= suspended_at:
        raise ValueError("expires_at must be strictly greater than suspended_at.")
    entropy = nonce if nonce is not None else os.urandom(16)
    if len(entropy) < 16:
        raise ValueError("Resumption token nonce must be at least 16 bytes.")

    preimage = _frame(
        RESUMPTION_DOMAIN,
        str(workflow_id).encode("utf-8"),
        branch_id.encode("utf-8"),
        str(step).encode("utf-8"),
        str(version).encode("utf-8"),
        audience.encode("utf-8"),
        suspended_at.isoformat().encode("utf-8"),
        expires_at.isoformat().encode("utf-8"),
        entropy,
    )
    raw_token = hmac.new(secret, preimage, hashlib.sha256).hexdigest()
    token_digest = hashlib.sha256(raw_token.encode("ascii")).digest()
    return IssuedResumptionToken(
        raw_token=raw_token,
        token_digest=token_digest,
        suspended_at=suspended_at,
        expires_at=expires_at,
    )


SUSPEND_HEAD_SQL = text(
    """
    UPDATE checkpoints AS c
       SET status              = 'SUSPENDED',
           suspended_at        = CAST(:suspended_at AS timestamptz),
           resume_token_digest = :resume_token_digest,
           expires_at          = CAST(:expires_at AS timestamptz),
           fencing_token       = CAST(:fencing_token AS bigint)
     WHERE c.workflow_id   = CAST(:workflow_id AS uuid)
       AND c.branch_id     = :branch_id
       AND c.version       = CAST(:expected_version AS bigint)
       AND c.status        = 'RUNNING'
       AND c.fencing_token <= CAST(:fencing_token AS bigint)
       AND NOT EXISTS (
             SELECT 1
               FROM checkpoints AS newer
              WHERE newer.workflow_id = c.workflow_id
                AND newer.branch_id   = c.branch_id
                AND newer.version     > c.version
           )
    RETURNING c.checkpoint_id, c.version;
    """
)

RESUME_CAS_SQL = text(
    """
    UPDATE checkpoints AS c
       SET status              = 'RUNNING',
           suspended_at        = NULL,
           resume_token_digest = NULL,
           expires_at          = NULL,
           fencing_token       = CAST(:new_fencing_token AS bigint)
     WHERE c.workflow_id         = CAST(:workflow_id AS uuid)
       AND c.branch_id           = :branch_id
       AND c.version             = CAST(:expected_version AS bigint)
       AND c.status              = 'SUSPENDED'
       AND c.resume_token_digest = :candidate_digest
       AND c.expires_at          > now()
       AND c.fencing_token       < CAST(:new_fencing_token AS bigint)
    RETURNING c.checkpoint_id, c.version, c.fencing_token, c.state_payload;
    """
)

INSPECT_SUSPENDED_HEAD_SQL = text(
    """
    SELECT status,
           resume_token_digest,
           expires_at,
           fencing_token,
            (expires_at <= now()) AS is_expired
      FROM checkpoints
     WHERE workflow_id = CAST(:workflow_id AS uuid)
       AND branch_id   = :branch_id
       AND version     = CAST(:expected_version AS bigint)
     LIMIT 1;
    """
)

SWEEP_EXPIRED_SUSPENSIONS_SQL = text(
    """
    WITH expired_candidates AS (
        SELECT c.checkpoint_id
          FROM checkpoints AS c
         WHERE c.status     = 'SUSPENDED'
           AND c.expires_at <= now()
         ORDER BY c.expires_at ASC
         LIMIT CAST(:batch_limit AS integer)
           FOR UPDATE SKIP LOCKED
    )
    UPDATE checkpoints AS target
       SET status              = 'RUNNING',
           suspended_at        = NULL,
           resume_token_digest = NULL,
           expires_at          = NULL,
           fencing_token       = nextval('fencing_token_seq')
      FROM expired_candidates AS ec
     WHERE target.checkpoint_id = ec.checkpoint_id
    RETURNING target.checkpoint_id,
              target.workflow_id,
              target.branch_id,
              target.version,
              target.fencing_token,
              target.state_payload;
    """
)


async def suspend_workflow(
    conn: AsyncConnection,
    workflow_id: uuid.UUID,
    branch_id: str,
    expected_version: int,
    fencing_token: int,
    token: IssuedResumptionToken,
) -> uuid.UUID:
    """Transition the current RUNNING branch head to SUSPENDED and store the token digest."""
    async with conn.begin_nested() if conn.in_transaction() else conn.begin():
        row = (
            await conn.execute(
                SUSPEND_HEAD_SQL,
                {
                    "workflow_id": str(workflow_id),
                    "branch_id": branch_id,
                    "expected_version": expected_version,
                    "fencing_token": fencing_token,
                    "suspended_at": token.suspended_at,
                    "resume_token_digest": token.token_digest,
                    "expires_at": token.expires_at,
                },
            )
        ).mappings().first()
        if row is None:
            raise SuspensionError(
                f"Failed to suspend workflow {workflow_id} on branch '{branch_id}' at v={expected_version}."
            )
        return uuid.UUID(str(row["checkpoint_id"]))


async def verify_and_resume_workflow(
    conn: AsyncConnection,
    workflow_id: uuid.UUID,
    branch_id: str,
    expected_version: int,
    raw_token: str,
    new_fencing_token: int,
) -> ResumedWorkflowState:
    """Verify the bearer token in constant time and atomically resume the SUSPENDED checkpoint."""
    candidate_digest = hashlib.sha256(raw_token.encode("ascii")).digest()

    async with conn.begin_nested() if conn.in_transaction() else conn.begin():
        head = (
            await conn.execute(
                INSPECT_SUSPENDED_HEAD_SQL,
                {
                    "workflow_id": str(workflow_id),
                    "branch_id": branch_id,
                    "expected_version": expected_version,
                },
            )
        ).mappings().first()

        if head is None or head["status"] != "SUSPENDED" or head["resume_token_digest"] is None:
            raise InvalidResumptionToken("Target checkpoint is not in a SUSPENDED state.")

        stored_digest = bytes(head["resume_token_digest"])
        if not hmac.compare_digest(stored_digest, candidate_digest):
            raise InvalidResumptionToken("Resumption token digest mismatch.")

        if bool(head["is_expired"]):
            raise SuspensionExpired("Resumption token has passed its expires_at deadline.")

        row = (
            await conn.execute(
                RESUME_CAS_SQL,
                {
                    "workflow_id": str(workflow_id),
                    "branch_id": branch_id,
                    "expected_version": expected_version,
                    "candidate_digest": candidate_digest,
                    "new_fencing_token": new_fencing_token,
                },
            )
        ).mappings().first()

        if row is None:
            raise InvalidResumptionToken(
                "Concurrent resumption or stale fencing token prevented workflow resumption."
            )

        raw_payload = row["state_payload"]
        parsed_payload: dict[str, Any] = (
            json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
        )
        return ResumedWorkflowState(
            checkpoint_id=uuid.UUID(str(row["checkpoint_id"])),
            workflow_id=workflow_id,
            branch_id=branch_id,
            version=int(row["version"]),
            fencing_token=int(row["fencing_token"]),
            state_payload=parsed_payload,
        )


async def sweep_expired_suspensions(
    conn: AsyncConnection,
    batch_limit: int = 100,
) -> Sequence[ExpiredSuspensionRecord]:
    """Claim up to batch_limit expired suspensions using FOR UPDATE SKIP LOCKED and ratcheted fencing."""
    async with conn.begin_nested() if conn.in_transaction() else conn.begin():
        rows = (
            await conn.execute(SWEEP_EXPIRED_SUSPENSIONS_SQL, {"batch_limit": batch_limit})
        ).mappings().all()

        reclaimed: list[ExpiredSuspensionRecord] = []
        for row in rows:
            raw_payload = row["state_payload"]
            parsed_payload: dict[str, Any] = (
                json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
            )
            reclaimed.append(
                ExpiredSuspensionRecord(
                    checkpoint_id=uuid.UUID(str(row["checkpoint_id"])),
                    workflow_id=uuid.UUID(str(row["workflow_id"])),
                    branch_id=str(row["branch_id"]),
                    version=int(row["version"]),
                    fencing_token=int(row["fencing_token"]),
                    state_payload=parsed_payload,
                )
            )
        return reclaimed