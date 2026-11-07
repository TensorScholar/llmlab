"""Optimistic concurrency control, compare-and-swap transitions, and durable state commits."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

CheckpointStatus = Literal["RUNNING", "SUSPENDED", "COMPLETED", "FAILED"]
WriteState = Literal["APPLIED", "INTENT", "COMMITTED"]


class DurableExecutionError(Exception):
    """Base exception for durable state engine errors."""


class ConcurrencyConflict(DurableExecutionError):
    """Raised when a branch head cannot be advanced due to terminal state or non-commutative conflict."""


class LeaseLost(DurableExecutionError):
    """Raised when the worker's fencing token has been superseded by a newer leaseholder."""

    def __init__(self, worker_token: int, head_token: int) -> None:
        super().__init__(
            f"Worker fencing token {worker_token} is stale compared to branch head token {head_token}."
        )
        self.worker_token = worker_token
        self.head_token = head_token


class RetryFromVersion(DurableExecutionError):
    """Raised when a commutative delta misses CAS and can be rebased onto the new head version."""

    def __init__(
        self,
        current_version: int,
        current_checkpoint_id: uuid.UUID,
        current_state_payload: dict[str, Any],
    ) -> None:
        super().__init__(f"CAS conflict on commutative transition; rebase onto version {current_version}.")
        self.current_version = current_version
        self.current_checkpoint_id = current_checkpoint_id
        self.current_state_payload = current_state_payload


@dataclass(frozen=True, slots=True)
class WriteRecord:
    """Single channel delta or recorded external operation associated with a checkpoint."""

    channel_name: str
    write_payload: dict[str, Any]
    write_state: WriteState = "APPLIED"
    side_effect_key: str | None = None
    recorded_response: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """Encapsulates all inputs required to advance a workflow branch by one logical step."""

    workflow_id: uuid.UUID
    branch_id: str
    expected_version: int
    step: int
    node_id: str
    graph_digest: str
    fencing_token: int
    state_payload: dict[str, Any]
    writes: Sequence[WriteRecord] = field(default_factory=tuple)
    next_status: CheckpointStatus = "RUNNING"
    writes_are_commutative: bool = False
    new_checkpoint_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def canonical_state_json(self) -> str:
        """Deterministically serialize state_payload for hashing and JSONB storage."""
        return json.dumps(self.state_payload, sort_keys=True, separators=(",", ":"))

    def compute_state_hash(self) -> bytes:
        """Return the 32-byte SHA-256 digest of the canonicalized state payload."""
        return hashlib.sha256(self.canonical_state_json().encode("utf-8")).digest()

    def head_params(self) -> dict[str, Any]:
        """Bind parameters for head verification and locking queries."""
        return {
            "workflow_id": str(self.workflow_id),
            "branch_id": self.branch_id,
            "expected_version": self.expected_version,
            "fencing_token": self.fencing_token,
        }


CAS_HEAD_SQL = text(
    """
    WITH advanced AS (
        UPDATE checkpoints AS c
           SET fencing_token = CAST(:fencing_token AS bigint)
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
        RETURNING c.checkpoint_id, c.version, c.fencing_token
    )
    SELECT checkpoint_id, version, fencing_token FROM advanced;
    """
)

READ_HEAD_SQL = text(
    """
    SELECT checkpoint_id,
           version,
           status,
           fencing_token,
           state_payload
      FROM checkpoints
     WHERE workflow_id = CAST(:workflow_id AS uuid)
       AND branch_id   = :branch_id
     ORDER BY version DESC
     LIMIT 1;
    """
)

INSERT_SNAPSHOT_SQL = text(
    """
    INSERT INTO checkpoints (
        checkpoint_id,
        workflow_id,
        branch_id,
        parent_checkpoint_id,
        step,
        version,
        node_id,
        graph_digest,
        status,
        fencing_token,
        state_payload,
        state_hash
    )
    VALUES (
        CAST(:checkpoint_id AS uuid),
        CAST(:workflow_id AS uuid),
        :branch_id,
        CAST(:parent_checkpoint_id AS uuid),
        CAST(:step AS bigint),
        CAST(:version AS bigint),
        :node_id,
        :graph_digest,
        :status,
        CAST(:fencing_token AS bigint),
        CAST(:state_payload AS jsonb),
        :state_hash
    )
    RETURNING checkpoint_id;
    """
)

INSERT_WRITES_SQL = text(
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
        :write_state,
        CAST(:write_payload AS jsonb),
        :side_effect_key,
        CAST(:recorded_response AS jsonb)
    );
    """
)


async def insert_snapshot(
    conn: AsyncConnection,
    tr: Transition,
    parent_id: uuid.UUID | None,
    new_version: int,
) -> uuid.UUID:
    """Insert a new immutable checkpoint snapshot row at new_version."""
    params: dict[str, Any] = {
        "checkpoint_id": str(tr.new_checkpoint_id),
        "workflow_id": str(tr.workflow_id),
        "branch_id": tr.branch_id,
        "parent_checkpoint_id": str(parent_id) if parent_id is not None else None,
        "step": tr.step,
        "version": new_version,
        "node_id": tr.node_id,
        "graph_digest": tr.graph_digest,
        "status": tr.next_status,
        "fencing_token": tr.fencing_token,
        "state_payload": tr.canonical_state_json(),
        "state_hash": tr.compute_state_hash(),
    }
    row = (await conn.execute(INSERT_SNAPSHOT_SQL, params)).mappings().first()
    if row is None:
        raise DurableExecutionError("Failed to insert checkpoint snapshot row.")
    return uuid.UUID(str(row["checkpoint_id"]))


async def insert_writes(
    conn: AsyncConnection,
    tr: Transition,
    checkpoint_id: uuid.UUID,
) -> None:
    """Batch-insert all channel deltas and recorded side-effect entries for checkpoint_id."""
    if not tr.writes:
        return

    payloads: list[dict[str, Any]] = []
    for idx, write in enumerate(tr.writes):
        payloads.append(
            {
                "checkpoint_id": str(checkpoint_id),
                "write_index": idx,
                "workflow_id": str(tr.workflow_id),
                "branch_id": tr.branch_id,
                "node_id": tr.node_id,
                "channel_name": write.channel_name,
                "write_state": write.write_state,
                "write_payload": json.dumps(write.write_payload, sort_keys=True),
                "side_effect_key": write.side_effect_key,
                "recorded_response": (
                    json.dumps(write.recorded_response, sort_keys=True)
                    if write.recorded_response is not None
                    else None
                ),
            }
        )
    await conn.execute(INSERT_WRITES_SQL, payloads)


async def commit_transition(conn: AsyncConnection, tr: Transition) -> uuid.UUID:
    """Atomically verify the branch head via CAS, insert version + 1, and persist writes."""
    async with conn.begin_nested() if conn.in_transaction() else conn.begin():
        row = (await conn.execute(CAS_HEAD_SQL, tr.head_params())).mappings().first()
        if row is None:
            head = (await conn.execute(READ_HEAD_SQL, tr.head_params())).mappings().first()
            if head is None:
                raise ConcurrencyConflict("Target workflow branch does not exist.")
            if head["status"] != "RUNNING":
                raise ConcurrencyConflict(
                    f"Cannot advance branch in state '{head['status']}'; expected 'RUNNING'."
                )
            head_fencing_token = int(head["fencing_token"])
            if tr.fencing_token < head_fencing_token:
                raise LeaseLost(worker_token=tr.fencing_token, head_token=head_fencing_token)

            if not tr.writes_are_commutative:
                raise ConcurrencyConflict(
                    f"Non-commutative delta at expected v={tr.expected_version} "
                    f"cannot rebase onto head v={head['version']}."
                )

            raw_payload = head["state_payload"]
            parsed_payload: dict[str, Any] = (
                json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
            )
            raise RetryFromVersion(
                current_version=int(head["version"]),
                current_checkpoint_id=uuid.UUID(str(head["checkpoint_id"])),
                current_state_payload=parsed_payload,
            )

        parent_checkpoint_id = uuid.UUID(str(row["checkpoint_id"]))
        next_version = int(row["version"]) + 1
        try:
            new_id = await insert_snapshot(
                conn=conn,
                tr=tr,
                parent_id=parent_checkpoint_id,
                new_version=next_version,
            )
            await insert_writes(conn=conn, tr=tr, checkpoint_id=new_id)
            return new_id
        except Exception as exc:
            raise ConcurrencyConflict(f"Concurrent version commit conflict: {exc}") from exc