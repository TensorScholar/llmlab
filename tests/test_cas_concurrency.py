"""Tests for optimistic concurrency control, fencing tokens, and commutative rebasing."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from llmlab.durable.cas import (
    ConcurrencyConflict,
    LeaseLost,
    RetryFromVersion,
    Transition,
    WriteRecord,
    commit_transition,
)
from tests.conftest import VALID_GRAPH_DIGEST, seed_genesis_checkpoint

VALID_UUIDV7_LITERAL = "0192f3a4-7c1b-7ae2-9d34-000000000001"


async def test_sequential_transition_commits_atomically(db_engine: AsyncEngine) -> None:
    """Verify a valid transition advances version to 2 and persists its channel writes."""
    workflow_id = uuid.UUID(VALID_UUIDV7_LITERAL)
    async with db_engine.connect() as conn:
        root_id = await seed_genesis_checkpoint(conn, workflow_id, fencing_token=1)

        tr = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="plan_node",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=1,
            state_payload={"step_count": 1, "tokens": 120},
            writes=[
                WriteRecord(
                    channel_name="step_count",
                    write_payload={"delta": 1},
                ),
                WriteRecord(
                    channel_name="tokens",
                    write_payload={"delta": 120},
                ),
            ],
        )
        ckpt_v2 = await commit_transition(conn, tr)

        head = (
            await conn.execute(
                text(
                    "SELECT checkpoint_id, parent_checkpoint_id, version, step, fencing_token "
                    "FROM checkpoints WHERE workflow_id = :wf ORDER BY version DESC LIMIT 1"
                ),
                {"wf": str(workflow_id)},
            )
        ).mappings().one()

        assert uuid.UUID(str(head["checkpoint_id"])) == ckpt_v2
        assert uuid.UUID(str(head["parent_checkpoint_id"])) == root_id
        assert int(head["version"]) == 2
        assert int(head["step"]) == 1

        write_count = (
            await conn.execute(
                text("SELECT count(*) FROM checkpoint_writes WHERE checkpoint_id = :cid"),
                {"cid": str(ckpt_v2)},
            )
        ).scalar_one()
        assert int(write_count) == 2


async def test_concurrent_non_commutative_race_allows_exactly_one_winner(
    db_engine: AsyncEngine,
) -> None:
    """Verify that when two workers race on version=1, exactly one commits version=2."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(conn, workflow_id, fencing_token=5)

    async def worker_attempt(worker_name: str) -> uuid.UUID:
        async with db_engine.connect() as worker_conn:
            tr = Transition(
                workflow_id=workflow_id,
                branch_id="main",
                expected_version=1,
                step=1,
                node_id=worker_name,
                graph_digest=VALID_GRAPH_DIGEST,
                fencing_token=5,
                state_payload={"step_count": 1, "winner": worker_name},
                writes=[WriteRecord(channel_name="winner", write_payload={"node": worker_name})],
                writes_are_commutative=False,
            )
            return await commit_transition(worker_conn, tr)

    results = await asyncio.gather(
        worker_attempt("worker_a"),
        worker_attempt("worker_b"),
        return_exceptions=True,
    )

    successes = [r for r in results if isinstance(r, uuid.UUID)]
    conflicts = [r for r in results if isinstance(r, ConcurrencyConflict)]

    assert len(successes) == 1
    assert len(conflicts) == 1


async def test_stale_fencing_token_raises_lease_lost(db_engine: AsyncEngine) -> None:
    """Verify a zombie worker presenting a lower fencing token is rejected with LeaseLost."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(conn, workflow_id, fencing_token=10)

        # Worker B (new leaseholder with token 11) advances the branch to version 2
        tr_new_leader = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="worker_b",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=11,
            state_payload={"step_count": 1, "leader": "B"},
        )
        await commit_transition(conn, tr_new_leader)

        # Zombie Worker A wakes up with stale token 10 and attempts to commit
        tr_zombie = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="worker_a",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=10,
            state_payload={"step_count": 1, "leader": "A"},
        )
        with pytest.raises(LeaseLost) as exc_info:
            await commit_transition(conn, tr_zombie)

        assert exc_info.value.worker_token == 10
        assert exc_info.value.head_token == 11


async def test_commutative_transition_raises_retry_from_version_and_rebases(
    db_engine: AsyncEngine,
) -> None:
    """Verify a commutative fan-out worker can catch RetryFromVersion and cleanly rebase."""
    workflow_id = uuid.uuid4()
    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(
            conn,
            workflow_id,
            fencing_token=1,
            initial_state={"visited_docs": ["doc_0"]},
        )

        # Branch 1 commits version 2 first
        tr_branch_1 = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="search_branch_1",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=1,
            state_payload={"visited_docs": ["doc_0", "doc_1"]},
            writes=[WriteRecord(channel_name="visited_docs", write_payload={"add": ["doc_1"]})],
            writes_are_commutative=True,
        )
        await commit_transition(conn, tr_branch_1)

        # Branch 2 tries to commit against expected_version=1 with writes_are_commutative=True
        tr_branch_2_stale = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="search_branch_2",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=1,
            state_payload={"visited_docs": ["doc_0", "doc_2"]},
            writes=[WriteRecord(channel_name="visited_docs", write_payload={"add": ["doc_2"]})],
            writes_are_commutative=True,
        )

        with pytest.raises(RetryFromVersion) as retry_exc:
            await commit_transition(conn, tr_branch_2_stale)

        assert retry_exc.value.current_version == 2
        merged_docs = sorted(
            set(retry_exc.value.current_state_payload["visited_docs"]) | {"doc_2"}
        )

        # Rebase onto version 2 using the set-union semilattice reducer
        tr_branch_2_rebased = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=retry_exc.value.current_version,
            step=2,
            node_id="search_branch_2",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=1,
            state_payload={"visited_docs": merged_docs},
            writes=[WriteRecord(channel_name="visited_docs", write_payload={"add": ["doc_2"]})],
            writes_are_commutative=True,
        )
        ckpt_v3 = await commit_transition(conn, tr_branch_2_rebased)
        assert ckpt_v3 is not None
        assert merged_docs == ["doc_0", "doc_1", "doc_2"]


async def test_uuidv7_transition_commits_with_valid_hex_literal(
    db_engine: AsyncEngine,
) -> None:
    """Verify a valid hexadecimal UUIDv7 literal commits without syntax errors or cast failures."""
    workflow_id = uuid.UUID(VALID_UUIDV7_LITERAL)
    async with db_engine.connect() as conn:
        root_id = await seed_genesis_checkpoint(conn, workflow_id, fencing_token=1)
        tr = Transition(
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            step=1,
            node_id="validate_uuidv7",
            graph_digest=VALID_GRAPH_DIGEST,
            fencing_token=1,
            state_payload={"uuid_verified": True},
        )
        ckpt = await commit_transition(conn, tr)
        assert isinstance(ckpt, uuid.UUID)
        assert ckpt != root_id
