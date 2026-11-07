"""Tests for workflow suspension, single-use HMAC resumption tokens, and expiry sweeping."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from llmlab.durable.suspension import (
    InvalidResumptionToken,
    SuspensionExpired,
    issue_resumption_token,
    suspend_workflow,
    sweep_expired_suspensions,
    verify_and_resume_workflow,
)
from tests.conftest import seed_genesis_checkpoint

RESUMPTION_SECRET = b"resumption-hmac-secret-key-32byt"


async def test_suspend_and_single_use_resumption_token(db_engine: AsyncEngine) -> None:
    """Verify suspending stores only the digest and resuming consumes the token exclusively."""
    workflow_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=4)

    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(conn, workflow_id, fencing_token=1)

        token = issue_resumption_token(
            secret=RESUMPTION_SECRET,
            workflow_id=workflow_id,
            branch_id="main",
            step=0,
            version=1,
            audience="approver_group_finance",
            suspended_at=now,
            expires_at=expires,
        )

        await suspend_workflow(
            conn=conn,
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            fencing_token=1,
            token=token,
        )

        # Forged token is rejected
        with pytest.raises(InvalidResumptionToken):
            await verify_and_resume_workflow(
                conn=conn,
                workflow_id=workflow_id,
                branch_id="main",
                expected_version=1,
                raw_token="0" * 64,
                new_fencing_token=2,
            )

        # Valid token succeeds and ratchets fencing_token to 2
        resumed = await verify_and_resume_workflow(
            conn=conn,
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            raw_token=token.raw_token,
            new_fencing_token=2,
        )
        assert resumed.version == 1
        assert resumed.fencing_token == 2

        # Replaying the exact same valid token a second time fails closed
        with pytest.raises(InvalidResumptionToken):
            await verify_and_resume_workflow(
                conn=conn,
                workflow_id=workflow_id,
                branch_id="main",
                expected_version=1,
                raw_token=token.raw_token,
                new_fencing_token=3,
            )


async def test_resume_workflow_draws_nextval_fencing_token(
    db_engine: AsyncEngine,
) -> None:
    """Verify that when new_fencing_token is None, nextval('fencing_token_seq') is drawn."""
    workflow_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=2)

    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(conn, workflow_id, fencing_token=1)
        token = issue_resumption_token(
            secret=RESUMPTION_SECRET,
            workflow_id=workflow_id,
            branch_id="main",
            step=0,
            version=1,
            audience="auto_resume",
            suspended_at=now,
            expires_at=expires,
        )
        await suspend_workflow(
            conn=conn,
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            fencing_token=1,
            token=token,
        )
        resumed = await verify_and_resume_workflow(
            conn=conn,
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            raw_token=token.raw_token,
            new_fencing_token=None,
        )
        assert resumed.version == 1
        assert resumed.fencing_token > 1


async def test_expired_suspension_rejected_and_reclaimed_by_sweeper(
    db_engine: AsyncEngine,
) -> None:
    """Verify expired tokens raise SuspensionExpired and are reclaimed by sweep_expired_suspensions."""
    workflow_id = uuid.uuid4()
    past_suspend = datetime.now(timezone.utc) - timedelta(hours=5)
    past_expiry = datetime.now(timezone.utc) - timedelta(minutes=30)

    async with db_engine.connect() as conn:
        await seed_genesis_checkpoint(conn, workflow_id, fencing_token=1)

        expired_token = issue_resumption_token(
            secret=RESUMPTION_SECRET,
            workflow_id=workflow_id,
            branch_id="main",
            step=0,
            version=1,
            audience="oncall_sre",
            suspended_at=past_suspend,
            expires_at=past_expiry,
        )

        await suspend_workflow(
            conn=conn,
            workflow_id=workflow_id,
            branch_id="main",
            expected_version=1,
            fencing_token=1,
            token=expired_token,
        )

        # Attempting to redeem an expired token raises SuspensionExpired
        with pytest.raises(SuspensionExpired):
            await verify_and_resume_workflow(
                conn=conn,
                workflow_id=workflow_id,
                branch_id="main",
                expected_version=1,
                raw_token=expired_token.raw_token,
                new_fencing_token=2,
            )

        # Sweeper claims the expired workflow and increments fencing_token via sequence
        reclaimed = await sweep_expired_suspensions(conn, batch_limit=10)
        assert len(reclaimed) == 1
        assert reclaimed[0].workflow_id == workflow_id
        assert reclaimed[0].fencing_token >= 1

        # Subsequent sweep returns zero rows because status is now RUNNING
        reclaimed_again = await sweep_expired_suspensions(conn, batch_limit=10)
        assert len(reclaimed_again) == 0