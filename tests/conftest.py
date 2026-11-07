"""Pytest fixtures for async PostgreSQL database sessions and genesis state seeding."""
from __future__ import annotations

import os
import pathlib
import uuid
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from llmlab.durable.cas import Transition, insert_snapshot

DEFAULT_DATABASE_URL = "postgresql+asyncpg://llmlab:llmlab_password@localhost:5432/llmlab"
SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "01_schema.sql"
VALID_GRAPH_DIGEST = "a" * 64


@pytest.fixture(scope="session")
def database_url() -> str:
    """Return the PostgreSQL connection URL from environment or default local compose."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


@pytest_asyncio.fixture
async def db_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """Create an AsyncEngine, ensure schema exists, and clean tables before each test."""
    engine = create_async_engine(database_url, echo=False, pool_size=10, max_overflow=20)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    async with engine.begin() as conn:
        raw_conn = await conn.get_raw_connection()
        assert raw_conn.driver_connection is not None
        await raw_conn.driver_connection.execute(schema_sql)
        await conn.execute(
            text("TRUNCATE TABLE checkpoint_writes, checkpoints RESTART IDENTITY CASCADE;")
        )
        await conn.execute(text("ALTER SEQUENCE fencing_token_seq RESTART WITH 1;"))

    try:
        yield engine
    finally:
        await engine.dispose()


async def seed_genesis_checkpoint(
    conn: AsyncConnection,
    workflow_id: uuid.UUID,
    branch_id: str = "main",
    fencing_token: int = 1,
    initial_state: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Insert a valid root checkpoint (step=0, version=1, parent_checkpoint_id=None)."""
    genesis = Transition(
        workflow_id=workflow_id,
        branch_id=branch_id,
        expected_version=0,
        step=0,
        node_id="__start__",
        graph_digest=VALID_GRAPH_DIGEST,
        fencing_token=fencing_token,
        state_payload=initial_state if initial_state is not None else {"step_count": 0, "tokens": 0},
    )
    async with conn.begin_nested() if conn.in_transaction() else conn.begin():
        await conn.execute(
            text("SELECT setval('fencing_token_seq', GREATEST(CAST(:ft AS bigint), 1), true);"),
            {"ft": fencing_token},
        )
        return await insert_snapshot(conn=conn, tr=genesis, parent_id=None, new_version=1)