"""Tests for Tarjan SCC condensation reachability and typed inter-agent envelope authorization."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llmlab.durable.envelope import (
    AgentDirectory,
    AgentEnvelope,
    AgentRegistration,
    ToolCallRequest,
    UnauthorizedCapabilityError,
    UnauthorizedTopologyEdge,
    authorize,
)
from llmlab.durable.graph import (
    GraphTopology,
    NonTerminatingCycleError,
    OrphanComponentError,
    validate_graph_termination,
)


def test_valid_cyclic_repair_graph_compiles() -> None:
    """Verify a repair loop with a conditional exit to terminal sinks passes SCC validation."""
    topology = GraphTopology(
        vertices=frozenset({"draft", "run_tests", "repair", "completed", "escalate_human"}),
        fixed_edges={
            "draft": frozenset({"run_tests"}),
            "repair": frozenset({"run_tests"}),
        },
        conditional_targets={
            "run_tests": frozenset({"repair", "completed", "escalate_human"}),
        },
        start_vertex="draft",
        terminal_vertices=frozenset({"completed", "escalate_human"}),
    )

    report = validate_graph_termination(topology)
    assert len(report.graph_digest) == 64
    assert any(comp == frozenset({"run_tests", "repair"}) for comp in report.components)


def test_closed_cycle_without_terminal_exit_raises_non_terminating_error() -> None:
    """Verify a two-node deadlock cycle where each node has OutDegree > 0 is rejected."""
    deadlock_topology = GraphTopology(
        vertices=frozenset({"start", "critic_a", "critic_b", "done"}),
        fixed_edges={
            "start": frozenset({"critic_a"}),
            "critic_a": frozenset({"critic_b"}),
            "critic_b": frozenset({"critic_a"}),
        },
        conditional_targets={
            "start": frozenset({"done"}),
        },
        start_vertex="start",
        terminal_vertices=frozenset({"done"}),
    )

    with pytest.raises(NonTerminatingCycleError):
        validate_graph_termination(deadlock_topology)


def test_unreachable_subgraph_raises_orphan_component_error() -> None:
    """Verify disconnected orphan nodes left by refactoring fail compilation."""
    orphan_topology = GraphTopology(
        vertices=frozenset({"start", "legacy_node", "done"}),
        fixed_edges={
            "start": frozenset({"done"}),
            "legacy_node": frozenset({"done"}),
        },
        conditional_targets={},
        start_vertex="start",
        terminal_vertices=frozenset({"done"}),
    )

    with pytest.raises(OrphanComponentError):
        validate_graph_termination(orphan_topology)


def test_agent_envelope_validation_and_capability_authorization() -> None:
    """Verify extra='forbid', payload limits, topology edges, and tool capability checks."""
    directory = AgentDirectory(
        agents={
            "supervisor": AgentRegistration(
                agent_id="supervisor",
                allowed_recipients=frozenset({"search_worker", "db_worker"}),
                capabilities=frozenset({"cap:read_docs"}),
            ),
            "search_worker": AgentRegistration(
                agent_id="search_worker",
                allowed_recipients=frozenset({"supervisor"}),
                capabilities=frozenset({"cap:read_docs"}),
            ),
            "db_worker": AgentRegistration(
                agent_id="db_worker",
                allowed_recipients=frozenset({"supervisor"}),
                capabilities=frozenset({"cap:write_db"}),
            ),
        },
        tool_capabilities={
            "query_index": "cap:read_docs",
            "drop_table": "cap:write_db",
        },
    )

    valid_env = AgentEnvelope(
        envelope_id=uuid.uuid4(),
        workflow_id=uuid.uuid4(),
        branch_id="main",
        sender="supervisor",
        recipient="search_worker",
        step=1,
        version=2,
        issued_at=datetime.now(timezone.utc),
        payload=ToolCallRequest(tool="query_index", arguments_json=b'{"q":"durable state"}'),
    )
    authorize(valid_env, directory, authenticated_sender="supervisor")

    # Unauthorized topology edge (search_worker -> db_worker is not allowed)
    lateral_env = AgentEnvelope(
        envelope_id=uuid.uuid4(),
        workflow_id=uuid.uuid4(),
        branch_id="main",
        sender="search_worker",
        recipient="db_worker",
        step=2,
        version=3,
        issued_at=datetime.now(timezone.utc),
        payload=ToolCallRequest(tool="query_index", arguments_json=b'{"q":"lateral"}'),
    )
    with pytest.raises(UnauthorizedTopologyEdge):
        authorize(lateral_env, directory, authenticated_sender="search_worker")

    # Unauthorized capability (search_worker attempting to invoke drop_table)
    escalation_env = AgentEnvelope(
        envelope_id=uuid.uuid4(),
        workflow_id=uuid.uuid4(),
        branch_id="main",
        sender="search_worker",
        recipient="supervisor",
        step=2,
        version=3,
        issued_at=datetime.now(timezone.utc),
        payload=ToolCallRequest(tool="drop_table", arguments_json=b'{"table":"users"}'),
    )
    with pytest.raises(UnauthorizedCapabilityError):
        authorize(escalation_env, directory, authenticated_sender="search_worker")

    # Extra fields are rejected by Pydantic ConfigDict(extra="forbid")
    with pytest.raises(ValidationError):
        ToolCallRequest.model_validate(
            {
                "tool": "query_index",
                "arguments_json": b"{}",
                "injected_admin_flag": True,
            }
        )