"""Typed inter-agent message envelopes and server-side capability authorization."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ARGUMENT_BYTES: Final[int] = 8192


class AuthorizationError(Exception):
    """Base exception for inter-agent authorization failures."""


class UnauthorizedTopologyEdge(AuthorizationError):
    """Raised when sender is not permitted to send messages to recipient."""


class UnauthorizedCapabilityError(AuthorizationError):
    """Raised when sender lacks the capability required to invoke the requested tool."""


class ToolCallRequest(BaseModel):
    """Strictly typed, size-bounded payload for an inter-agent tool request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments_json: bytes = Field(max_length=MAX_ARGUMENT_BYTES)

    @field_validator("arguments_json")
    @classmethod
    def _validate_json_object(cls, value: bytes) -> bytes:
        try:
            decoded = json.loads(value.decode("utf-8"))
        except Exception as exc:
            raise ValueError("arguments_json must be valid UTF-8 encoded JSON.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("arguments_json must decode to a JSON object.")
        return value


class AgentEnvelope(BaseModel):
    """Immutable, schema-enforced envelope for all inter-agent communication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope_id: uuid.UUID
    workflow_id: uuid.UUID
    branch_id: str = Field(min_length=1, max_length=128)
    sender: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    recipient: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    step: int = Field(ge=0)
    version: int = Field(ge=1)
    issued_at: datetime
    payload: ToolCallRequest


@dataclass(frozen=True, slots=True)
class AgentRegistration:
    """Server-side identity and capability grant record for an agent node."""

    agent_id: str
    allowed_recipients: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)


class AgentDirectory:
    """Authoritative server-side registry mapping agents to allowed edges and tool capabilities."""

    def __init__(
        self,
        agents: Mapping[str, AgentRegistration],
        tool_capabilities: Mapping[str, str],
    ) -> None:
        self._agents: dict[str, AgentRegistration] = dict(agents)
        self._tool_capabilities: dict[str, str] = dict(tool_capabilities)

    def get_agent(self, agent_id: str) -> AgentRegistration:
        if agent_id not in self._agents:
            raise AuthorizationError(f"Unknown agent identity: '{agent_id}'.")
        return self._agents[agent_id]

    def required_capability(self, tool_name: str) -> str:
        if tool_name not in self._tool_capabilities:
            raise UnauthorizedCapabilityError(
                f"Tool '{tool_name}' is not registered in the capability directory."
            )
        return self._tool_capabilities[tool_name]


def authorize(
    envelope: AgentEnvelope,
    directory: AgentDirectory,
    authenticated_sender: str | None = None,
) -> None:
    """Enforce sender identity, allowed communication topology, and tool capability grants."""
    if authenticated_sender is not None and envelope.sender != authenticated_sender:
        raise AuthorizationError(
            f"Envelope sender '{envelope.sender}' does not match harness identity '{authenticated_sender}'."
        )

    sender_reg = directory.get_agent(envelope.sender)
    directory.get_agent(envelope.recipient)

    if envelope.recipient not in sender_reg.allowed_recipients:
        raise UnauthorizedTopologyEdge(
            f"Agent '{envelope.sender}' is not authorized to message '{envelope.recipient}'."
        )

    required_cap = directory.required_capability(envelope.payload.tool)
    if required_cap not in sender_reg.capabilities:
        raise UnauthorizedCapabilityError(
            f"Agent '{envelope.sender}' lacks capability '{required_cap}' required by tool '{envelope.payload.tool}'."
        )