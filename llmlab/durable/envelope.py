"""Typed inter-agent message envelopes and server-side capability authorization."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AbstractSet, Any, Final, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ARGUMENT_BYTES: Final[int] = 8192


class AuthorizationError(PermissionError):
    """Base exception for inter-agent authorization failures."""


class UnauthorizedTopologyEdge(AuthorizationError):
    """Raised when sender is not permitted to send messages to recipient."""


class UnauthorizedCapabilityError(AuthorizationError):
    """Raised when sender lacks the capability required to invoke the requested tool."""


class ToolCallRequest(BaseModel):
    """Strictly typed, size-bounded payload for an inter-agent tool request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments_json: str | bytes = Field(...)

    @field_validator("arguments_json")
    @classmethod
    def enforce_utf8_byte_limit(cls, value: str | bytes) -> str | bytes:
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            raw_str = value
        elif isinstance(value, bytes):
            encoded = value
            try:
                raw_str = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("arguments_json must be valid UTF-8.") from exc
        else:
            raise TypeError("arguments_json must be str or bytes.")

        if len(encoded) > MAX_ARGUMENT_BYTES:
            raise ValueError(
                f"Serialized arguments exceed MAX_ARGUMENT_BYTES "
                f"({len(encoded)} > {MAX_ARGUMENT_BYTES})."
            )

        try:
            decoded = json.loads(raw_str)
        except Exception as exc:
            raise ValueError("arguments_json must be valid UTF-8 encoded JSON.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("arguments_json must decode to a JSON object.")

        return value

    _validate_json_object = enforce_utf8_byte_limit


class AgentEnvelope(BaseModel):
    """Immutable, schema-enforced envelope for all inter-agent communication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope_id: uuid.UUID
    workflow_id: uuid.UUID
    branch_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    step: int = Field(ge=0)
    version: int = Field(ge=1)
    sender: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    recipient: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    issued_at: datetime
    payload: ToolCallRequest


@dataclass(frozen=True, slots=True)
class AgentRegistration:
    """Server-side identity and capability grant record for an agent node."""

    agent_id: str
    allowed_recipients: AbstractSet[str] = field(default_factory=frozenset)
    capabilities: AbstractSet[str] = field(default_factory=frozenset)
    granted_capabilities: AbstractSet[str] = field(default_factory=frozenset)

    def __init__(
        self,
        agent_id: str,
        allowed_recipients: AbstractSet[str] = frozenset(),
        capabilities: AbstractSet[str] | None = None,
        granted_capabilities: AbstractSet[str] | None = None,
    ) -> None:
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "allowed_recipients", frozenset(allowed_recipients))
        caps = (
            granted_capabilities
            if granted_capabilities is not None
            else (capabilities if capabilities is not None else frozenset())
        )
        object.__setattr__(self, "capabilities", frozenset(caps))
        object.__setattr__(self, "granted_capabilities", frozenset(caps))


@dataclass(frozen=True, slots=True)
class AgentDirectory:
    """Authoritative server-side registry mapping agents to allowed edges and tool capabilities."""

    agents: Mapping[str, AgentRegistration]
    tool_required_capability: Mapping[str, str] = field(default_factory=dict)
    tool_capabilities: Mapping[str, str] = field(default_factory=dict)

    def __init__(
        self,
        agents: Mapping[str, AgentRegistration],
        tool_capabilities: Mapping[str, str] | None = None,
        tool_required_capability: Mapping[str, str] | None = None,
    ) -> None:
        object.__setattr__(self, "agents", dict(agents))
        tools = (
            tool_required_capability
            if tool_required_capability is not None
            else (tool_capabilities or {})
        )
        object.__setattr__(self, "tool_required_capability", dict(tools))
        object.__setattr__(self, "tool_capabilities", dict(tools))

    def get_agent(self, agent_id: str) -> AgentRegistration:
        if agent_id not in self.agents:
            raise AuthorizationError(f"Unknown agent identity: '{agent_id}'.")
        return self.agents[agent_id]

    def required_capability(self, tool_name: str) -> str:
        if tool_name not in self.tool_required_capability:
            raise UnauthorizedCapabilityError(
                f"Tool '{tool_name}' is not registered in the capability directory."
            )
        return self.tool_required_capability[tool_name]


def authorize(
    envelope: AgentEnvelope,
    directory: AgentDirectory | None = None,
    *,
    authenticated_sender: str | None = None,
    **kwargs: Any,
) -> None:
    """Enforce sender identity, allowed communication topology, and tool capability grants."""
    target_directory = directory if directory is not None else kwargs.get("directory")
    if target_directory is None:
        raise ValueError("AgentDirectory must be provided to authorize().")

    auth_sender = authenticated_sender or kwargs.get("authenticated_sender")
    if auth_sender is not None and envelope.sender != auth_sender:
        raise AuthorizationError(
            f"Sender spoofing detected: envelope claims '{envelope.sender}', "
            f"but harness authenticated '{auth_sender}'."
        )

    sender_reg = target_directory.agents.get(envelope.sender)
    if sender_reg is None:
        raise AuthorizationError(f"Unregistered sender agent '{envelope.sender}'.")

    if envelope.recipient not in target_directory.agents:
        raise AuthorizationError(f"Unregistered recipient agent '{envelope.recipient}'.")

    if envelope.recipient not in sender_reg.allowed_recipients:
        raise UnauthorizedTopologyEdge(
            f"Agent '{envelope.sender}' is not authorized to message '{envelope.recipient}'."
        )

    if envelope.payload.tool not in target_directory.tool_required_capability:
        raise UnauthorizedCapabilityError(
            f"Tool '{envelope.payload.tool}' is not registered in the capability directory."
        )

    required_cap = target_directory.tool_required_capability[envelope.payload.tool]
    if (
        required_cap not in sender_reg.capabilities
        and required_cap not in sender_reg.granted_capabilities
    ):
        raise UnauthorizedCapabilityError(
            f"Agent '{envelope.sender}' lacks capability '{required_cap}' required by tool '{envelope.payload.tool}'."
        )
