"""The Five Distributed Systems Contracts and the Payment-Enabled Enterprise Assistant harness."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from llmlab.durable.envelope import AgentDirectory, AgentEnvelope, ToolCallRequest, authorize
from llmlab.durable.idempotency import EffectIdentity, ProviderCapability, idempotency_key
from llmlab.tracing.budgets import MemoryBudget


class ContractViolationError(RuntimeError):
    """Base error raised when a workflow transition violates one of the five system contracts."""


class ResourceContractViolation(ContractViolationError):
    """Raised when an iteration violates HBM capacity or the step token-budget ceiling."""


class DeadlineContractViolation(ContractViolationError):
    """Raised when remaining deadline slack is insufficient to safely dispatch a downstream step."""


class StateContractViolation(ContractViolationError):
    """Raised when a state transition violates version monotonicity or fencing-token ownership."""


class EffectContractViolation(ContractViolationError):
    """Raised when an external mutation lacks a committed INTENT record or hits an Ambiguous timeout."""


class AuthorityContractViolation(ContractViolationError):
    """Raised when an inter-agent tool invocation fails sender or capability authorization."""


@dataclass(frozen=True, slots=True)
class ResourceContract:
    """Contract 1: Byte-exact HBM KV admission and unified step token-budget governance."""

    memory_budget: MemoryBudget
    max_step_tokens: int
    max_decode_step_ms: float
    active_parameters: int

    def validate_step_admission(
        self,
        *,
        active_decode_tokens: int,
        prefill_chunk_tokens: int,
        resident_kv_tokens: int,
    ) -> float:
        """Verify token budget, HBM footprint, and Roofline step duration against decode SLO."""
        total_step_tokens = active_decode_tokens + prefill_chunk_tokens
        if total_step_tokens > self.max_step_tokens:
            raise ResourceContractViolation(
                f"Step token count {total_step_tokens} (decode={active_decode_tokens} + "
                f"prefill_chunk={prefill_chunk_tokens}) exceeds B_tokens={self.max_step_tokens}."
            )
        try:
            self.memory_budget.verify_hbm_admission(
                resident_kv_tokens=resident_kv_tokens + prefill_chunk_tokens,
                step_tokens=total_step_tokens,
            )
        except Exception as exc:
            raise ResourceContractViolation(str(exc)) from exc

        step_ms = self.memory_budget.estimate_step_duration_ms(
            active_parameters=self.active_parameters,
            step_tokens=total_step_tokens,
            resident_kv_tokens=resident_kv_tokens + prefill_chunk_tokens,
        )
        if step_ms > self.max_decode_step_ms:
            raise ResourceContractViolation(
                f"Estimated fused step duration {step_ms:.2f} ms exceeds TPOT ceiling "
                f"{self.max_decode_step_ms:.2f} ms (unchunked prefill starvation)."
            )
        return step_ms


@dataclass(frozen=True, slots=True)
class DeadlineContract:
    """Contract 2: Monotonic deadline propagation and pre-dispatch slack verification."""

    workflow_deadline_ns: int
    min_external_rpc_budget_ms: float

    def remaining_slack_ms(self, current_ns: int) -> float:
        """Compute remaining wall-clock budget D_rem = (tau_deadline - tau_now) in ms."""
        return (self.workflow_deadline_ns - current_ns) / 1_000_000.0

    def verify_dispatch_slack(self, current_ns: int, operation: str) -> float:
        """Abort before opening a network socket if remaining slack cannot cover the RPC."""
        slack_ms = self.remaining_slack_ms(current_ns)
        if slack_ms <= 0.0:
            raise DeadlineContractViolation(
                f"Deadline already expired before '{operation}' (slack={slack_ms:.2f} ms)."
            )
        if slack_ms < self.min_external_rpc_budget_ms:
            raise DeadlineContractViolation(
                f"Insufficient deadline slack ({slack_ms:.2f} ms < required "
                f"{self.min_external_rpc_budget_ms:.2f} ms) to safely dispatch '{operation}'."
            )
        return slack_ms


@dataclass(frozen=True, slots=True)
class StateContract:
    """Contract 3: Compare-and-swap version monotonicity and monotonic lease fencing."""

    expected_version: int
    fencing_token: int

    def verify_transition(
        self,
        *,
        head_version: int,
        head_fencing_token: int,
        proposed_version: int,
    ) -> None:
        """Enforce v_{t+1} == v_t + 1 and f_worker >= f_head."""
        if self.fencing_token < head_fencing_token:
            raise StateContractViolation(
                f"Stale worker fencing token {self.fencing_token} < head token {head_fencing_token}."
            )
        if head_version != self.expected_version:
            raise StateContractViolation(
                f"CAS version mismatch: expected {self.expected_version}, observed {head_version}."
            )
        if proposed_version != head_version + 1:
            raise StateContractViolation(
                f"Non-monotonic version transition: {head_version} -> {proposed_version}."
            )


@dataclass(frozen=True, slots=True)
class EffectContract:
    """Contract 4: Two-phase INTENT outbox and deterministic HMAC idempotency framing."""

    secret: bytes
    capability: ProviderCapability

    def mint_intent(
        self,
        ident: EffectIdentity,
        *,
        intent_committed_in_db: bool,
    ) -> str:
        """Verify that Phase 1 INTENT is durably committed and return deterministic key kappa."""
        if not intent_committed_in_db:
            raise EffectContractViolation(
                "External write forbidden: Phase 1 'INTENT' row must be committed before socket dispatch."
            )
        return idempotency_key(self.secret, ident)

    def classify_post_dispatch_timeout(self, effect_key: str) -> Literal["AMBIGUOUS_RECONCILE"]:
        """Classify a network/gateway timeout after INTENT commit as an Ambiguous state."""
        if self.capability == ProviderCapability.NON_IDEMPOTENT:
            raise EffectContractViolation(
                f"Ambiguous timeout on NON_IDEMPOTENT provider for key {effect_key}; "
                "automatic retry forbidden—escalate to Saga/human reconciliation."
            )
        return "AMBIGUOUS_RECONCILE"


@dataclass(frozen=True, slots=True)
class AuthorityContract:
    """Contract 5: Typed AgentEnvelope validation and server-side capability authorization."""

    directory: AgentDirectory

    def verify_tool_invocation(
        self,
        *,
        workflow_id: uuid.UUID,
        branch_id: str,
        step: int,
        version: int,
        authenticated_sender: str,
        claimed_sender: str,
        recipient: str,
        tool_name: str,
        arguments_json: str,
        issued_at: datetime,
    ) -> AgentEnvelope:
        """Construct a schema-locked AgentEnvelope and enforce server-side AgentDirectory grants."""
        try:
            envelope = AgentEnvelope(
                envelope_id=uuid.uuid4(),
                workflow_id=workflow_id,
                branch_id=branch_id,
                step=step,
                version=version,
                sender=claimed_sender,
                recipient=recipient,
                issued_at=issued_at,
                payload=ToolCallRequest(tool=tool_name, arguments_json=arguments_json),
            )
            authorize(
                envelope,
                directory=self.directory,
                authenticated_sender=authenticated_sender,
            )
            return envelope
        except Exception as exc:
            raise AuthorityContractViolation(str(exc)) from exc
