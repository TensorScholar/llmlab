"""The Four Production Budgets (Latency, Memory, Financial, Trust) and their coupling laws."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from llmlab.tracing.spans import StreamingLatencyBreakdown, TokenUsage, TraceGraph


class BudgetViolationError(RuntimeError):
    """Base exception raised when an execution step or trace breaches a production budget."""


class LatencyBudgetExceeded(BudgetViolationError):
    """Raised when TTFT, TPOT, P99 ITL, or end-to-end makespan exceeds its SLO ceiling."""


class MemoryBudgetExceeded(BudgetViolationError):
    """Raised when static weights, dynamic KV cache, and activations exceed HBM capacity."""


class FinancialBudgetExceeded(BudgetViolationError):
    """Raised when cumulative or expected task spend exceeds the USD ceiling C_max."""


class TrustBudgetExceeded(BudgetViolationError):
    """Raised when compounded trajectory error or unverified side-effect risk exceeds limits."""


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """SLO ceilings across prefill (TTFT), decode (TPOT / P99 ITL), and workflow makespan."""

    max_ttft_ms: float
    max_tpot_ms: float
    max_p99_itl_ms: float
    workflow_deadline_ms: float

    def __post_init__(self) -> None:
        if (
            self.max_ttft_ms <= 0.0
            or self.max_tpot_ms <= 0.0
            or self.max_p99_itl_ms <= 0.0
            or self.workflow_deadline_ms <= 0.0
        ):
            raise ValueError("All latency budget bounds must be strictly positive.")

    def check_stream(self, breakdown: StreamingLatencyBreakdown) -> None:
        """Verify that a streaming LLM invocation meets TTFT, mean TPOT, and P99 ITL bounds."""
        if breakdown.ttft_ms > self.max_ttft_ms:
            raise LatencyBudgetExceeded(
                f"TTFT {breakdown.ttft_ms:.2f} ms exceeded SLO {self.max_ttft_ms:.2f} ms."
            )
        if breakdown.tpot_ms > self.max_tpot_ms:
            raise LatencyBudgetExceeded(
                f"Mean TPOT {breakdown.tpot_ms:.2f} ms exceeded SLO {self.max_tpot_ms:.2f} ms."
            )
        if breakdown.p99_itl_ms > self.max_p99_itl_ms:
            raise LatencyBudgetExceeded(
                f"P99 ITL {breakdown.p99_itl_ms:.2f} ms (max gap {breakdown.max_itl_gap_ms:.2f} ms) "
                f"exceeded SLO {self.max_p99_itl_ms:.2f} ms."
            )

    def remaining_slack_ms(self, elapsed_ms: float) -> float:
        """Compute remaining workflow deadline slack D_rem = D_max - elapsed_ms."""
        return self.workflow_deadline_ms - elapsed_ms


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """HBM capacity decomposition across static weights, dynamic KV blocks, and activations."""

    hbm_capacity_bytes: int
    weight_bytes: int
    kv_bytes_per_token: int
    activation_bytes_per_token: int
    system_reserve_bytes: int
    peak_compute_tflops: float = 989.4
    peak_hbm_bandwidth_gbps: float = 3350.0

    def __post_init__(self) -> None:
        if self.hbm_capacity_bytes <= 0 or self.weight_bytes <= 0 or self.kv_bytes_per_token <= 0:
            raise ValueError("HBM capacity, weight bytes, and KV bytes/token must be positive.")
        if self.activation_bytes_per_token < 0 or self.system_reserve_bytes < 0:
            raise ValueError("Activation and system reserve bytes must be non-negative.")

    @property
    def ridge_point_flops_per_byte(self) -> float:
        """Hardware Roofline ridge point I* = Peak_TFLOP/s / Peak_HBM_GB/s."""
        return (self.peak_compute_tflops * 1e12) / (self.peak_hbm_bandwidth_gbps * 1e9)

    def available_kv_pool_bytes(self, step_tokens: int) -> int:
        """Compute HBM bytes available for the KV cache given active step token activations."""
        act_bytes = step_tokens * self.activation_bytes_per_token
        avail = (
            self.hbm_capacity_bytes
            - self.weight_bytes
            - self.system_reserve_bytes
            - act_bytes
        )
        return max(0, avail)

    def max_resident_kv_tokens(self, step_tokens: int) -> int:
        """Maximum concurrent context tokens that can reside in HBM without preemption."""
        return self.available_kv_pool_bytes(step_tokens) // self.kv_bytes_per_token

    def total_hbm_footprint_bytes(self, resident_kv_tokens: int, step_tokens: int) -> int:
        """Total HBM footprint M_total = M_weights + M_KV + M_act + M_sys."""
        return (
            self.weight_bytes
            + self.system_reserve_bytes
            + resident_kv_tokens * self.kv_bytes_per_token
            + step_tokens * self.activation_bytes_per_token
        )

    def verify_hbm_admission(self, resident_kv_tokens: int, step_tokens: int) -> None:
        """Enforce M_total <= M_HBM before scheduling an iteration."""
        footprint = self.total_hbm_footprint_bytes(resident_kv_tokens, step_tokens)
        if footprint > self.hbm_capacity_bytes:
            raise MemoryBudgetExceeded(
                f"HBM admission rejected: required {footprint / (1024**3):.2f} GiB exceeds "
                f"capacity {self.hbm_capacity_bytes / (1024**3):.2f} GiB."
            )

    def estimate_step_duration_ms(
        self,
        active_parameters: int,
        step_tokens: int,
        resident_kv_tokens: int,
        mfu: float = 0.50,
        mbu: float = 0.80,
    ) -> float:
        """Roofline-bounded step latency max(T_compute, T_memory) in milliseconds."""
        flops = 2.0 * active_parameters * step_tokens
        hbm_bytes = self.weight_bytes + resident_kv_tokens * self.kv_bytes_per_token
        t_compute_s = flops / (self.peak_compute_tflops * 1e12 * mfu)
        t_memory_s = hbm_bytes / (self.peak_hbm_bandwidth_gbps * 1e9 * mbu)
        return max(t_compute_s, t_memory_s) * 1000.0


@dataclass(frozen=True, slots=True)
class FinancialBudget:
    """Token economics and expected cost per successful task (USD/task) under retries."""

    price_uncached_per_1m: float
    price_cached_per_1m: float
    price_completion_per_1m: float
    max_cost_usd: float
    price_speculative_draft_per_1m: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.price_uncached_per_1m < 0.0
            or self.price_cached_per_1m < 0.0
            or self.price_completion_per_1m < 0.0
            or self.max_cost_usd <= 0.0
        ):
            raise ValueError("Token prices must be non-negative and max_cost_usd > 0.")

    def compute_token_cost_usd(self, usage: TokenUsage) -> float:
        """Exact USD cost for a TokenUsage record, including speculative draft overhead."""
        c_uncached = (usage.prompt_uncached / 1_000_000.0) * self.price_uncached_per_1m
        c_cached = (usage.prompt_cached / 1_000_000.0) * self.price_cached_per_1m
        c_comp = (usage.completion / 1_000_000.0) * self.price_completion_per_1m
        c_draft = (usage.speculative_drafted / 1_000_000.0) * self.price_speculative_draft_per_1m
        return c_uncached + c_cached + c_comp + c_draft

    @staticmethod
    def expected_uncheckpointed_cost_usd(
        step_cost_usd: float,
        steps: int,
        per_step_failure_prob: float,
    ) -> float:
        """Closed-form E[C_restart(T)] = (c / p) * ((1 - p)^(-T) - 1) from Chapter 10 Eq. 10.3."""
        if steps <= 0:
            return 0.0
        if per_step_failure_prob <= 0.0:
            return step_cost_usd * steps
        if per_step_failure_prob >= 1.0:
            raise ValueError("per_step_failure_prob must be in [0.0, 1.0).")
        p = per_step_failure_prob
        return (step_cost_usd / p) * (math.pow(1.0 - p, -steps) - 1.0)

    @staticmethod
    def expected_durable_cost_usd(
        step_cost_usd: float,
        steps: int,
        per_step_failure_prob: float,
    ) -> float:
        """Closed-form E[C_durable(T)] = T * c / (1 - p) from Chapter 10 Eq. 10.4."""
        if steps <= 0:
            return 0.0
        if not (0.0 <= per_step_failure_prob < 1.0):
            raise ValueError("per_step_failure_prob must be in [0.0, 1.0).")
        return (steps * step_cost_usd) / (1.0 - per_step_failure_prob)

    def check_spend(self, cost_usd: float) -> None:
        """Enforce cost_usd <= C_max."""
        if cost_usd > self.max_cost_usd:
            raise FinancialBudgetExceeded(
                f"Workflow spend ${cost_usd:.4f} exceeded financial ceiling C_max=${self.max_cost_usd:.4f}."
            )


@dataclass(frozen=True, slots=True)
class TrustBudget:
    """Bounds on compounded trajectory failure probability and unverified side-effect risk."""

    max_trajectory_failure_prob: float
    max_unverified_steps: int
    require_idempotent_external_writes: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.max_trajectory_failure_prob < 1.0):
            raise ValueError("max_trajectory_failure_prob must be in (0.0, 1.0).")
        if self.max_unverified_steps < 1:
            raise ValueError("max_unverified_steps must be >= 1.")

    @staticmethod
    def compounded_failure_probability(step_error_probs: Sequence[float]) -> float:
        """Compute P_fail(T) = 1 - prod_{t=1}^T (1 - epsilon_t) across unverified steps."""
        survival = 1.0
        for eps in step_error_probs:
            if not (0.0 <= eps <= 1.0):
                raise ValueError("Each step error probability must be in [0.0, 1.0].")
            survival *= 1.0 - eps
        return 1.0 - survival

    def verify_trajectory_trust(
        self,
        unverified_step_error_probs: Sequence[float],
        has_unreconciled_external_write: bool = False,
    ) -> None:
        """Enforce compounded hallucination/error ceiling and external effect safety."""
        if self.require_idempotent_external_writes and has_unreconciled_external_write:
            raise TrustBudgetExceeded(
                "External side-effecting tool invoked without two-phase INTENT/idempotency protection."
            )
        if len(unverified_step_error_probs) > self.max_unverified_steps:
            raise TrustBudgetExceeded(
                f"Unverified chain depth {len(unverified_step_error_probs)} exceeds "
                f"max_unverified_steps={self.max_unverified_steps}."
            )
        p_fail = self.compounded_failure_probability(unverified_step_error_probs)
        if p_fail > self.max_trajectory_failure_prob:
            raise TrustBudgetExceeded(
                f"Compounded trajectory failure probability {p_fail:.4f} exceeds "
                f"trust ceiling {self.max_trajectory_failure_prob:.4f}."
            )


@dataclass(frozen=True, slots=True)
class FourBudgetAssessment:
    """Unified evaluation summary across Latency, Memory, Financial, and Trust budgets."""

    critical_path_ms: float
    remaining_deadline_slack_ms: float
    peak_hbm_footprint_bytes: int
    total_cost_usd: float
    compounded_failure_prob: float


@dataclass(frozen=True, slots=True)
class FourBudgetLedger:
    """Coupled evaluator enforcing all four production budgets across a workflow trace."""

    latency: LatencyBudget
    memory: MemoryBudget
    financial: FinancialBudget
    trust: TrustBudget

    def evaluate_trace(
        self,
        trace: TraceGraph,
        *,
        resident_kv_tokens: int,
        step_tokens: int,
        unverified_step_error_probs: Sequence[float],
        has_unreconciled_external_write: bool = False,
    ) -> FourBudgetAssessment:
        """Audit a TraceGraph against all four coupled budgets, raising on any violation."""
        report = trace.analyze_critical_path()

        # 1. Memory budget check (checked first because HBM/prefill contention drives latency)
        self.memory.verify_hbm_admission(
            resident_kv_tokens=resident_kv_tokens,
            step_tokens=step_tokens,
        )

        # 2. Latency budget check across all LLM streaming spans and overall workflow makespan
        for span in trace.spans:
            if span.latency_breakdown is not None:
                self.latency.check_stream(span.latency_breakdown)

        slack_ms = self.latency.remaining_slack_ms(report.critical_path_ms)
        if slack_ms < 0.0:
            raise LatencyBudgetExceeded(
                f"Workflow critical path {report.critical_path_ms:.2f} ms exhausted "
                f"deadline {self.latency.workflow_deadline_ms:.2f} ms (slack={slack_ms:.2f} ms)."
            )

        # 3. Financial budget check
        cost_usd = self.financial.compute_token_cost_usd(report.aggregated_tokens)
        self.financial.check_spend(cost_usd)

        # 4. Trust & safety budget check
        self.trust.verify_trajectory_trust(
            unverified_step_error_probs=unverified_step_error_probs,
            has_unreconciled_external_write=has_unreconciled_external_write,
        )

        return FourBudgetAssessment(
            critical_path_ms=report.critical_path_ms,
            remaining_deadline_slack_ms=slack_ms,
            peak_hbm_footprint_bytes=self.memory.total_hbm_footprint_bytes(
                resident_kv_tokens, step_tokens
            ),
            total_cost_usd=cost_usd,
            compounded_failure_prob=self.trust.compounded_failure_probability(
                unverified_step_error_probs
            ),
        )
