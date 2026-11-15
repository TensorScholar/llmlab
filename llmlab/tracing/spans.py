"""Distributed trace spans, streaming latency decomposition, and DAG critical-path analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

SpanStatus = Literal["OK", "ERROR", "TIMEOUT", "CANCELLED"]


class SpanKind(str, Enum):
    """OpenInference / OpenTelemetry GenAI semantic span discriminator."""

    AGENT = "AGENT"
    CHAIN = "CHAIN"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    TOOL = "TOOL"
    GUARDRAIL = "GUARDRAIL"
    CHECKPOINT = "CHECKPOINT"


class TraceTopologyError(ValueError):
    """Base error for invalid distributed trace graph topologies."""


class TemporalContainmentError(TraceTopologyError):
    """Raised when a child span violates parent start/end temporal containment."""

    def __init__(self, parent_id: str, child_id: str, reason: str) -> None:
        super().__init__(
            f"Temporal containment violated between parent '{parent_id}' and child '{child_id}': {reason}"
        )
        self.parent_id = parent_id
        self.child_id = child_id
        self.reason = reason


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Exact token accounting for a single LLM invocation span."""

    prompt_uncached: int
    prompt_cached: int
    completion: int
    speculative_drafted: int = 0
    speculative_accepted: int = 0

    def __post_init__(self) -> None:
        if self.prompt_uncached < 0 or self.prompt_cached < 0 or self.completion < 0:
            raise ValueError("Token counts must be non-negative integers.")
        if self.speculative_drafted < 0 or self.speculative_accepted < 0:
            raise ValueError("Speculative token counts must be non-negative.")
        if self.speculative_accepted > self.speculative_drafted:
            raise ValueError("speculative_accepted cannot exceed speculative_drafted.")

    @property
    def prompt_total(self) -> int:
        """Total input prompt length S_in = S_in^uncached + S_in^cached."""
        return self.prompt_uncached + self.prompt_cached

    @property
    def total_tokens(self) -> int:
        """Total logical tokens processed (S_in + N_out)."""
        return self.prompt_total + self.completion

    @property
    def cache_hit_ratio(self) -> float:
        """Prefix cache hit rate in [0.0, 1.0]."""
        total = self.prompt_total
        return self.prompt_cached / total if total > 0 else 0.0

    @property
    def speculative_Rejection_tokens(self) -> int:
        """Drafted tokens discarded during target verification."""
        return self.speculative_drafted - self.speculative_accepted


@dataclass(frozen=True, slots=True)
class StreamingLatencyBreakdown:
    """Token-level latency decomposition into TTFT, ITL sequence, and TPOT."""

    ttft_ms: float
    itl_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.ttft_ms < 0.0:
            raise ValueError("ttft_ms must be non-negative.")
        if any(gap < 0.0 for gap in self.itl_ms):
            raise ValueError("All inter-token latency (ITL) values must be non-negative.")

    @property
    def completion_tokens(self) -> int:
        """Total output tokens N_out = 1 (first token) + len(itl_ms)."""
        return 1 + len(self.itl_ms)

    @property
    def decode_total_ms(self) -> float:
        """Cumulative autoregressive decode time across tokens 2..N_out."""
        return math.fsum(self.itl_ms)

    @property
    def total_stream_ms(self) -> float:
        """End-to-end LLM generation duration: TTFT + sum(ITL)."""
        return self.ttft_ms + self.decode_total_ms

    @property
    def tpot_ms(self) -> float:
        """Mean Time Per Output Token (TPOT) across decode steps."""
        if not self.itl_ms:
            return 0.0
        return self.decode_total_ms / len(self.itl_ms)

    @property
    def max_itl_gap_ms(self) -> float:
        """Worst-case mid-stream pause (max_j ITL_j)."""
        if not self.itl_ms:
            return 0.0
        return max(self.itl_ms)

    def quantile_itl_ms(self, q: float) -> float:
        """Compute empirical quantile q in [0, 1] over the ITL sequence."""
        if not (0.0 <= q <= 1.0):
            raise ValueError("Quantile q must be in [0.0, 1.0].")
        if not self.itl_ms:
            return 0.0
        ordered = sorted(self.itl_ms)
        idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[idx]

    @property
    def p99_itl_ms(self) -> float:
        """99th-percentile Inter-Token Latency."""
        return self.quantile_itl_ms(0.99)


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """Immutable semantic execution span within a distributed workflow trace."""

    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start_ns: int
    end_ns: int
    status: SpanStatus = "OK"
    token_usage: TokenUsage | None = None
    latency_breakdown: StreamingLatencyBreakdown | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.span_id or not self.trace_id:
            raise ValueError("span_id and trace_id must be non-empty.")
        if self.end_ns < self.start_ns:
            raise ValueError(
                f"Span '{self.span_id}' has end_ns ({self.end_ns}) < start_ns ({self.start_ns})."
            )

    @property
    def duration_ms(self) -> float:
        """Physical wall-clock span duration in milliseconds."""
        return (self.end_ns - self.start_ns) / 1_000_000.0


@dataclass(frozen=True, slots=True)
class CriticalPathReport:
    """Result of topological critical-path and concurrency analysis over a TraceGraph."""

    root_span_id: str
    critical_path_span_ids: tuple[str, ...]
    critical_path_ms: float
    total_leaf_work_ms: float
    concurrency_efficiency: float
    aggregated_tokens: TokenUsage


class TraceGraph:
    """Directed rooted tree of TraceSpans enforcing causal and temporal invariants."""

    def __init__(self, spans: Sequence[TraceSpan]) -> None:
        if not spans:
            raise TraceTopologyError("TraceGraph requires at least one span.")
        self._spans_by_id: dict[str, TraceSpan] = {}
        for s in spans:
            if s.span_id in self._spans_by_id:
                raise TraceTopologyError(f"Duplicate span_id '{s.span_id}' in trace.")
            self._spans_by_id[s.span_id] = s

        trace_ids = {s.trace_id for s in spans}
        if len(trace_ids) != 1:
            raise TraceTopologyError(f"All spans must share one trace_id; found {sorted(trace_ids)}.")

        roots = [s for s in spans if s.parent_id is None]
        if len(roots) != 1:
            raise TraceTopologyError(f"TraceGraph must have exactly 1 root span; found {len(roots)}.")
        self._root: TraceSpan = roots[0]

        self._children: dict[str, list[TraceSpan]] = {s.span_id: [] for s in spans}
        for s in spans:
            if s.parent_id is not None:
                if s.parent_id not in self._spans_by_id:
                    raise TraceTopologyError(
                        f"Span '{s.span_id}' references unknown parent_id '{s.parent_id}'."
                    )
                self._children[s.parent_id].append(s)

        for child_list in self._children.values():
            child_list.sort(key=lambda sp: (sp.start_ns, sp.end_ns, sp.span_id))

        self._validate_acyclic_and_connected()

    @property
    def root(self) -> TraceSpan:
        return self._root

    @property
    def spans(self) -> tuple[TraceSpan, ...]:
        return tuple(self._spans_by_id.values())

    def children_of(self, span_id: str) -> tuple[TraceSpan, ...]:
        return tuple(self._children.get(span_id, ()))

    def _validate_acyclic_and_connected(self) -> None:
        visited: set[str] = set()
        stack: list[str] = [self._root.span_id]
        while stack:
            curr = stack.pop()
            if curr in visited:
                raise TraceTopologyError(f"Cycle detected involving span '{curr}'.")
            visited.add(curr)
            for child in self._children[curr]:
                stack.append(child.span_id)
        if len(visited) != len(self._spans_by_id):
            orphans = sorted(set(self._spans_by_id) - visited)
            raise TraceTopologyError(f"Disconnected spans in trace: {orphans}")

    def validate_temporal_containment(self) -> None:
        """Enforce tau_start,p <= tau_start,c <= tau_end,c <= tau_end,p across all edges."""
        for parent_id, children in self._children.items():
            parent = self._spans_by_id[parent_id]
            for child in children:
                if child.start_ns < parent.start_ns:
                    raise TemporalContainmentError(
                        parent_id=parent.span_id,
                        child_id=child.span_id,
                        reason=f"child start_ns ({child.start_ns}) < parent start_ns ({parent.start_ns})",
                    )
                if child.end_ns > parent.end_ns:
                    raise TemporalContainmentError(
                        parent_id=parent.span_id,
                        child_id=child.span_id,
                        reason=(
                            f"child end_ns ({child.end_ns}) > parent end_ns ({parent.end_ns}) "
                            "(orphaned child continued executing after parent closed/timed out)"
                        ),
                    )

    def aggregate_tokens(self) -> TokenUsage:
        """Aggregate token usage across all LLM spans in the trace."""
        uncached = 0
        cached = 0
        completion = 0
        drafted = 0
        accepted = 0
        for s in self._spans_by_id.values():
            if s.token_usage is not None:
                uncached += s.token_usage.prompt_uncached
                cached += s.token_usage.prompt_cached
                completion += s.token_usage.completion
                drafted += s.token_usage.speculative_drafted
                accepted += s.token_usage.speculative_accepted
        return TokenUsage(
            prompt_uncached=uncached,
            prompt_cached=cached,
            completion=completion,
            speculative_drafted=drafted,
            speculative_accepted=accepted,
        )

    def analyze_critical_path(self) -> CriticalPathReport:
        """Compute critical path duration, leaf work sum, and concurrency speedup ratio."""
        self.validate_temporal_containment()

        # Leaf spans are spans with no children; they represent actual execution stages.
        leaf_spans = [s for s in self._spans_by_id.values() if not self._children[s.span_id]]
        total_leaf_work_ms = math.fsum(s.duration_ms for s in leaf_spans)

        # Compute union of active leaf intervals to measure true non-overlapping critical work,
        # plus identify the root-to-leaf path with maximum cumulative child duration.
        def longest_chain(span_id: str) -> tuple[float, list[str]]:
            span = self._spans_by_id[span_id]
            children = self._children[span_id]
            if not children:
                return span.duration_ms, [span_id]

            # Group concurrent children into overlapping clusters; within a cluster, the slowest
            # branch dominates the critical path.
            clusters: list[list[TraceSpan]] = []
            for ch in children:
                if not clusters or ch.start_ns >= max(c.end_ns for c in clusters[-1]):
                    clusters.append([ch])
                else:
                    clusters[-1].append(ch)

            chosen_ids: list[str] = [span_id]
            for cluster in clusters:
                best_dur = -1.0
                best_chain: list[str] = []
                for ch in cluster:
                    dur, chain = longest_chain(ch.span_id)
                    if dur > best_dur:
                        best_dur = dur
                        best_chain = chain
                chosen_ids.extend(best_chain)

            return span.duration_ms, chosen_ids

        _, crit_ids = longest_chain(self._root.span_id)
        critical_path_ms = self._root.duration_ms
        concurrency_efficiency = (
            total_leaf_work_ms / critical_path_ms if critical_path_ms > 0.0 else 1.0
        )

        return CriticalPathReport(
            root_span_id=self._root.span_id,
            critical_path_span_ids=tuple(crit_ids),
            critical_path_ms=critical_path_ms,
            total_leaf_work_ms=total_leaf_work_ms,
            concurrency_efficiency=concurrency_efficiency,
            aggregated_tokens=self.aggregate_tokens(),
        )
