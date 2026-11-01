"""Static graph topology representation, Tarjan SCC decomposition, and termination verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, AbstractSet, Sequence


class GraphCompilationError(Exception):
    """Base exception raised when a workflow graph fails static structural validation."""


class OrphanComponentError(GraphCompilationError):
    """Raised when one or more vertices or strongly connected components are unreachable from v_start."""


class NonTerminatingCycleError(GraphCompilationError):
    """Raised when a strongly connected component has zero outgoing edges and contains no terminal sink."""


@dataclass(frozen=True, slots=True)
class GraphTopology:
    """Static control-flow specification of a cyclic agent state machine."""

    vertices: frozenset[str]
    fixed_edges: Mapping[str, frozenset[str]]
    conditional_targets: Mapping[str, frozenset[str]]
    start_vertex: str
    terminal_vertices: frozenset[str] = field(default_factory=frozenset)

    def adjacency_map(self) -> dict[str, frozenset[str]]:
        """Return the combined outgoing edge set (E_fixed ∪ E_cond) for every vertex in V."""
        adj: dict[str, frozenset[str]] = {}
        for v in self.vertices:
            fixed = self.fixed_edges.get(v, frozenset())
            cond = self.conditional_targets.get(v, frozenset())
            adj[v] = fixed | cond
        return adj

    def digest(self) -> str:
        """Compute a canonical 64-hex-char SHA-256 digest of the static topology."""
        canonical = {
            "vertices": sorted(self.vertices),
            "fixed_edges": {k: sorted(v) for k, v in sorted(self.fixed_edges.items())},
            "conditional_targets": {
                k: sorted(v) for k, v in sorted(self.conditional_targets.items())
            },
            "start_vertex": self.start_vertex,
            "terminal_vertices": sorted(self.terminal_vertices),
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CompilationReport:
    """Output of static graph compilation and SCC condensation analysis."""

    graph_digest: str
    components: tuple[frozenset[str], ...]
    condensation_edges: Mapping[int, frozenset[int]]


def tarjan_scc(
    vertices: AbstractSet[str],
    adjacency: Mapping[str, AbstractSet[str]],
) -> list[frozenset[str]]:
    """Partition a directed graph into Strongly Connected Components in O(|V| + |E|) time."""
    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[frozenset[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index_counter
        indices[node] = index_counter
        lowlink[node] = index_counter
        index_counter += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(adjacency.get(node, frozenset())):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.add(w)
                if w == node:
                    break
            sccs.append(frozenset(component))

    for vertex in sorted(vertices):
        if vertex not in indices:
            strongconnect(vertex)

    return sccs


def validate_graph_termination(topology: GraphTopology) -> CompilationReport:
    """Verify structural invariants, entry reachability, and SCC sink validity."""
    if not topology.vertices:
        raise GraphCompilationError("Graph vertex set V cannot be empty.")
    if topology.start_vertex not in topology.vertices:
        raise GraphCompilationError(f"start_vertex '{topology.start_vertex}' is not in V.")
    if not topology.terminal_vertices:
        raise GraphCompilationError("terminal_vertices (V_end) must be non-empty.")
    if not topology.terminal_vertices.issubset(topology.vertices):
        unknown_terminals = topology.terminal_vertices - topology.vertices
        raise GraphCompilationError(f"Terminal vertices not in V: {sorted(unknown_terminals)}")
    if topology.start_vertex in topology.terminal_vertices:
        raise GraphCompilationError("start_vertex cannot be a terminal sink.")

    adj = topology.adjacency_map()

    for src, targets in adj.items():
        if src not in topology.vertices:
            raise GraphCompilationError(f"Edge source '{src}' is not declared in V.")
        unknown_targets = targets - topology.vertices
        if unknown_targets:
            raise GraphCompilationError(
                f"Vertex '{src}' routes to unknown target(s): {sorted(unknown_targets)}"
            )
        if src in topology.terminal_vertices and targets:
            raise GraphCompilationError(
                f"Terminal sink '{src}' must have OutDegree = 0, but has edges to {sorted(targets)}."
            )

    # 1. Entry reachability audit (BFS from v_start)
    visited: set[str] = set()
    queue: list[str] = [topology.start_vertex]
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        for nxt in adj.get(curr, frozenset()):
            if nxt not in visited:
                queue.append(nxt)

    unreachable = topology.vertices - visited
    if unreachable:
        raise OrphanComponentError(
            f"Vertices unreachable from start_vertex '{topology.start_vertex}': {sorted(unreachable)}"
        )

    # 2. Tarjan SCC decomposition and condensation DAG construction
    sccs: Sequence[frozenset[str]] = tarjan_scc(topology.vertices, adj)
    vertex_to_scc: dict[str, int] = {}
    for scc_idx, comp in enumerate(sccs):
        for v in comp:
            vertex_to_scc[v] = scc_idx

    condensation_edges: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
    for u, targets in adj.items():
        u_scc = vertex_to_scc[u]
        for v in targets:
            v_scc = vertex_to_scc[v]
            if u_scc != v_scc:
                condensation_edges[u_scc].add(v_scc)

    # 3. Sink validity audit: OutDegree(C) == 0 => C ∩ V_end ≠ ∅
    for scc_idx, comp in enumerate(sccs):
        if len(condensation_edges[scc_idx]) == 0:
            if comp.isdisjoint(topology.terminal_vertices):
                raise NonTerminatingCycleError(
                    f"Sink SCC {sorted(comp)} has no outgoing edges in the condensation graph "
                    "and contains no terminal vertex in V_end."
                )

    return CompilationReport(
        graph_digest=topology.digest(),
        components=tuple(sccs),
        condensation_edges={k: frozenset(v) for k, v in condensation_edges.items()},
    )