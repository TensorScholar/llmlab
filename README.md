# `llmlab` (`v0.1.0`) — Reference Implementation

[![CI](https://github.com/TensorScholar/llmlab/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorScholar/llmlab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official companion reference repository for **Inside Production LLM Systems: Engineering Latency, Cost, Reliability, and Security** by **Muhammad Atashi** (Manning Publications), featuring the complete runnable implementation for **Chapter 10: Durable and Multi-Agent Execution**.

---

## Architectural Overview

`llmlab` (`v0.1.0`) is engineered with a **100% local-first, zero-cloud-dependency** runtime (Python 3.11+, PostgreSQL 16, SQLAlchemy 2.0 `asyncio`, `asyncpg`, and Pydantic v2). Every SQL statement, state-machine invariant, and concurrency primitive in this repository maps 1-to-1 to the formal definitions and code listings in **Chapter 10: Durable and Multi-Agent Execution**:

* **`sql/01_schema.sql` (Section 10.3.2 — Physical Storage Schema & Structural Invariants):** Defines the append-only `checkpoints` lineage table, the `checkpoint_writes` delta and side-effect ledger, the monotonic `fencing_token_seq` sequence, partial unique indexes for idempotency keys, and `CHECK` constraints enforcing root/child lineage and suspension state consistency.
* **`llmlab/durable/graph.py` (Section 10.1.3 — Static Cyclic Graph Verification):** Implements `GraphTopology`, `tarjan_scc` for $O(|V| + |E|)$ Strongly Connected Components (SCC) decomposition, entry reachability auditing from $v_{\text{start}}$, and condensation DAG sink validation (`validate_graph_termination`) to reject deadlocked cycles (`NonTerminatingCycleError`) and unreachable subgraphs (`OrphanComponentError`) at compile time.
* **`llmlab/durable/cas.py` (Sections 10.3.3 & 10.3.4 — Optimistic Concurrency Control & Commutative Rebasing):** Implements atomic branch head advancement via row-locked Compare-and-Swap (`CAS_HEAD_SQL`), split-brain zombie rejection using monotonic fencing tokens (`LeaseLost`), and conflict inspection (`READ_HEAD_SQL`) that distinguishes non-commutative races (`ConcurrencyConflict`) from join-semilattice transitions eligible for automatic state rebasing (`RetryFromVersion`).
* **`llmlab/durable/idempotency.py` (Section 10.4 — Deterministic Idempotency & Two-Phase Side Effects):** Implements length-prefixed `HMAC-SHA256` idempotency key derivation (`idempotency_key`) and the 3-state physical (`APPLIED`, `INTENT`, `COMMITTED`) to 5-state logical (`Attempted`, `Dispatched`, `Confirmed`, `Ambiguous`, `Reconciled`) execution protocol (`execute_idempotent_effect`, `RECORD_INTENT_SQL`, `CONFIRM_EFFECT_SQL`) across `IDEMPOTENT_BY_KEY`, `STATUS_QUERYABLE`, and `NON_IDEMPOTENT` external providers (`UnknownSideEffectOutcome` / `UnknownOutcomeError`).
* **`llmlab/durable/suspension.py` (Section 10.5 — Zero-Compute Human-in-the-Loop Suspension):** Implements zero-worker-compute `SUSPENDED` checkpoint transitions (`SUSPEND_HEAD_SQL`), audience-, version-, and expiry-bound HMAC resumption tokens stored exclusively by SHA-256 digest (`issue_resumption_token`), constant-time single-use CAS redemption (`verify_and_resume_workflow`, `RESUME_CAS_SQL`), and lock-free batch expiration reclamation using `FOR UPDATE SKIP LOCKED` (`sweep_expired_suspensions`, `SWEEP_EXPIRED_SUSPENSIONS_SQL`).
* **`llmlab/durable/envelope.py` (Section 10.6.2 — Typed Inter-Agent Envelopes & Capability Authorization):** Implements frozen, size-bounded (`MAX_ARGUMENT_BYTES = 8192`) Pydantic v2 message envelopes (`AgentEnvelope`, `ToolCallRequest`) with `ConfigDict(extra="forbid", frozen=True)` and server-side topology edge and tool capability enforcement via `AgentDirectory` and `authorize`.

---

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── sql/
│   └── 01_schema.sql
├── llmlab/
│   ├── __init__.py
│   ├── py.typed
│   └── durable/
│       ├── __init__.py
│       ├── cas.py
│       ├── envelope.py
│       ├── graph.py
│       ├── idempotency.py
│       └── suspension.py
└── tests/
    ├── conftest.py
    ├── test_cas_concurrency.py
    ├── test_graph_termination.py
    ├── test_idempotency_crash.py
    └── test_suspension_tokens.py
```

---

## Quickstart & Verification Commands

Reproduce the complete static type check and PostgreSQL 16 concurrency/crash-recovery test suite locally in 4 steps:

```bash
# 1. Start the local PostgreSQL 16 container and initialize sql/01_schema.sql
docker compose up -d

# 2. Install llmlab in editable mode with development dependencies
pip install -e ".[dev]"

# 3. Run strict static type analysis across package and test suite
mypy llmlab tests

# 4. Execute the unit and PostgreSQL concurrency/crash-recovery integration tests
pytest -v
```
