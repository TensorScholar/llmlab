# Changelog

All notable changes to the `llmlab` reference repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-09

### Added
- Reviewer verification harness (`make verify-book`) chaining `mypy --strict` and `pytest`.
- Official reviewer quickstart guide (`docs/reviewer-quickstart.md`).
- Dual-mode architectural diagrams under `assets/figures/` (digital color and print-ready grayscale).
- Grace-period and row-level locking controls for external side-effect reconciliation.

## [0.1.1] - 2026-11-07

### Fixed
- **Valid UUIDv7 Literals**: Replaced invalid test UUID literals with valid RFC 9562 hexadecimal UUIDv7 literals (`0192f3a4-7c1b-7ae2-9d34-000000000001`), ensuring all PostgreSQL `uuid` casts and test fixtures execute without syntax or parsing errors.
- **Savepoint Handling in Transitions**: Hardened `commit_transition` to dynamically evaluate `conn.in_transaction()` before branching to `conn.begin_nested()` or `conn.begin()`, preventing savepoint errors and aborted transaction block failures.
- **Resumption Token & Monotonic Fencing**: Reconciled `RESUME_CAS_SQL` to draw from `nextval('fencing_token_seq')` by default while guarding with `c.fencing_token < next_fence.token` to prevent race conditions during workflow resumption.
- **Sweeper Race Guarding**: Added explicit `target.status = 'SUSPENDED' AND target.expires_at <= now()` re-validation inside `SWEEP_EXPIRED_SUSPENSIONS_SQL` to protect concurrent row claiming alongside `FOR UPDATE SKIP LOCKED`.

### Changed
- **ToolCallRequest Payload Interoperability**: Supported both UTF-8 validated `str` and `bytes` payloads in `ToolCallRequest.arguments_json`, ensuring full fidelity with Chapter 10 specification while preserving zero-copy binary compatibility.
- **Tarjan SCC Strongly Typed Overloads**: Added `@overload` signatures to `tarjan_scc`, allowing direct invocation with `GraphTopology` or `(vertices, adjacency)` mappings.
- **Envelope Authorization Bounds**: Reconciled `AgentEnvelope` bounds and `authorize()` keyword/positional calling conventions, supporting both `granted_capabilities` and `capabilities` attribute mapping on `AgentRegistration`.
- **Repository Attribution**: Rebranded repository metadata to publisher-neutral attribution for the upcoming book *Inside Production LLM Systems: Engineering Latency, Cost, Reliability, and Security for Production AI Infrastructure* by Muhammad Atashi.
- **Verification Status**: Added "Public, CI-Verified" status badges.

## [0.1.0] - 2026-11-02

### Added
- Initial release of the `llmlab` reference implementation for Chapter 10: Durable and Multi-Agent Execution.
- 100% local-first, zero-cloud runtime supporting Python 3.11+, PostgreSQL 16, SQLAlchemy 2.0 asyncio, asyncpg, and Pydantic v2.
- Static cyclic graph compilation and validation via Tarjan's SCC algorithm and condensation DAG sink reachability audits (`llmlab/durable/graph.py`).
- Optimistic Concurrency Control (OCC) state transition driver with row-locked Compare-and-Swap, monotonic fencing tokens, and commutative semilattice state rebasing (`llmlab/durable/cas.py`).
- Deterministic HMAC-SHA256 idempotency key generation and two-phase side-effect protocol across idempotent, status-queryable, and non-idempotent providers (`llmlab/durable/idempotency.py`).
- Zero-compute human-in-the-loop workflow suspension and single-use HMAC resumption tokens with lock-free expiry sweeping (`llmlab/durable/suspension.py`).
- Immutable, schema-locked inter-agent message envelopes with strict Pydantic v2 validation and server-side capability authorization (`llmlab/durable/envelope.py`).
- PostgreSQL physical persistence schema with append-only checkpoints, side-effect ledgers, and structural invariants (`sql/01_schema.sql`).
