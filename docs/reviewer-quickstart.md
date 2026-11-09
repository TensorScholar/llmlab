# Reviewer Quickstart & Reproducibility Manifest

Welcome, O'Reilly Editorial & Technical Reviewers. This suite allows you to independently verify the core architectural claims of Chapter 10 in under 10 minutes without requiring a GPU or paid model APIs.

## Prerequisites
- Python 3.11+
- Docker Engine / Docker Desktop

## Quickstart (Execution)
Run these commands from the repository root:

```bash
# 1. Start isolated PostgreSQL 16 service with exact schema & credentials
docker compose up -d

# 2. Install package in editable mode with development test dependencies
pip install -e ".[dev]"

# 3. Execute the complete verification harness
make verify-book
```

## What This Verifies
1. **CAS Race Condition:** Two concurrent workers racing on `expected_version` resolve to exactly 1 winner.
2. **Intent/Commit Crash Recovery:** Simulated SIGKILL between outbox `INTENT` and `COMMITTED` states safely resolves via provider-capability protocols.
3. **Graph Topology:** Tarjan SCC condensation properly rejects deadlocked cyclic graphs.
4. **Type Safety:** 100% clean static analysis under `mypy --strict`.

## Teardown
When finished, stop the isolated container:

```bash
docker compose down -v
```
