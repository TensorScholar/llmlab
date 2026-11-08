# Reviewer Quickstart & Reproducibility Manifest

Welcome, O'Reilly Editorial & Technical Reviewers. This suite allows you to independently verify the core architectural claims of Chapter 10 without requiring a GPU or paid model APIs.

## Prerequisites
- Python 3.11+
- A running PostgreSQL 16 instance (e.g., via Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16-alpine)

## Execution
Run the following command at the repository root:
make verify-book

## What This Verifies (10-Minute Target)
1. **CAS Race Condition:** Two workers racing on expected_version resolve to exactly 1 winner.
2. **Intent/Commit Crash Recovery:** Simulated SIGKILL between outbox INTENT and COMMITTED states safely resolves via provider-capability protocols.
3. **Graph Topology:** SCC condensation properly rejects deadlocked cyclic graphs.
