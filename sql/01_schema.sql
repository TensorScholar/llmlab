-- sql/01_schema.sql
-- Core schema for durable state transitions, two-phase side-effect records,
-- monotonic fencing tokens, and human-in-the-loop suspensions.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SEQUENCE IF NOT EXISTS fencing_token_seq
    AS bigint
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    NO CYCLE;

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id        uuid        PRIMARY KEY,
    workflow_id          uuid        NOT NULL,
    branch_id            text        NOT NULL DEFAULT 'main',
    parent_checkpoint_id uuid        NULL,
    step                 bigint      NOT NULL CHECK (step >= 0),
    version              bigint      NOT NULL CHECK (version >= 1),
    node_id              text        NOT NULL,
    graph_digest         text        NOT NULL CHECK (graph_digest ~ '^[0-9a-f]{64}$'),
    status               varchar(16) NOT NULL DEFAULT 'RUNNING'
                                     CHECK (status IN ('RUNNING', 'SUSPENDED', 'COMPLETED', 'FAILED')),
    fencing_token        bigint      NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    state_payload        jsonb       NOT NULL,
    state_hash           bytea       NOT NULL CHECK (octet_length(state_hash) = 32),
    committed_at         timestamptz NOT NULL DEFAULT now(),
    suspended_at         timestamptz NULL,
    resume_token_digest  bytea       NULL,
    expires_at           timestamptz NULL,

    CONSTRAINT checkpoints_branch_version_uniq
        UNIQUE (workflow_id, branch_id, version),

    CONSTRAINT checkpoints_workflow_checkpoint_uniq
        UNIQUE (workflow_id, checkpoint_id),

    CONSTRAINT checkpoints_no_self_parent
        CHECK (parent_checkpoint_id IS DISTINCT FROM checkpoint_id),

    CONSTRAINT checkpoints_root_or_child
        CHECK (parent_checkpoint_id IS NOT NULL OR (step = 0 AND version = 1)),

    CONSTRAINT checkpoints_suspension_consistent CHECK (
        (status = 'SUSPENDED'
            AND suspended_at IS NOT NULL
            AND resume_token_digest IS NOT NULL
            AND octet_length(resume_token_digest) = 32
            AND expires_at IS NOT NULL)
        OR
        (status <> 'SUSPENDED'
            AND suspended_at IS NULL
            AND resume_token_digest IS NULL
            AND expires_at IS NULL)
    ),

    CONSTRAINT checkpoints_parent_fk
        FOREIGN KEY (workflow_id, parent_checkpoint_id)
        REFERENCES checkpoints (workflow_id, checkpoint_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS checkpoints_head_lookup_idx
    ON checkpoints (workflow_id, branch_id, version DESC);

CREATE INDEX IF NOT EXISTS checkpoints_suspended_expiry_idx
    ON checkpoints (expires_at)
    WHERE status = 'SUSPENDED';

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    checkpoint_id     uuid        NOT NULL,
    write_index       integer     NOT NULL CHECK (write_index >= 0),
    workflow_id       uuid        NOT NULL,
    branch_id         text        NOT NULL,
    node_id           text        NOT NULL,
    channel_name      varchar(64) NOT NULL CHECK (channel_name ~ '^[a-z][a-z0-9_]{0,63}$'),
    write_state       varchar(16) NOT NULL DEFAULT 'APPLIED'
                                  CHECK (write_state IN ('APPLIED', 'INTENT', 'COMMITTED')),
    write_payload     jsonb       NOT NULL,
    side_effect_key   char(64)    NULL,
    recorded_response jsonb       NULL,
    recorded_at       timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT checkpoint_writes_pkey
        PRIMARY KEY (checkpoint_id, write_index),

    CONSTRAINT checkpoint_writes_state_pairing CHECK (
        (write_state = 'APPLIED'   AND side_effect_key IS NULL     AND recorded_response IS NULL)
     OR (write_state = 'INTENT'    AND side_effect_key IS NOT NULL AND recorded_response IS NULL)
     OR (write_state = 'COMMITTED' AND side_effect_key IS NOT NULL AND recorded_response IS NOT NULL)
    ),

    CONSTRAINT checkpoint_writes_key_shape
        CHECK (side_effect_key IS NULL OR side_effect_key ~ '^[0-9a-f]{64}$'),

    CONSTRAINT checkpoint_writes_checkpoint_fk
        FOREIGN KEY (workflow_id, checkpoint_id)
        REFERENCES checkpoints (workflow_id, checkpoint_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS checkpoint_writes_effect_key_uniq
    ON checkpoint_writes (workflow_id, side_effect_key)
    WHERE side_effect_key IS NOT NULL;