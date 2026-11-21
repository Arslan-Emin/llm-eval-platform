"""SQLite storage: one file holds eval runs, case results, checks and trace events.

Schema changes are numbered migrations. The database remembers how many it has
applied in SQLite's built-in `PRAGMA user_version`, so `connect()` upgrades an old
file in place and never re-runs a migration. Each migration runs in a single
transaction together with the version bump: it is applied completely or not at all.

Connections use autocommit mode (`isolation_level=None`); writes are grouped
explicitly with `transaction()`, which starts with `BEGIN IMMEDIATE` so a writer
takes the write lock up front instead of failing halfway when another process
(the ingest server, the CLI) writes at the same time.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DB_PATH_ENV = "EVALPLAT_DB"
DEFAULT_DB_PATH = "evalplat.db"

# Append only: never edit a migration that may already have been applied somewhere.
MIGRATIONS: list[str] = [
    # 1: eval runs, case results, checks, trace events
    """
    CREATE TABLE runs (
        id            INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,     -- results file stem, e.g. 20261121-152221_deepseek-flash
        run_at        TEXT NOT NULL,     -- UTC ISO timestamp written by the eval runner
        provider      TEXT,
        model         TEXT NOT NULL,
        thinking      TEXT,
        eval_clock    TEXT,
        cases_file    TEXT,
        repeats       INTEGER,
        trace_source  TEXT,              -- events.source of this run's traces, NULL if none
        wall_time_s   REAL,
        summary_json  TEXT NOT NULL,     -- the summary exactly as the runner reported it
        source_path   TEXT,
        ingested_at   TEXT NOT NULL,
        UNIQUE (run_at, model)
    );

    CREATE TABLE case_results (
        id                   INTEGER PRIMARY KEY,
        run_id               INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
        case_id              TEXT NOT NULL,
        repeat_index         INTEGER NOT NULL,  -- 0 .. repeats-1, in file order
        category             TEXT NOT NULL,
        conversation_id      TEXT,              -- joins with events.conversation_id
        passed               INTEGER NOT NULL,
        error                TEXT,
        expected_submit      INTEGER,
        submitted            INTEGER,
        user_turns           INTEGER,
        llm_calls            INTEGER,
        input_tokens         INTEGER,
        cached_input_tokens  INTEGER,
        output_tokens        INTEGER,
        latency_ms           REAL,
        cost_usd             REAL,
        wall_time_s          REAL,
        final_values_json    TEXT,
        transcript_json      TEXT,
        UNIQUE (run_id, case_id, repeat_index)
    );
    CREATE INDEX case_results_conversation ON case_results (conversation_id);

    CREATE TABLE checks (
        id              INTEGER PRIMARY KEY,
        case_result_id  INTEGER NOT NULL REFERENCES case_results (id) ON DELETE CASCADE,
        position        INTEGER NOT NULL,
        kind            TEXT NOT NULL,     -- field | submit | reply | rag | tool
        description     TEXT NOT NULL,
        passed          INTEGER NOT NULL,
        turn            INTEGER,           -- NULL = checked at the end of the conversation
        UNIQUE (case_result_id, position)
    );

    -- One row per trace event. Fields used in queries get their own column; the
    -- full event is kept in data_json so nothing is lost when the format grows.
    CREATE TABLE events (
        event_id             TEXT PRIMARY KEY,
        source               TEXT NOT NULL,  -- trace file name, or 'live' for HTTP ingestion
        type                 TEXT NOT NULL,  -- llm_call | tool_call | agent_turn
        timestamp            TEXT NOT NULL,
        conversation_id      TEXT,
        turn_index           INTEGER,
        name                 TEXT,           -- tool name (tool_call)
        model                TEXT,
        ok                   INTEGER,
        input_tokens         INTEGER,
        cached_input_tokens  INTEGER,
        output_tokens        INTEGER,
        cost_usd             REAL,
        latency_ms           REAL,
        duration_ms          REAL,
        data_json            TEXT NOT NULL
    );
    CREATE INDEX events_conversation ON events (conversation_id, timestamp);
    CREATE INDEX events_source_type ON events (source, type);
    """,
    # 2: LLM judge verdicts, one per (rubric, agent reply)
    """
    CREATE TABLE judgments (
        id                   INTEGER PRIMARY KEY,
        case_result_id       INTEGER REFERENCES case_results (id) ON DELETE CASCADE,
        conversation_id      TEXT NOT NULL,
        turn_index           INTEGER NOT NULL,
        rubric               TEXT NOT NULL,
        rubric_version       TEXT NOT NULL,
        -- Hash of the exact prompt: the same question is never paid for twice, and
        -- editing a rubric's text automatically makes old verdicts stale.
        prompt_hash          TEXT NOT NULL,
        judge_model          TEXT NOT NULL,
        verdict              TEXT NOT NULL CHECK (verdict IN ('pass', 'fail')),
        reason               TEXT,
        input_tokens         INTEGER,
        cached_input_tokens  INTEGER,
        output_tokens        INTEGER,
        cost_usd             REAL,
        latency_ms           REAL,
        -- 1 = verdict copied from an identical question judged before (no API call,
        -- no cost). Repeats of a case often produce the very same reply.
        cached               INTEGER NOT NULL DEFAULT 0,
        created_at           TEXT NOT NULL,
        UNIQUE (case_result_id, turn_index, rubric, judge_model, prompt_hash)
    );
    CREATE INDEX judgments_prompt ON judgments (prompt_hash, judge_model);
    CREATE INDEX judgments_conversation ON judgments (conversation_id, turn_index);
    CREATE INDEX judgments_case ON judgments (case_result_id);
    """,
    # 3: human labels for calibrating the judge
    """
    -- The replies picked for labeling, in the order they are shown.
    CREATE TABLE label_queue (
        prompt_hash      TEXT PRIMARY KEY,  -- the judge question being labeled
        rubric           TEXT NOT NULL,
        conversation_id  TEXT NOT NULL,
        turn_index       INTEGER NOT NULL,
        position         INTEGER NOT NULL,
        added_at         TEXT NOT NULL
    );

    -- A human's answer to the same question the judge got. Keyed by prompt hash, so
    -- it is compared with the judge verdict for exactly that prompt.
    CREATE TABLE labels (
        id               INTEGER PRIMARY KEY,
        prompt_hash      TEXT NOT NULL,
        rubric           TEXT NOT NULL,
        conversation_id  TEXT NOT NULL,
        turn_index       INTEGER NOT NULL,
        label            TEXT NOT NULL CHECK (label IN ('pass', 'fail', 'not_applicable')),
        note             TEXT,
        labeler          TEXT NOT NULL,
        created_at       TEXT NOT NULL,
        UNIQUE (prompt_hash, labeler)
    );
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


def resolve_db_path(path: str | Path | None = None) -> Path:
    """Explicit path, else the EVALPLAT_DB environment variable, else ./evalplat.db."""
    return Path(path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the database and bring its schema up to date."""
    target = ":memory:" if str(path) == ":memory:" else resolve_db_path(path)
    if target != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the dashboard read while the ingest server writes.
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection, migrations: list[str] | None = None) -> int:
    """Apply the migrations this database has not seen yet; return the new version."""
    migrations = MIGRATIONS if migrations is None else migrations
    current = schema_version(conn)
    if current > len(migrations):
        raise RuntimeError(
            f"Database schema version {current} is newer than this code "
            f"({len(migrations)}); upgrade evalplat."
        )
    for version, script in enumerate(migrations[current:], start=current + 1):
        try:
            # executescript runs several statements; BEGIN/COMMIT make them atomic,
            # and user_version is part of the same transaction.
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;"
            )
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return schema_version(conn)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Group writes: commit if the block succeeds, roll everything back if it raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
