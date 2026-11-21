import sqlite3

import pytest

from evalplat import store


def table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def test_connect_creates_schema(tmp_path):
    conn = store.connect(tmp_path / "sub" / "test.db")
    assert store.schema_version(conn) == store.SCHEMA_VERSION
    assert {"runs", "case_results", "checks", "events"} <= table_names(conn)


def test_reconnecting_does_not_rerun_migrations(tmp_path):
    path = tmp_path / "test.db"
    store.connect(path).close()
    conn = store.connect(path)  # would fail with "table already exists" if re-run
    assert store.schema_version(conn) == store.SCHEMA_VERSION


def test_only_new_migrations_are_applied():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    assert store.migrate(conn, ["CREATE TABLE a (x);"]) == 1
    assert store.migrate(conn, ["CREATE TABLE a (x);", "CREATE TABLE b (y);"]) == 2
    assert {"a", "b"} <= {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}


def test_failed_migration_leaves_no_partial_changes():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    broken = "CREATE TABLE a (x); CREATE TABLE a (x);"  # second statement fails
    with pytest.raises(sqlite3.OperationalError):
        store.migrate(conn, [broken])
    assert store.schema_version(conn) == 0
    assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0


def test_database_newer_than_code_is_refused():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer than this code"):
        store.migrate(conn)


def test_transaction_rolls_back_on_error():
    conn = store.connect(":memory:")
    with pytest.raises(ValueError):
        with store.transaction(conn):
            insert_run(conn)
            raise ValueError("boom")
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_deleting_a_run_deletes_its_cases_and_checks():
    conn = store.connect(":memory:")
    with store.transaction(conn):
        run_id = insert_run(conn)
        case_id = conn.execute(
            "INSERT INTO case_results (run_id, case_id, repeat_index, category, passed)"
            " VALUES (?, 'c1', 0, 'normal', 1)",
            (run_id,),
        ).lastrowid
        conn.execute(
            "INSERT INTO checks (case_result_id, position, kind, description, passed)"
            " VALUES (?, 0, 'field', 'd', 1)",
            (case_id,),
        )
    with store.transaction(conn):
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    assert conn.execute("SELECT count(*) FROM case_results").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM checks").fetchone()[0] == 0


def test_resolve_db_path_prefers_argument_then_env(monkeypatch):
    monkeypatch.setenv(store.DB_PATH_ENV, "from_env.db")
    assert store.resolve_db_path("arg.db").name == "arg.db"
    assert store.resolve_db_path().name == "from_env.db"
    monkeypatch.delenv(store.DB_PATH_ENV)
    assert store.resolve_db_path().name == store.DEFAULT_DB_PATH


def insert_run(conn):
    return conn.execute(
        "INSERT INTO runs (name, run_at, model, summary_json, ingested_at)"
        " VALUES ('r', '2026-01-01T00:00:00+00:00', 'm', '{}', 'now')"
    ).lastrowid
