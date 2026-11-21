import json

import pytest

from evalplat import store
from evalplat.cli import main
from evalplat.ingest import FileReport, ingest_events, ingest_paths, ingest_results
from tests.factories import SAMPLE_DIR, make_case, make_event, make_results


@pytest.fixture
def conn():
    return store.connect(":memory:")


def count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --------------------------------------------------------------------------- #
# Eval results
# --------------------------------------------------------------------------- #


def test_results_are_stored_with_cases_and_checks(conn):
    payload = make_results(
        [make_case("c1"), make_case("c2", passed=False)],
        trace_file="logs/traces/eval_20261121-152221.jsonl",
    )
    report = ingest_results(conn, payload, name="run-a")

    assert report.added == 2
    run = conn.execute("SELECT * FROM runs").fetchone()
    assert run["name"] == "run-a"
    assert run["trace_source"] == "eval_20261121-152221.jsonl"
    assert json.loads(run["summary_json"])["cases"]["total"] == 1
    rows = conn.execute("SELECT case_id, passed, llm_calls FROM case_results ORDER BY case_id")
    assert [tuple(r) for r in rows] == [("c1", 1, 4), ("c2", 0, 4)]
    assert count(conn, "checks") == 2


def test_repeated_cases_get_a_repeat_index(conn):
    ingest_results(
        conn, make_results([make_case("c1"), make_case("c1"), make_case("c2")]), name="r"
    )
    rows = conn.execute("SELECT case_id, repeat_index FROM case_results ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("c1", 0), ("c1", 1), ("c2", 0)]


def test_ingesting_the_same_run_twice_adds_nothing(conn):
    payload = make_results([make_case()])
    ingest_results(conn, payload, name="r")
    report = ingest_results(conn, payload, name="r")
    assert (report.added, report.duplicates) == (0, 1)
    assert count(conn, "runs") == 1
    assert count(conn, "case_results") == 1


def test_broken_case_rolls_back_the_whole_run(conn, tmp_path):
    broken = make_case("c2")
    del broken["category"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(make_results([make_case("c1"), broken])), encoding="utf-8")

    [report] = ingest_paths(conn, [path])

    assert report.kind == "skipped" and report.invalid == 1
    assert "category" in report.notes[0]
    assert count(conn, "runs") == 0
    assert count(conn, "case_results") == 0


def test_json_files_that_are_not_eval_runs_are_skipped(conn, tmp_path):
    (tmp_path / "trace_summary.json").write_text('{"summary": {}}', encoding="utf-8")
    [report] = ingest_paths(conn, [tmp_path])
    assert report.kind == "skipped"
    assert report.invalid == 0


# --------------------------------------------------------------------------- #
# Trace events
# --------------------------------------------------------------------------- #


def test_events_are_stored_with_query_columns(conn):
    events = [
        make_event("e1", model="deepseek-flash", ok=True, cost_usd=0.0002, latency_ms=1074.1),
        make_event("e2", "tool_call", name="update_fields", ok=False, duration_ms=7.6),
    ]
    report = ingest_events(conn, events, source="t.jsonl", report=FileReport("t", "traces"))

    assert report.added == 2
    llm = conn.execute("SELECT * FROM events WHERE event_id = 'e1'").fetchone()
    assert (llm["model"], llm["ok"], llm["latency_ms"], llm["name"]) == (
        "deepseek-flash",
        1,
        1074.1,
        None,
    )
    tool = conn.execute("SELECT * FROM events WHERE event_id = 'e2'").fetchone()
    assert (tool["name"], tool["ok"], tool["duration_ms"]) == ("update_fields", 0, 7.6)
    assert json.loads(tool["data_json"])["name"] == "update_fields"


def test_duplicate_and_invalid_events_are_counted(conn):
    events = [
        make_event("e1"),
        make_event("e1"),
        make_event("e2", schema_version=2),
        make_event("e3", type_="unknown"),
        {"event_id": "e4"},
        "not an object",
    ]
    report = ingest_events(conn, events, source="t.jsonl", report=FileReport("t", "traces"))
    assert (report.added, report.duplicates, report.invalid) == (1, 1, 4)
    assert "unsupported schema_version 2" in report.notes


def test_bad_lines_in_a_trace_file_do_not_stop_the_rest(conn, tmp_path):
    path = tmp_path / "eval_1.jsonl"
    path.write_text(
        json.dumps(make_event("e1")) + "\n{broken\n\n" + json.dumps(make_event("e2")) + "\n",
        encoding="utf-8",
    )
    [report] = ingest_paths(conn, [path])
    assert (report.added, report.invalid) == (2, 1)
    assert report.notes == ["line 2: invalid JSON"]
    sources = {r[0] for r in conn.execute("SELECT source FROM events")}
    assert sources == {"eval_1.jsonl"}


# --------------------------------------------------------------------------- #
# Real data from conversational-form-agent (data/sample)
# --------------------------------------------------------------------------- #


def test_sample_data_matches_the_numbers_reported_by_the_agent_project(conn):
    reports = ingest_paths(conn, [SAMPLE_DIR])
    assert all(r.invalid == 0 for r in reports)

    runs = {r["name"]: r for r in conn.execute("SELECT * FROM runs")}
    assert len(runs) == 4
    cases = dict(
        conn.execute(
            "SELECT r.name, count(*) FROM case_results c"
            " JOIN runs r ON r.id = c.run_id GROUP BY r.name"
        )
    )
    assert cases["20261119-225827_deepseek-flash_baseline"] == 75
    assert cases["20261121-152221_deepseek-flash"] == 90

    # Same totals as evals/results/20261121-152221_trace_summary.json in the agent repo.
    latest = runs["20261121-152221_deepseek-flash"]
    calls, errors, cost = conn.execute(
        "SELECT count(*), sum(ok = 0), round(sum(cost_usd), 6) FROM events"
        " WHERE source = ? AND type = 'llm_call'",
        (latest["trace_source"],),
    ).fetchone()
    assert (calls, errors, cost) == (385, 0, 0.058227)

    # Every case of a traced run can be joined to its trace events.
    unlinked = conn.execute(
        """
        SELECT count(*) FROM case_results c
        WHERE c.run_id = ? AND NOT EXISTS (
            SELECT 1 FROM events e WHERE e.conversation_id = c.conversation_id)
        """,
        (latest["id"],),
    ).fetchone()[0]
    assert unlinked == 0


def test_cli_ingest_is_idempotent(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "ingest", str(SAMPLE_DIR / "results")]) == 0
    assert main(["--db", db, "ingest", str(SAMPLE_DIR / "results")]) == 0
    out = capsys.readouterr().out
    assert "318 case result(s)" in out
    assert "0 case result(s)" in out


def test_cli_ingest_reports_missing_path(tmp_path):
    assert main(["--db", str(tmp_path / "x.db"), "ingest", str(tmp_path / "nope")]) == 2
