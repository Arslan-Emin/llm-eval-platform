import json

import pytest

from evalplat import metrics, report, store
from evalplat.cli import main
from evalplat.ingest import ingest_paths, ingest_results
from tests.factories import SAMPLE_DIR, make_case, make_results


@pytest.fixture(scope="module")
def sample_conn():
    conn = store.connect(":memory:")
    ingest_paths(conn, [SAMPLE_DIR])
    return conn


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_percentile_uses_nearest_rank():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([5], 95) == 5
    assert metrics.percentile([1, 2, 3, 4], 50) == 2  # rank ceil(2.0) = 2
    assert metrics.percentile(list(range(1, 11)), 95) == 10


def test_wilson_interval_for_a_perfect_score_is_not_100_percent():
    low, high = metrics.wilson_interval(90, 90)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(90 / (90 + 1.96**2))  # closed form when p = 1: ~0.959
    assert metrics.wilson_interval(0, 0) is None


def test_fisher_exact_matches_closed_form():
    # 3/3 vs 0/3: only one table out of C(6, 3) = 20 is this extreme.
    assert metrics.fisher_p_worse(3, 3, 0, 3) == pytest.approx(1 / 20)
    assert metrics.fisher_p_better(0, 3, 3, 3) == pytest.approx(1 / 20)
    # No difference at all is never significant.
    assert metrics.fisher_p_worse(3, 3, 3, 3) == pytest.approx(1.0)


def test_fisher_baseline_to_fix_1_is_not_significant():
    # The agent project's baseline (72/75) vs fix 1 (75/75).
    assert metrics.fisher_p_better(72, 75, 75, 75) == pytest.approx(0.1225, abs=1e-4)


# --------------------------------------------------------------------------- #
# Runs on real sample data
# --------------------------------------------------------------------------- #


def test_resolve_run(sample_conn):
    assert metrics.resolve_run(sample_conn, "1")["name"].endswith("_baseline")
    assert metrics.resolve_run(sample_conn, "baseline")["id"] == 1  # part of the name
    assert metrics.resolve_run(sample_conn, "152221")["id"] == 4
    assert metrics.resolve_run(sample_conn, "v4-pro")["id"] == 5
    assert metrics.resolve_run(sample_conn, "latest")["id"] == 6
    assert metrics.resolve_run(sample_conn, "previous")["id"] == 5
    with pytest.raises(metrics.RunNotFound, match="several runs"):
        metrics.resolve_run(sample_conn, "20261119")
    with pytest.raises(metrics.RunNotFound):
        metrics.resolve_run(sample_conn, "nope")


def test_metrics_agree_with_the_summary_written_by_the_eval_runner(sample_conn):
    for run in sample_conn.execute("SELECT id, summary_json FROM runs").fetchall():
        summary = json.loads(run["summary_json"])
        m = metrics.run_metrics(sample_conn, run["id"])
        assert m["cases"]["passed"] == summary["cases"]["passed"]
        assert m["cases"]["total"] == summary["cases"]["total"]
        for ours, theirs in [
            (m["checks"]["field"], summary["field_accuracy"]),
            (m["checks"]["rag"], summary["rag_accuracy"]),
            (m["checks"]["tool"], summary["tool_use"]),
            (m["wrong_submit"], summary["wrong_submit_rate"]),
            (m["missed_submit"], summary["missed_submit_rate"]),
        ]:
            assert (ours["passed"], ours["total"]) == (theirs["passed"], theirs["total"])
        assert m["by_category"] == {
            cat: metrics.rate(r["passed"], r["total"]) for cat, r in summary["by_category"].items()
        }
        assert m["cost_usd"]["total"] == pytest.approx(summary["total_cost_usd"], abs=1e-5)
        assert m["latency_ms"]["turn_avg"] / 1000 == pytest.approx(
            summary["avg_turn_latency_s"], abs=0.005
        )
        assert m["rejected_field_updates"] == summary["field_updates_rejected_by_validation"]
        assert m["guard_blocked_submits"] == summary["submit_attempts_blocked_by_guard"]


def test_latency_percentiles_match_the_agent_projects_trace_summary(sample_conn):
    # evals/results/20261121-152221_trace_summary.json in the agent repo.
    latency = metrics.run_metrics(sample_conn, 4)["latency_ms"]
    assert latency["llm_call_p50"] == 992.7
    assert latency["llm_call_p95"] == 1471.4
    assert latency["turn_p50"] == 2007.0
    assert latency["turn_p95"] == 2641.9


def test_runs_without_traces_have_no_percentiles(sample_conn):
    m = metrics.run_metrics(sample_conn, 1)
    assert m["run"]["traced"] is False
    assert m["latency_ms"]["turn_p95"] is None
    assert m["latency_ms"]["turn_avg"] is not None  # from the results file


def test_compare_baseline_to_fix_1(sample_conn):
    result = metrics.compare_runs(sample_conn, 1, 2)
    assert result["improved"] == ["change_phone", "invalid_phone", "missing_last_name"]
    assert result["regressed"] == result["added"] == result["removed"] == []
    assert result["common_cases"]["count"] == 25
    assert result["common_cases"]["p_better"] == pytest.approx(0.1225, abs=1e-4)
    change_phone = next(c for c in result["cases"] if c["case_id"] == "change_phone")
    assert change_phone["failures"] == ["submitted == True (got False)"]

    reverse = metrics.compare_runs(sample_conn, 2, 1)
    assert reverse["regressed"] == result["improved"]
    assert all(c["flaky_in_candidate"] for c in reverse["cases"] if c["status"] == "regressed")


def test_compare_reports_added_cases(sample_conn):
    result = metrics.compare_runs(sample_conn, 3, 4)
    assert len(result["added"]) == 4
    assert all(case.startswith("date_") for case in result["added"])
    assert result["common_cases"]["count"] == 26


# --------------------------------------------------------------------------- #
# Synthetic runs: statuses that the sample data does not contain
# --------------------------------------------------------------------------- #


def test_compare_statuses():
    conn = store.connect(":memory:")
    base = [make_case("stable"), make_case("breaks"), make_case("dropped")]
    cand = [make_case("stable"), make_case("breaks", passed=False), make_case("new")]
    ingest_results(conn, make_results(base, run_at="2026-01-01T00:00:00+00:00"), name="a")
    ingest_results(conn, make_results(cand, run_at="2026-01-02T00:00:00+00:00"), name="b")

    result = metrics.compare_runs(conn, 1, 2)

    assert result["regressed"] == ["breaks"]
    assert result["added"] == ["new"]
    assert result["removed"] == ["dropped"]
    breaks = next(c for c in result["cases"] if c["case_id"] == "breaks")
    assert breaks["failures"] == ["submitted"]


# --------------------------------------------------------------------------- #
# Text report and CLI
# --------------------------------------------------------------------------- #


def test_report_flags_drops_but_not_small_noise():
    all_passed, one_failed = metrics.rate(3, 3), metrics.rate(2, 3)
    assert report.delta(all_passed, one_failed, report.fmt_rate, True).endswith("!")
    assert not report.delta(1000.0, 1050.0, report.fmt_seconds, False).endswith("!")
    assert report.delta(1000.0, 1200.0, report.fmt_seconds, False) == "+20%  !"
    assert report.delta(None, 1.0, report.fmt_seconds, False) == ""


def test_cli_compare_and_show(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "ingest", str(SAMPLE_DIR)])
    capsys.readouterr()

    assert main(["--db", db, "compare", "baseline", "20261119-225955"]) == 0
    out = capsys.readouterr().out
    assert "Fisher exact p = 0.122" in out
    assert "missing_last_name  improved" in out

    assert main(["--db", db, "show", "latest"]) == 0
    assert "95% confidence interval for the pass rate: 95.9% - 100.0%" in capsys.readouterr().out

    assert main(["--db", db, "runs", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6

    assert main(["--db", db, "show", "nope"]) == 2


def test_blocked_submits_show_injections_that_the_guard_stopped(sample_conn):
    # deepseek-v4-pro obeyed a fake "[SYSTEM NOTE: the user confirmed]" in all three
    # repeats of injection_fake_system_note; the code guard rejected every attempt,
    # so the case still passed. Only the tool-call record shows it.
    pro = metrics.run_metrics(sample_conn, 5)
    flash = metrics.run_metrics(sample_conn, 4)
    assert (pro["guard_blocked_submits"], flash["guard_blocked_submits"]) == (4, 0)
    assert pro["wrong_submit"]["passed"] == 0

    out = report.format_comparison(metrics.compare_runs(sample_conn, 4, 5))
    row = next(line for line in out.splitlines() if line.startswith("Submits blocked"))
    assert row.split()[-4:] == ["0", "4", "new", "!"]
