"""Smoke tests: every dashboard page renders against the sample data without errors.

Streamlit's AppTest runs a page script headless; layout and charts are checked by
eye in a browser, these tests catch exceptions and wrong numbers on the pages.
"""

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from evalplat import store  # noqa: E402
from evalplat.ingest import ingest_paths  # noqa: E402
from tests.factories import SAMPLE_DIR  # noqa: E402


def render_page(page: str) -> None:
    # Runs inside AppTest; imports must be local because only this function's
    # source is executed.
    from evalplat import dashboard

    getattr(dashboard, page)()


@pytest.fixture
def sample_db(tmp_path, monkeypatch):
    path = tmp_path / "dashboard.db"
    conn = store.connect(path)
    ingest_paths(conn, [SAMPLE_DIR])
    conn.close()
    monkeypatch.setenv(store.DB_PATH_ENV, str(path))
    monkeypatch.delenv("EVALPLAT_SEED_DIR", raising=False)
    return path


def run(page: str, **query: str) -> AppTest:
    app = AppTest.from_function(render_page, args=(page,), default_timeout=30)
    for key, value in query.items():
        app.query_params[key] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_runs_page(sample_db):
    app = run("page_runs")
    assert app.title[0].value == "Eval runs"
    assert [m.value for m in app.metric][:2] == ["100.0%", "0.0%"]


def test_run_page_shows_the_runs_failures(sample_db):
    app = run("page_run", run="20261119-225827_deepseek-flash_baseline")
    assert app.metric[0].value == "96.0%"
    assert any("Failed conversations (3)" in s.value for s in app.subheader)


def test_compare_page_reports_significance(sample_db):
    app = run(
        "page_compare",
        baseline="20261119-225827_deepseek-flash_baseline",
        candidate="20261119-225955_deepseek-flash",
    )
    # Not significant, so shown as neutral information rather than a success.
    assert "Fisher exact p = 0.122" in app.info[0].value
    assert not app.success


def test_traces_page(sample_db):
    app = run("page_traces")
    assert "350 conversations" in app.caption[0].value


def test_conversation_page(sample_db):
    conn = store.connect(sample_db)
    conversation_id = conn.execute(
        "SELECT conversation_id FROM case_results WHERE case_id = 'change_phone'"
        " AND conversation_id IS NOT NULL LIMIT 1"
    ).fetchone()[0]
    app = run("page_conversation", conversation=conversation_id)
    assert any("change_phone" in m.value for m in app.markdown)
    assert any(s.value == "Timeline" for s in app.subheader)


def test_empty_database_shows_how_to_load_data(tmp_path, monkeypatch):
    monkeypatch.setenv(store.DB_PATH_ENV, str(tmp_path / "empty.db"))
    monkeypatch.delenv("EVALPLAT_SEED_DIR", raising=False)
    app = run("page_runs")
    assert "evalplat ingest" in app.info[0].value


def test_empty_database_is_seeded_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv(store.DB_PATH_ENV, str(tmp_path / "seeded.db"))
    monkeypatch.setenv("EVALPLAT_SEED_DIR", str(SAMPLE_DIR))
    app = run("page_runs")
    assert app.metric[0].value == "100.0%"
