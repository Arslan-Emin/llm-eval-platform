import pytest

from evalplat import store, traces
from evalplat.ingest import FileReport, ingest_events, ingest_paths
from tests.factories import SAMPLE_DIR, make_event


@pytest.fixture(scope="module")
def sample_conn():
    conn = store.connect(":memory:")
    ingest_paths(conn, [SAMPLE_DIR])
    return conn


def test_trace_sources_are_linked_to_their_run(sample_conn):
    sources = {s["source"]: s for s in traces.list_trace_sources(sample_conn)}
    assert sources["eval_20261121-152221.jsonl"]["run_name"] == "20261121-152221_deepseek-flash"
    assert sources["eval_20261121-152221.jsonl"]["conversations"] == 90
    assert sources["2026-11-21.jsonl"]["run_name"] is None  # app traces, not an eval


def test_conversations_are_joined_to_eval_cases(sample_conn):
    conversations = traces.list_conversations(sample_conn, source="eval_20261121-152221.jsonl")
    assert len(conversations) == 90
    assert all(c["case_id"] for c in conversations)
    assert sum(c["llm_calls"] for c in conversations) == 385

    app = traces.list_conversations(sample_conn, source="2026-11-21.jsonl")
    assert [c["case_id"] for c in app] == [None]


def test_conversation_ids(sample_conn):
    # 348 eval conversations + 2 from the app's own traces. The first two runs were
    # recorded before the agent had tracing and carry no conversation ids at all.
    assert len(traces.conversation_ids(sample_conn)) == 350
    untraced = sample_conn.execute(
        "SELECT count(*) FROM case_results WHERE conversation_id IS NULL"
    ).fetchone()[0]
    assert untraced == 150


def test_conversation_trace_of_an_eval_case(sample_conn):
    conv = next(
        c
        for c in traces.list_conversations(sample_conn, source="eval_20261121-152221.jsonl")
        if c["case_id"] == "normal_all_in_one_tr"
    )
    trace = traces.conversation_trace(sample_conn, conv["conversation_id"])

    assert trace["case"]["case_id"] == "normal_all_in_one_tr"
    assert trace["checks"] and all(c["passed"] for c in trace["checks"])
    assert len(trace["transcript"]) == 2
    assert [t["turn_index"] for t in trace["turns"]] == [0, 1]
    assert trace["totals"]["has_content"] is True
    for turn in trace["turns"]:
        assert turn["summary"]["type"] == "agent_turn"
        for span in turn["spans"]:
            assert 0 <= span["start_ms"] <= span["end_ms"]
    # The last turn submits the form.
    labels = [s["label"] for s in trace["turns"][1]["spans"]]
    assert "submit_form" in labels


def test_spans_are_placed_on_one_time_axis():
    conn = store.connect(":memory:")
    events = [
        make_event("a", "llm_call", timestamp="2026-01-01T00:00:01.000+00:00", latency_ms=1000.0,
                   tool_calls=["update_fields"]),
        make_event("b", "tool_call", timestamp="2026-01-01T00:00:01.100+00:00", duration_ms=50.0,
                   name="update_fields"),
        make_event("c", "agent_turn", timestamp="2026-01-01T00:00:02.000+00:00",
                   duration_ms=2000.0),
    ]  # fmt: skip
    ingest_events(conn, events, source="t", report=FileReport("t", "traces"))

    trace = traces.conversation_trace(conn, "conv1")

    spans = {s["label"]: s for s in trace["turns"][0]["spans"]}
    assert spans["turn"]["start_ms"] == 0.0 and spans["turn"]["end_ms"] == 2000.0
    assert (spans["LLM -> update_fields"]["start_ms"], spans["LLM -> update_fields"]["end_ms"]) == (
        0.0,
        1000.0,
    )
    assert (spans["update_fields"]["start_ms"], spans["update_fields"]["end_ms"]) == (
        1050.0,
        1100.0,
    )
    assert trace["case"] is None
    assert trace["totals"]["has_content"] is False


def test_unknown_conversation_is_empty(sample_conn):
    trace = traces.conversation_trace(sample_conn, "nope")
    assert trace["case"] is None and trace["turns"] == []
