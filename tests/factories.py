"""Builders for eval results and trace events in the agent project's format."""

from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def make_case(case_id="c1", passed=True, conversation_id="conv1", checks=None):
    return {
        "id": case_id,
        "category": "normal",
        "conversation_id": conversation_id,
        "passed": passed,
        "error": None,
        "expected_submit": True,
        "submitted": passed,
        "checks": checks
        if checks is not None
        else [{"kind": "submit", "description": "submitted", "passed": passed, "turn": None}],
        "final_values": {"full_name": "Ali Veli"},
        "transcript": [{"user": "hi", "agent": "hello", "tool_calls": []}],
        "metrics": {"llm_calls": 4.0, "input_tokens": 100.0, "cost_usd": 0.001, "user_turns": 2.0},
    }


def make_results(cases, run_at="2026-11-21T15:22:21+00:00", trace_file=None):
    return {
        "run": {
            "run_at_utc": run_at,
            "provider": "deepseek",
            "model": "deepseek-flash",
            "repeats": 1,
            "trace_file": trace_file,
        },
        "summary": {"cases": {"passed": 1, "total": 1, "rate": 1.0}},
        "cases": cases,
    }


def make_event(event_id="e1", type_="llm_call", **fields):
    return {
        "schema_version": 1,
        "event_id": event_id,
        "type": type_,
        "timestamp": "2026-11-21T15:22:29.438+00:00",
        "conversation_id": "conv1",
        "turn_index": 0,
        **fields,
    }
