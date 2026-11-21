"""Read trace events back as conversations, turns and timed spans.

A trace event is written when its work *ends*, and carries how long it took
(`latency_ms` for LLM calls, `duration_ms` for tool calls and agent turns). So each
event becomes a span `[timestamp - duration, timestamp]`; laid out on one time axis
they show where a conversation spent its time (LLM waits vs tools).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

SPAN_DURATION_FIELD = {
    "llm_call": "latency_ms",
    "tool_call": "duration_ms",
    "agent_turn": "duration_ms",
}


def list_trace_sources(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per trace file (or 'live'), newest first, with the run it belongs to."""
    rows = conn.execute(
        """
        SELECT e.source, count(*) AS events,
               count(DISTINCT e.conversation_id) AS conversations,
               min(e.timestamp) AS first_event, max(e.timestamp) AS last_event,
               (SELECT r.name FROM runs r WHERE r.trace_source = e.source) AS run_name
        FROM events e GROUP BY e.source ORDER BY last_event DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def list_conversations(
    conn: sqlite3.Connection, *, source: str | None = None, limit: int = 1000
) -> list[dict[str, Any]]:
    """Traced conversations with totals, joined to their eval case when there is one."""
    rows = conn.execute(
        f"""
        WITH conv AS (
            SELECT conversation_id,
                   min(source) AS source,
                   min(timestamp) AS started_at,
                   sum(type = 'agent_turn') AS turns,
                   sum(type = 'llm_call') AS llm_calls,
                   sum(type = 'llm_call' AND ok = 0) AS llm_errors,
                   sum(type = 'tool_call') AS tool_calls,
                   sum(type = 'tool_call' AND ok = 0) AS failed_tool_calls,
                   sum(CASE WHEN type = 'llm_call' THEN cost_usd END) AS cost_usd,
                   sum(CASE WHEN type = 'agent_turn' THEN duration_ms END) AS duration_ms,
                   max(CASE WHEN type = 'agent_turn'
                       THEN json_extract(data_json, '$.submitted') END) AS submitted
            FROM events
            WHERE conversation_id IS NOT NULL {"AND source = ?" if source else ""}
            GROUP BY conversation_id
        )
        SELECT conv.*, c.case_id, c.repeat_index, c.passed, r.name AS run_name
        FROM conv
        LEFT JOIN case_results c ON c.conversation_id = conv.conversation_id
        LEFT JOIN runs r ON r.id = c.run_id
        ORDER BY conv.started_at DESC
        LIMIT ?
        """,
        (source, limit) if source else (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _parse_time(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)


def _span_label(event: dict[str, Any]) -> str:
    if event["type"] == "tool_call":
        return event.get("name") or "tool"
    if event["type"] == "llm_call":
        calls = event.get("tool_calls") or []
        return f"LLM -> {', '.join(calls)}" if calls else "LLM -> reply"
    return "turn"


def conversation_trace(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any]:
    """Everything known about one conversation: its eval case (if any) and its trace,
    grouped by turn, with each event turned into a timed span."""
    case = conn.execute(
        """
        SELECT c.*, r.name AS run_name FROM case_results c JOIN runs r ON r.id = c.run_id
        WHERE c.conversation_id = ?
        """,
        (conversation_id,),
    ).fetchone()
    checks = (
        [
            dict(row)
            for row in conn.execute(
                "SELECT kind, description, passed, turn FROM checks"
                " WHERE case_result_id = ? ORDER BY position",
                (case["id"],),
            )
        ]
        if case
        else []
    )
    rows = conn.execute(
        "SELECT data_json FROM events WHERE conversation_id = ? ORDER BY timestamp, rowid",
        (conversation_id,),
    ).fetchall()
    events = [json.loads(row["data_json"]) for row in rows]

    spans = []
    for event in events:
        end = _parse_time(event["timestamp"])
        duration = event.get(SPAN_DURATION_FIELD[event["type"]]) or 0.0
        spans.append({"event": event, "end": end, "duration_ms": duration})
    origin = min((s["end"].timestamp() * 1000 - s["duration_ms"] for s in spans), default=0.0)

    turns: dict[int, dict[str, Any]] = {}
    for span in spans:
        event = span["event"]
        end_ms = span["end"].timestamp() * 1000 - origin
        turn = turns.setdefault(
            event.get("turn_index", -1),
            {"turn_index": event.get("turn_index"), "summary": None, "spans": []},
        )
        if event["type"] == "agent_turn":
            turn["summary"] = event
        turn["spans"].append(
            {
                "type": event["type"],
                "label": _span_label(event),
                "start_ms": round(end_ms - span["duration_ms"], 1),
                "end_ms": round(end_ms, 1),
                "duration_ms": span["duration_ms"],
                "ok": event.get("ok"),
                "event": event,
            }
        )

    llm_calls = [e for e in events if e["type"] == "llm_call"]
    return {
        "conversation_id": conversation_id,
        "case": dict(case) if case else None,
        "checks": checks,
        "transcript": json.loads(case["transcript_json"] or "[]") if case else [],
        "final_values": json.loads(case["final_values_json"] or "null") if case else None,
        "turns": [turns[key] for key in sorted(turns, key=lambda k: (k is None, k))],
        "totals": {
            "events": len(events),
            "llm_calls": len(llm_calls),
            "cost_usd": sum(e.get("cost_usd") or 0 for e in llm_calls),
            "input_tokens": sum(e.get("input_tokens") or 0 for e in llm_calls),
            "output_tokens": sum(e.get("output_tokens") or 0 for e in llm_calls),
            "has_content": any("output_text" in e or "user_text" in e for e in events),
        },
    }


def conversation_ids(conn: sqlite3.Connection) -> list[str]:
    """Every conversation the store knows, traced or not, newest first."""
    rows = conn.execute(
        """
        SELECT conversation_id, max(started) AS started FROM (
            SELECT conversation_id, min(timestamp) AS started FROM events
            WHERE conversation_id IS NOT NULL GROUP BY conversation_id
            UNION ALL
            SELECT c.conversation_id, r.run_at FROM case_results c JOIN runs r ON r.id = c.run_id
            WHERE c.conversation_id IS NOT NULL
        ) GROUP BY conversation_id ORDER BY started DESC
        """
    ).fetchall()
    return [row["conversation_id"] for row in rows]
