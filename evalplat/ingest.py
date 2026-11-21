"""Load eval result files and JSONL trace files into the store.

Two input formats, both produced by the conversational-form-agent project:

- Eval results (`evals/results/*.json`): `{"run": ..., "summary": ..., "cases": [...]}`.
  A run is identified by (`run_at_utc`, `model`).
- Trace events (`logs/traces/*.jsonl`): one JSON event per line, identified by
  `event_id`.

Ingestion is idempotent: loading the same file twice adds nothing the second time,
so `evalplat ingest <folder>` can simply be re-run after new files appear. Each file
is written in one transaction, so a broken file never leaves half a run behind.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

from evalplat.store import transaction

SUPPORTED_EVENT_SCHEMA_VERSIONS = {1}
EVENT_TYPES = {"llm_call", "tool_call", "agent_turn"}
LIVE_SOURCE = "live"


class IngestError(ValueError):
    """The file is not in a format evalplat understands."""


@dataclass
class FileReport:
    path: str
    kind: str  # results | traces | skipped
    added: int = 0
    duplicates: int = 0
    invalid: int = 0
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        counts = f"added {self.added}, duplicates {self.duplicates}, invalid {self.invalid}"
        notes = f"  ({'; '.join(self.notes)})" if self.notes else ""
        return f"[{self.kind}] {self.path}: {counts}{notes}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _int(value: Any) -> int | None:
    # Per-case metrics are sums written as floats (e.g. "llm_calls": 4.0).
    return None if value is None else int(round(value))


def _bool(value: Any) -> int | None:
    return None if value is None else int(bool(value))


# --------------------------------------------------------------------------- #
# Eval results
# --------------------------------------------------------------------------- #


def ingest_results(
    conn: sqlite3.Connection, payload: dict[str, Any], *, name: str, source_path: str | None = None
) -> FileReport:
    """Store one eval run with its case results and checks."""
    report = FileReport(source_path or name, "results")
    try:
        run, summary, cases = payload["run"], payload["summary"], payload["cases"]
        run_at, model = run["run_at_utc"], run["model"]
    except (KeyError, TypeError) as error:
        raise IngestError(f"not an eval results file, missing {error}") from None

    trace_file = run.get("trace_file")
    with transaction(conn):
        existing = conn.execute(
            "SELECT id FROM runs WHERE run_at = ? AND model = ?", (run_at, model)
        ).fetchone()
        if existing:
            report.duplicates = 1
            report.notes.append("run already ingested")
            return report

        run_id = conn.execute(
            """
            INSERT INTO runs (name, run_at, provider, model, thinking, eval_clock, cases_file,
                              repeats, trace_source, wall_time_s, summary_json, source_path,
                              ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                run_at,
                run.get("provider"),
                model,
                run.get("deepseek_thinking"),
                run.get("eval_clock"),
                run.get("cases_file"),
                run.get("repeats"),
                # The runner writes a relative path; traces are matched by file name.
                PurePath(trace_file).name if trace_file else None,
                run.get("wall_time_s"),
                json.dumps(summary, ensure_ascii=False),
                source_path,
                _now(),
            ),
        ).lastrowid

        repeat_counter: dict[str, int] = defaultdict(int)
        for position, case in enumerate(cases):
            try:
                repeat_index = repeat_counter[case["id"]]
                repeat_counter[case["id"]] += 1
                _insert_case(conn, run_id, case, repeat_index)
            except (KeyError, TypeError) as error:
                # Raising inside the transaction rolls back the whole run.
                raise IngestError(f"case #{position}: missing {error}") from None
            report.added += 1
    return report


def _insert_case(
    conn: sqlite3.Connection, run_id: int, case: dict[str, Any], repeat_index: int
) -> None:
    metrics = case.get("metrics", {})
    case_row_id = conn.execute(
        """
        INSERT INTO case_results (run_id, case_id, repeat_index, category, conversation_id,
                                  passed, error, expected_submit, submitted, user_turns,
                                  llm_calls, input_tokens, cached_input_tokens, output_tokens,
                                  latency_ms, cost_usd, wall_time_s, final_values_json,
                                  transcript_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            case["id"],
            repeat_index,
            case["category"],
            case.get("conversation_id"),
            _bool(case["passed"]),
            case.get("error"),
            _bool(case.get("expected_submit")),
            _bool(case.get("submitted")),
            _int(metrics.get("user_turns")),
            _int(metrics.get("llm_calls")),
            _int(metrics.get("input_tokens")),
            _int(metrics.get("cached_input_tokens")),
            _int(metrics.get("output_tokens")),
            metrics.get("latency_ms"),
            metrics.get("cost_usd"),
            metrics.get("wall_time_s"),
            json.dumps(case.get("final_values"), ensure_ascii=False),
            json.dumps(case.get("transcript"), ensure_ascii=False),
        ),
    ).lastrowid
    conn.executemany(
        """
        INSERT INTO checks (case_result_id, position, kind, description, passed, turn)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (case_row_id, position, c["kind"], c["description"], _bool(c["passed"]), c["turn"])
            for position, c in enumerate(case.get("checks", []))
        ],
    )


# --------------------------------------------------------------------------- #
# Trace events
# --------------------------------------------------------------------------- #


def validate_event(event: Any) -> str | None:
    """Return why the event cannot be stored, or None if it is fine."""
    if not isinstance(event, dict):
        return "not a JSON object"
    version = event.get("schema_version")
    if version not in SUPPORTED_EVENT_SCHEMA_VERSIONS:
        return f"unsupported schema_version {version!r}"
    for key in ("event_id", "type", "timestamp"):
        if not isinstance(event.get(key), str) or not event[key]:
            return f"missing {key}"
    if event["type"] not in EVENT_TYPES:
        return f"unknown event type {event['type']!r}"
    return None


def ingest_events(
    conn: sqlite3.Connection, events: Iterable[Any], *, source: str, report: FileReport
) -> FileReport:
    """Store trace events; events already in the store (same event_id) are skipped."""
    with transaction(conn):
        for event in events:
            problem = validate_event(event)
            if problem:
                report.invalid += 1
                if len(report.notes) < 3:
                    report.notes.append(problem)
                continue
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO events (event_id, source, type, timestamp, conversation_id,
                                              turn_index, name, model, ok, input_tokens,
                                              cached_input_tokens, output_tokens, cost_usd,
                                              latency_ms, duration_ms, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    source,
                    event["type"],
                    event["timestamp"],
                    event.get("conversation_id"),
                    event.get("turn_index"),
                    event.get("name") if event["type"] == "tool_call" else None,
                    event.get("model"),
                    _bool(event.get("ok")),
                    event.get("input_tokens"),
                    event.get("cached_input_tokens"),
                    event.get("output_tokens"),
                    event.get("cost_usd"),
                    event.get("latency_ms"),
                    event.get("duration_ms"),
                    json.dumps(event, ensure_ascii=False),
                ),
            )
            if cursor.rowcount:
                report.added += 1
            else:
                report.duplicates += 1
    return report


def ingest_trace_file(conn: sqlite3.Connection, path: Path) -> FileReport:
    report = FileReport(str(path), "traces")
    events: list[Any] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                report.invalid += 1
                if len(report.notes) < 3:
                    report.notes.append(f"line {number}: invalid JSON")
    return ingest_events(conn, events, source=path.name, report=report)


# --------------------------------------------------------------------------- #
# Files and folders
# --------------------------------------------------------------------------- #


def ingest_file(conn: sqlite3.Connection, path: Path) -> FileReport:
    if path.suffix == ".jsonl":
        return ingest_trace_file(conn, path)
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return FileReport(str(path), "skipped", invalid=1, notes=[f"invalid JSON: {error}"])
        # Other JSON files next to the results (trace summaries, retrieval
        # benchmarks) are not eval runs; skip them instead of failing.
        if not (isinstance(payload, dict) and {"run", "summary", "cases"} <= payload.keys()):
            return FileReport(str(path), "skipped", notes=["not an eval results file"])
        try:
            return ingest_results(conn, payload, name=path.stem, source_path=str(path))
        except IngestError as error:
            return FileReport(str(path), "skipped", invalid=1, notes=[str(error)])
    return FileReport(str(path), "skipped", notes=["unsupported file type"])


def ingest_paths(conn: sqlite3.Connection, paths: Iterable[Path]) -> list[FileReport]:
    """Ingest files and (recursively) folders, in a stable order."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl"})
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return [ingest_file(conn, file) for file in files]
