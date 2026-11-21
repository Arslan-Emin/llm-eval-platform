"""Metrics for one eval run and the comparison of two runs, computed from the store.

Everything is recomputed from case results, checks and trace events rather than
copied from the summary the runner wrote, so runs from any source are measured the
same way (tests check that both agree on the sample data).

Statistics are kept simple and honest about sample size:
- Percentiles use the nearest-rank method (same as the agent project's tracer).
- A pass rate comes with a 95% Wilson score interval: 90/90 means "at least ~96%",
  not "100%".
- Two runs are compared on the cases they have in common, with a one-sided Fisher
  exact test: "how likely is a difference at least this large if both runs had the
  same true pass rate?". With 3 repeats per case, small differences are often noise.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from typing import Any

Rate = dict[str, Any]  # {"passed": int, "total": int, "rate": float | None}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def rate(passed: int, total: int) -> Rate:
    return {"passed": passed, "total": total, "rate": passed / total if total else None}


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    # ceil, not round: Python's round() is banker's rounding (round(2.5) == 2).
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% confidence interval for a pass rate; unlike p ± z·se it stays inside [0, 1]
    and does not collapse to a single point when every run passed."""
    if total == 0:
        return None
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _hypergeometric(k: int, total_passed: int, drawn: int, population: int) -> float:
    return (
        math.comb(total_passed, k)
        * math.comb(population - total_passed, drawn - k)
        / math.comb(population, drawn)
    )


def fisher_p_worse(base_passed: int, base_total: int, cand_passed: int, cand_total: int) -> float:
    """One-sided Fisher exact test: probability that the candidate passes this few
    (or fewer) times if both runs share one true pass rate. Small = real drop."""
    total_passed = base_passed + cand_passed
    population = base_total + cand_total
    low = max(0, total_passed - base_total)
    return min(
        1.0,
        sum(
            _hypergeometric(k, total_passed, cand_total, population)
            for k in range(low, cand_passed + 1)
        ),
    )


def fisher_p_better(base_passed: int, base_total: int, cand_passed: int, cand_total: int) -> float:
    """Mirror of `fisher_p_worse`: small = real improvement."""
    return fisher_p_worse(
        base_total - base_passed, base_total, cand_total - cand_passed, cand_total
    )


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


class RunNotFound(LookupError):
    pass


def resolve_run(conn: sqlite3.Connection, ref: str | int) -> sqlite3.Row:
    """Find a run by id, exact name, a unique part of its name, or 'latest' / 'previous'."""
    text = str(ref)
    if text in {"latest", "previous"}:
        rows = conn.execute("SELECT * FROM runs ORDER BY run_at DESC LIMIT 2").fetchall()
        index = 0 if text == "latest" else 1
        if len(rows) <= index:
            raise RunNotFound(f"no {text} run")
        return rows[index]
    if text.isdigit():
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (int(text),)).fetchone()
        if row:
            return row
    row = conn.execute("SELECT * FROM runs WHERE name = ?", (text,)).fetchone()
    if row:
        return row
    # instr() is a plain substring test (LIKE would treat '_' in names as a wildcard).
    matches = conn.execute(
        "SELECT * FROM runs WHERE instr(name, ?) > 0 ORDER BY run_at", (text,)
    ).fetchall()
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = ", ".join(m["name"] for m in matches)
        raise RunNotFound(f"{text!r} matches several runs: {names}")
    raise RunNotFound(f"no run {text!r}")


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.id, r.name, r.run_at, r.model, r.thinking, r.repeats,
               r.trace_source IS NOT NULL AS traced,
               count(c.id) AS runs_total, sum(c.passed) AS runs_passed,
               count(DISTINCT c.case_id) AS unique_cases,
               sum(c.cost_usd) AS cost_usd
        FROM runs r LEFT JOIN case_results c ON c.run_id = r.id
        GROUP BY r.id ORDER BY r.run_at
        """
    ).fetchall()
    return [
        {
            **{k: row[k] for k in ("id", "name", "run_at", "model", "thinking", "repeats")},
            "traced": bool(row["traced"]),
            "unique_cases": row["unique_cases"],
            "cases": rate(row["runs_passed"] or 0, row["runs_total"]),
            "cost_per_conversation_usd": (row["cost_usd"] or 0) / row["runs_total"]
            if row["runs_total"]
            else None,
        }
        for row in rows
    ]


def _run_events(conn: sqlite3.Connection, run_id: int, event_type: str) -> list[sqlite3.Row]:
    # Joined on conversation_id rather than the trace file name: precise even if
    # one trace file holds several runs or events arrived live over HTTP.
    return conn.execute(
        """
        SELECT * FROM events
        WHERE type = ? AND conversation_id IN (
            SELECT conversation_id FROM case_results WHERE run_id = ?)
        """,
        (event_type, run_id),
    ).fetchall()


def run_metrics(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise RunNotFound(f"no run with id {run_id}")
    cases = conn.execute("SELECT * FROM case_results WHERE run_id = ?", (run_id,)).fetchall()
    checks = conn.execute(
        """
        SELECT ch.kind, ch.passed FROM checks ch
        JOIN case_results c ON c.id = ch.case_result_id WHERE c.run_id = ?
        """,
        (run_id,),
    ).fetchall()

    n = len(cases)
    passed = sum(c["passed"] for c in cases)

    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in cases:
        by_category[c["category"]][0] += c["passed"]
        by_category[c["category"]][1] += 1
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ch in checks:
        by_kind[ch["kind"]][0] += ch["passed"]
        by_kind[ch["kind"]][1] += 1

    no_submit = [c for c in cases if c["expected_submit"] == 0]
    should_submit = [c for c in cases if c["expected_submit"] == 1]

    # Tool statistics come from transcripts, which every run has (traces may not).
    tools: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "failed": 0})
    for c in cases:
        for turn in json.loads(c["transcript_json"] or "[]"):
            for call in turn.get("tool_calls", []):
                tools[call["name"]]["calls"] += 1
                tools[call["name"]]["failed"] += call.get("ok") is False

    def total(column: str) -> float:
        return sum(c[column] or 0 for c in cases)

    user_turns = total("user_turns")
    input_tokens = total("input_tokens")
    llm_calls = [e for e in _run_events(conn, run_id, "llm_call") if e["ok"]]
    turns = _run_events(conn, run_id, "agent_turn")
    llm_latency = [e["latency_ms"] for e in llm_calls]
    turn_latency = [e["latency_ms"] for e in turns]

    return {
        "run": {
            "id": run["id"],
            "name": run["name"],
            "run_at": run["run_at"],
            "provider": run["provider"],
            "model": run["model"],
            "thinking": run["thinking"],
            "repeats": run["repeats"],
            "cases_file": run["cases_file"],
            "traced": bool(turns),
        },
        "unique_cases": len({c["case_id"] for c in cases}),
        "cases": rate(passed, n),
        "cases_ci95": wilson_interval(passed, n),
        "errors": sum(c["error"] is not None for c in cases),
        "by_category": {k: rate(*v) for k, v in sorted(by_category.items())},
        "checks": {k: rate(*v) for k, v in sorted(by_kind.items())},
        # Submitted although it should not have / did not submit although it should.
        "wrong_submit": rate(sum(c["submitted"] == 1 for c in no_submit), len(no_submit)),
        "missed_submit": rate(sum(c["submitted"] == 0 for c in should_submit), len(should_submit)),
        "tools": dict(sorted(tools.items())),
        "cost_usd": {
            "total": total("cost_usd"),
            "per_conversation": total("cost_usd") / n if n else None,
        },
        "tokens": {
            "input_per_conversation": input_tokens / n if n else None,
            "output_per_conversation": total("output_tokens") / n if n else None,
            "cache_hit_rate": total("cached_input_tokens") / input_tokens if input_tokens else None,
        },
        "latency_ms": {
            "turn_avg": total("latency_ms") / user_turns if user_turns else None,
            # Percentiles need per-turn data, which only traced runs have.
            "turn_p50": percentile(turn_latency, 50),
            "turn_p95": percentile(turn_latency, 95),
            "llm_call_p50": percentile(llm_latency, 50),
            "llm_call_p95": percentile(llm_latency, 95),
        },
    }


# --------------------------------------------------------------------------- #
# Comparing two runs
# --------------------------------------------------------------------------- #


def _case_stats(conn: sqlite3.Connection, run_id: int) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    rows = conn.execute(
        """
        SELECT c.case_id, c.category, c.passed, c.error,
               (SELECT json_group_array(ch.description) FROM checks ch
                WHERE ch.case_result_id = c.id AND ch.passed = 0) AS failed_checks
        FROM case_results c WHERE c.run_id = ? ORDER BY c.case_id, c.repeat_index
        """,
        (run_id,),
    ).fetchall()
    for row in rows:
        entry = stats.setdefault(
            row["case_id"],
            {"category": row["category"], "passed": 0, "total": 0, "failures": []},
        )
        entry["passed"] += row["passed"]
        entry["total"] += 1
        failures = json.loads(row["failed_checks"]) + ([row["error"]] if row["error"] else [])
        for failure in failures:
            if failure not in entry["failures"]:
                entry["failures"].append(failure)
    return stats


def compare_runs(conn: sqlite3.Connection, baseline_id: int, candidate_id: int) -> dict[str, Any]:
    """Compare a candidate run against a baseline, case by case and overall."""
    base_cases = _case_stats(conn, baseline_id)
    cand_cases = _case_stats(conn, candidate_id)

    case_rows = []
    for case_id in sorted(base_cases.keys() | cand_cases.keys()):
        base, cand = base_cases.get(case_id), cand_cases.get(case_id)
        if base is None:
            status = "added"
        elif cand is None:
            status = "removed"
        else:
            base_rate = base["passed"] / base["total"]
            cand_rate = cand["passed"] / cand["total"]
            status = (
                "regressed"
                if cand_rate < base_rate
                else "improved"
                if cand_rate > base_rate
                else "unchanged"
            )
        case_rows.append(
            {
                "case_id": case_id,
                "category": (cand or base)["category"],
                "status": status,
                "baseline": None if base is None else rate(base["passed"], base["total"]),
                "candidate": None if cand is None else rate(cand["passed"], cand["total"]),
                # Why it failed: from the candidate for a regression, else the baseline.
                "failures": (cand or {}).get("failures", [])
                if status in {"regressed", "added", "unchanged"}
                else (base or {}).get("failures", []),
                # Passed some repeats and failed others: non-deterministic behaviour.
                "flaky_in_candidate": cand is not None and 0 < cand["passed"] < cand["total"],
            }
        )

    common = [r for r in case_rows if r["baseline"] and r["candidate"]]
    base_passed = sum(r["baseline"]["passed"] for r in common)
    base_total = sum(r["baseline"]["total"] for r in common)
    cand_passed = sum(r["candidate"]["passed"] for r in common)
    cand_total = sum(r["candidate"]["total"] for r in common)

    def ids(status: str) -> list[str]:
        return [r["case_id"] for r in case_rows if r["status"] == status]

    return {
        "baseline": run_metrics(conn, baseline_id),
        "candidate": run_metrics(conn, candidate_id),
        "common_cases": {
            "count": len(common),
            "baseline": rate(base_passed, base_total),
            "candidate": rate(cand_passed, cand_total),
            "p_worse": fisher_p_worse(base_passed, base_total, cand_passed, cand_total)
            if common
            else None,
            "p_better": fisher_p_better(base_passed, base_total, cand_passed, cand_total)
            if common
            else None,
        },
        "regressed": ids("regressed"),
        "improved": ids("improved"),
        "added": ids("added"),
        "removed": ids("removed"),
        "cases": case_rows,
    }
