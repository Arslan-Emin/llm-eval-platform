"""Regression gate: fail CI when a candidate eval run is worse than the baseline.

    evalplat gate evals/results/<baseline>.json evals/results/   # newest run in the folder

LLM agents are not deterministic, so "any drop fails" would make CI fail on noise,
and "only statistically significant drops fail" would let a broken case through.
The gate is a hybrid:

FAIL (exit code 1)
- safety: the candidate submitted a form it must not submit, or a case crashed
- a significant drop of the pass rate on the cases both runs share (one-sided
  Fisher exact test, p < 0.05)
- a case that newly fails in most of its repeats (pass rate down by more than half),
  or a new case that fails in most of its repeats

WARN (exit code 0, shown in the report)
- a case that lost a single repeat (flaky), a smaller drop that is not significant
- cost per conversation or latency per user turn up by 25% or more
- more submit attempts blocked by the code guard (the model tried to submit early)

Only the standard library is used, so the gate installs in seconds in another
project's CI. Inputs are eval result files (or a folder: its newest run is used).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evalplat import metrics, store
from evalplat.ingest import IngestError, ingest_results

STEP_SUMMARY_ENV = "GITHUB_STEP_SUMMARY"


@dataclass(frozen=True)
class GateRules:
    alpha: float = 0.05  # significance level for the overall pass-rate drop
    max_case_drop: float = 0.5  # a case losing more than this share of its repeats fails
    max_cost_increase: float = 0.25  # warn above this relative increase
    max_latency_increase: float = 0.25


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str


@dataclass
class GateResult:
    baseline: str
    candidate: str
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = {c.status for c in self.checks}
        return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "fail" else 0

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))


class GateInputError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Loading runs
# --------------------------------------------------------------------------- #


def _results_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ok = isinstance(payload, dict) and {"run", "summary", "cases"} <= payload.keys()
    return payload if ok else None


def resolve_input(path: Path) -> Path:
    """A results file, or the newest results file (by run time) in a folder."""
    if path.is_file():
        return path
    if path.is_dir():
        runs = [(p, _results_payload(p)) for p in sorted(path.glob("*.json"))]
        runs = [(p, payload) for p, payload in runs if payload]
        if not runs:
            raise GateInputError(f"no eval result files in {path}")
        return max(runs, key=lambda item: item[1]["run"]["run_at_utc"])[0]
    raise GateInputError(f"no such file or folder: {path}")


def _load(conn: sqlite3.Connection, path: Path) -> int:
    payload = _results_payload(path)
    if payload is None:
        raise GateInputError(f"not an eval results file: {path}")
    try:
        ingest_results(conn, payload, name=path.stem, source_path=str(path))
    except IngestError as error:
        raise GateInputError(f"{path}: {error}") from None
    row = conn.execute(
        "SELECT id FROM runs WHERE run_at = ? AND model = ?",
        (payload["run"]["run_at_utc"], payload["run"]["model"]),
    ).fetchone()
    return row["id"]


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def _relative(base: float | None, cand: float | None) -> float | None:
    return None if base in (None, 0) or cand is None else (cand - base) / base


def evaluate(comparison: dict[str, Any], rules: GateRules = GateRules()) -> GateResult:
    """Apply the gate rules to a `metrics.compare_runs` result."""
    base, cand = comparison["baseline"], comparison["candidate"]
    result = GateResult(base["run"]["name"], cand["run"]["name"])

    # Safety: hard limits, independent of the baseline.
    wrong = cand["wrong_submit"]
    if wrong["passed"]:
        result.add(
            "Wrong or premature submits",
            "fail",
            f"{wrong['passed']} of {wrong['total']} conversations submitted a form "
            "they must not submit",
        )
    else:
        result.add("Wrong or premature submits", "ok", f"0 of {wrong['total']}")
    if cand["errors"]:
        result.add("Crashed cases", "fail", f"{cand['errors']} case(s) raised an error")
    else:
        result.add("Crashed cases", "ok", "none")

    # Overall pass rate on the shared cases.
    common = comparison["common_cases"]
    if common["count"]:
        b, c = common["baseline"], common["candidate"]
        text = (
            f"{b['passed']}/{b['total']} -> {c['passed']}/{c['total']} on "
            f"{common['count']} shared cases"
        )
        if c["rate"] < b["rate"]:
            p = common["p_worse"]
            significant = p < rules.alpha
            result.add(
                "Pass rate",
                "fail" if significant else "warn",
                f"{text}, one-sided Fisher p = {p:.3f} "
                f"({'significant' if significant else 'could be noise'})",
            )
        else:
            result.add("Pass rate", "ok", text)
    else:
        result.add("Pass rate", "warn", "the runs share no cases; nothing to compare")

    # Per case: broken (fail) vs flaky (warn).
    broken, flaky = [], []
    for case in comparison["cases"]:
        cand_rate = case["candidate"]["rate"] if case["candidate"] else None
        if case["status"] == "regressed":
            drop = case["baseline"]["rate"] - cand_rate
            line = (
                f"{case['case_id']} {case['baseline']['passed']}/{case['baseline']['total']}"
                f" -> {case['candidate']['passed']}/{case['candidate']['total']}"
            )
            (broken if drop > rules.max_case_drop else flaky).append(line)
        elif case["status"] == "added" and cand_rate is not None and cand_rate < 1:
            new = case["candidate"]
            line = f"{case['case_id']} (new) {new['passed']}/{new['total']}"
            (broken if cand_rate < 1 - rules.max_case_drop else flaky).append(line)
    if broken:
        result.add("Broken cases", "fail", "; ".join(broken))
    else:
        result.add("Broken cases", "ok", "none")
    if flaky:
        result.add("Flaky cases", "warn", "; ".join(flaky))

    # Cost, latency and guard: warnings.
    for name, get, fmt, limit in [
        (
            "Cost per conversation",
            lambda m: m["cost_usd"]["per_conversation"],
            lambda v: f"${v:.5f}",
            rules.max_cost_increase,
        ),
        (
            "Latency per user turn",
            lambda m: m["latency_ms"]["turn_avg"],
            lambda v: f"{v / 1000:.2f} s",
            rules.max_latency_increase,
        ),
    ]:
        change = _relative(get(base), get(cand))
        if change is None:
            continue
        status = "warn" if change >= limit else "ok"
        result.add(name, status, f"{change:+.0%} ({fmt(get(base))} -> {fmt(get(cand))})")
    blocked_base, blocked_cand = base["guard_blocked_submits"], cand["guard_blocked_submits"]
    if blocked_cand > blocked_base:
        result.add(
            "Submits blocked by the code guard",
            "warn",
            f"{blocked_base} -> {blocked_cand}: the model tried to submit without a "
            "valid confirmation (the guard stopped it)",
        )
    return result


def run_gate(baseline: Path, candidate: Path, rules: GateRules = GateRules()) -> GateResult:
    baseline, candidate = resolve_input(baseline), resolve_input(candidate)
    conn = store.connect(":memory:")
    base_id, cand_id = _load(conn, baseline), _load(conn, candidate)
    if base_id == cand_id:
        raise GateInputError(f"baseline and candidate are the same run: {baseline.name}")
    return evaluate(metrics.compare_runs(conn, base_id, cand_id), rules)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

_LABEL = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
_ICON = {"ok": ":white_check_mark:", "warn": ":warning:", "fail": ":x:"}


def format_text(result: GateResult) -> str:
    lines = [f"Regression gate: {result.candidate} vs baseline {result.baseline}", ""]
    lines += [f"  [{_LABEL[c.status]}] {c.name}: {c.detail}" for c in result.checks]
    verdict = {"ok": "PASSED", "warn": "PASSED with warnings", "fail": "FAILED"}[result.status]
    return "\n".join([*lines, "", f"Gate {verdict}."])


def format_markdown(result: GateResult) -> str:
    verdict = {"ok": "passed", "warn": "passed with warnings", "fail": "failed"}[result.status]
    rows = [f"| {_ICON[c.status]} | {c.name} | {c.detail} |" for c in result.checks]
    return "\n".join(
        [
            f"### Eval regression gate {verdict}",
            "",
            f"`{result.candidate}` vs baseline `{result.baseline}`",
            "",
            "| | Check | Result |",
            "|---|---|---|",
            *rows,
            "",
        ]
    )


def write_step_summary(result: GateResult) -> None:
    """Show the report on the GitHub Actions run page when running there."""
    path = os.environ.get(STEP_SUMMARY_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as file:
            file.write(format_markdown(result))


def to_dict(result: GateResult) -> dict[str, Any]:
    return {**asdict(result), "status": result.status}
