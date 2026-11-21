"""Plain-text rendering of run metrics and comparisons for the CLI (ASCII only, so it
prints on any console)."""

from __future__ import annotations

from typing import Any

SIGNIFICANCE = 0.05


# --------------------------------------------------------------------------- #
# Value formatting
# --------------------------------------------------------------------------- #


def fmt_rate(value: dict[str, Any] | None) -> str:
    if not value or not value["total"]:
        return "n/a"
    return f"{value['rate'] * 100:.1f}% ({value['passed']}/{value['total']})"


def fmt_seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.2f} s"


def fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.5f}"


def fmt_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


# --------------------------------------------------------------------------- #
# Metric rows: (label, getter, formatter, higher_is_better)
# --------------------------------------------------------------------------- #


def _metric_rows(metrics: dict[str, Any]) -> list[tuple[str, Any, Any, bool | None]]:
    rows: list[tuple[str, Any, Any, bool | None]] = [
        ("Cases passed", lambda m: m["cases"], fmt_rate, True),
    ]
    for kind in sorted(metrics["checks"]):
        rows.append((f"  {kind} checks", lambda m, k=kind: m["checks"].get(k), fmt_rate, True))
    rows += [
        ("Wrong / premature submits", lambda m: m["wrong_submit"], fmt_rate, False),
        ("Missed submits", lambda m: m["missed_submit"], fmt_rate, False),
        ("Cost / conversation", lambda m: m["cost_usd"]["per_conversation"], fmt_usd, False),
        (
            "Input tokens / conversation",
            lambda m: m["tokens"]["input_per_conversation"],
            fmt_number,
            False,
        ),
        (
            "Output tokens / conversation",
            lambda m: m["tokens"]["output_per_conversation"],
            fmt_number,
            False,
        ),
        ("Prompt cache hit rate", lambda m: m["tokens"]["cache_hit_rate"], fmt_percent, True),
        ("Latency / user turn, avg", lambda m: m["latency_ms"]["turn_avg"], fmt_seconds, False),
        ("Latency / user turn, p50", lambda m: m["latency_ms"]["turn_p50"], fmt_seconds, False),
        ("Latency / user turn, p95", lambda m: m["latency_ms"]["turn_p95"], fmt_seconds, False),
        ("LLM call latency, p50", lambda m: m["latency_ms"]["llm_call_p50"], fmt_seconds, False),
        ("LLM call latency, p95", lambda m: m["latency_ms"]["llm_call_p95"], fmt_seconds, False),
    ]
    return rows


def _scalar(value: Any) -> float | None:
    if isinstance(value, dict):
        return value["rate"] if value.get("total") else None
    return value


def _delta(base: Any, cand: Any, fmt: Any, higher_is_better: bool | None) -> str:
    b, c = _scalar(base), _scalar(cand)
    if b is None or c is None:
        return ""
    if fmt in (fmt_rate, fmt_percent):
        text = f"{(c - b) * 100:+.1f} pts"
    elif b == 0:
        text = "" if c == 0 else "new"
    else:
        text = f"{(c - b) / b * 100:+.0f}%"
    worse = higher_is_better is not None and (c < b if higher_is_better else c > b)
    # Any drop in a pass rate is flagged; for cost, latency, tokens and cache hits
    # small changes are normal run-to-run noise, so only a 10% change is flagged.
    if worse and fmt is not fmt_rate and b and abs(c - b) / b < 0.10:
        worse = False
    return f"{text}  !" if worse else text


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def format_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No runs yet. Load some with: evalplat ingest <folder>"
    rows = [
        [
            str(r["id"]),
            r["name"],
            r["model"] + (f" (thinking {r['thinking']})" if r["thinking"] else ""),
            f"{r['unique_cases']} x {r['repeats']}",
            fmt_rate(r["cases"]),
            fmt_usd(r["cost_per_conversation_usd"]),
            "yes" if r["traced"] else "no",
        ]
        for r in runs
    ]
    return table(["id", "name", "model", "cases", "passed", "cost/conv", "traces"], rows)


def _describe(run: dict[str, Any]) -> str:
    thinking = f", thinking {run['thinking']}" if run["thinking"] else ""
    return f"#{run['id']} {run['name']} ({run['model']}{thinking})"


def format_run(metrics: dict[str, Any]) -> str:
    rows = [[label, fmt(get(metrics))] for label, get, fmt, _ in _metric_rows(metrics)]
    low, high = metrics["cases_ci95"] or (None, None)
    lines = [
        f"Run {_describe(metrics['run'])}",
        f"{metrics['unique_cases']} cases x {metrics['run']['repeats']} repeats, "
        f"run at {metrics['run']['run_at']}",
        "",
        table(["metric", "value"], rows),
        "",
        table(
            ["category", "passed"],
            [[cat, fmt_rate(r)] for cat, r in metrics["by_category"].items()],
        ),
    ]
    if low is not None:
        lines += [
            "",
            f"95% confidence interval for the pass rate: {low * 100:.1f}% - {high * 100:.1f}%",
        ]
    if not metrics["run"]["traced"]:
        lines.append("No trace events for this run: latency percentiles are not available.")
    return "\n".join(lines)


def format_comparison(comparison: dict[str, Any]) -> str:
    base, cand = comparison["baseline"], comparison["candidate"]
    rows = [
        [label, fmt(get(base)), fmt(get(cand)), _delta(get(base), get(cand), fmt, better)]
        for label, get, fmt, better in _metric_rows(cand)
    ]
    categories = sorted(base["by_category"].keys() | cand["by_category"].keys())
    category_rows = [
        [
            cat,
            fmt_rate(base["by_category"].get(cat)),
            fmt_rate(cand["by_category"].get(cat)),
            _delta(base["by_category"].get(cat), cand["by_category"].get(cat), fmt_rate, True),
        ]
        for cat in categories
    ]
    lines = [
        f"Baseline:  {_describe(base['run'])}",
        f"Candidate: {_describe(cand['run'])}",
        "",
        table(["metric", "baseline", "candidate", "change"], rows),
        "",
        table(["category", "baseline", "candidate", "change"], category_rows),
        "",
        _verdict(comparison),
    ]

    # Changed cases, plus cases that stayed flaky (passed some repeats, failed others).
    shown = [
        c for c in comparison["cases"] if c["status"] != "unchanged" or c["flaky_in_candidate"]
    ]
    if shown:
        status = {c["case_id"]: c["status"] for c in shown}
        for case in shown:
            if case["status"] == "unchanged":
                status[case["case_id"]] = "flaky"
        lines += [
            "",
            table(
                ["case", "status", "baseline", "candidate"],
                [
                    [
                        c["case_id"],
                        status[c["case_id"]],
                        fmt_rate(c["baseline"]),
                        fmt_rate(c["candidate"]),
                    ]
                    for c in shown
                ],
            ),
        ]
        failing = [c for c in shown if c["failures"]]
        if failing:
            lines += ["", "Failed checks:"]
        for case in failing:
            lines.append(f"  {case['case_id']}")
            lines += [f"    - {failure}" for failure in case["failures"][:3]]
    if not base["run"]["traced"] or not cand["run"]["traced"]:
        lines += ["", "Note: latency percentiles need trace events; n/a where a run has none."]
    return "\n".join(lines)


def _verdict(comparison: dict[str, Any]) -> str:
    common = comparison["common_cases"]
    if not common["count"]:
        return "The two runs have no cases in common."
    b, c = common["baseline"], common["candidate"]
    change = (c["rate"] - b["rate"]) * 100
    text = (
        f"Pass rate on the {common['count']} cases both runs share: "
        f"{fmt_rate(b)} -> {fmt_rate(c)} ({change:+.1f} pts)."
    )
    if change == 0:
        return text
    p = common["p_better"] if change > 0 else common["p_worse"]
    direction = "improvement" if change > 0 else "drop"
    if p < SIGNIFICANCE:
        return (
            f"{text}\nOne-sided Fisher exact p = {p:.3f}: the {direction} is unlikely to be noise."
        )
    return (
        f"{text}\nOne-sided Fisher exact p = {p:.3f}: not significant at {SIGNIFICANCE}; "
        "with this many repeats, a change this size can be noise."
    )
