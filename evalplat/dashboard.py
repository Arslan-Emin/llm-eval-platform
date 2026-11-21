"""Streamlit dashboard: runs, one run, run comparison, traces, one conversation.

Run with `streamlit run streamlit_app.py`. All numbers come from the same
functions as the CLI (`metrics`, `traces`, `report`), so both always agree.

The selected run / conversation lives in the URL (`?run=...`), so every view can be
shared as a link. Each rerun opens its own SQLite connection: Streamlit
runs every session in its own thread, and a connection cannot cross threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evalplat import charts, metrics, report, store, traces
from evalplat.ingest import ingest_paths

SEED_DIR_ENV = "EVALPLAT_SEED_DIR"
PAGES: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


@st.cache_resource
def _prepare_database(path: str, seed_dir: str | None) -> str:
    """Create the database once per process; load the seed folder into an empty one
    (the public demo starts from an empty disk and seeds itself from data/sample)."""
    conn = store.connect(path)
    try:
        if seed_dir and not conn.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
            ingest_paths(conn, [Path(seed_dir)])
    finally:
        conn.close()
    return path


def get_conn() -> sqlite3.Connection:
    path = _prepare_database(str(store.resolve_db_path()), os.environ.get(SEED_DIR_ENV))
    return store.connect(path)


def theme_mode() -> str | None:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None)


def run_label(run: dict[str, Any]) -> str:
    thinking = f", thinking {run['thinking']}" if run.get("thinking") else ""
    return f"{run['name']} ({run['model']}{thinking})"


def short_name(name: str) -> str:
    """'20261119-225827_deepseek-flash_baseline' -> '11-19 22:58' (chart axis label)."""
    stamp = name.partition("_")[0]
    if len(stamp) == 15 and stamp[8] == "-":
        label = f"{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
        return label
    return name


def url_selectbox(
    label: str,
    options: list[str],
    param: str,
    *,
    default: int = 0,
    format_func: Any = str,
    container: Any = st,
) -> str:
    """A selectbox whose value is kept in the URL query parameter `param`.

    Streamlit's own query-param binding writes the *formatted* label to the URL, so
    links would carry long display text; this keeps the raw value instead. The
    widget has no key, so a new URL value (from a link) re-initialises it."""
    current = st.query_params.get(param)
    index = options.index(current) if current in options else default
    value = container.selectbox(label, options, index=index, format_func=format_func)
    if st.query_params.get(param) != value:
        st.query_params[param] = value
    return value


def no_runs_message() -> None:
    st.info(
        "No eval runs in the database yet. Load some with "
        "`evalplat ingest <results folder> <traces folder>`, then reload this page."
    )


def rate_value(rate: dict[str, Any] | None) -> float | None:
    return rate["rate"] if rate and rate["total"] else None


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_runs() -> None:
    st.title("Eval runs")
    conn = get_conn()
    runs = metrics.list_runs(conn)
    if not runs:
        no_runs_message()
        return
    details = [metrics.run_metrics(conn, r["id"]) for r in runs]

    latest, previous = details[-1], (details[-2] if len(details) > 1 else None)
    st.caption(f"Latest run: {run_label(latest['run'])}, {latest['run']['run_at']}")
    _headline(latest, previous)

    trend = pd.DataFrame(
        [
            {
                "run": short_name(m["run"]["name"]),
                "name": m["run"]["name"],
                "pass_rate": rate_value(m["cases"]),
                "ci_low": (m["cases_ci95"] or (None, None))[0],
                "ci_high": (m["cases_ci95"] or (None, None))[1],
                "cost": m["cost_usd"]["per_conversation"],
                "p95": (m["latency_ms"]["turn_p95"] or 0) / 1000 or None,
            }
            for m in details
        ]
    )
    mode = theme_mode()
    left, middle, right = st.columns(3)
    left.altair_chart(
        charts.trend_chart(
            trend,
            value="pass_rate",
            title="Pass rate",
            axis_format=".0%",
            low="ci_low",
            high="ci_high",
            zero=False,
            mode=mode,
        )
    )
    middle.altair_chart(
        charts.trend_chart(
            trend, value="cost", title="Cost / conversation", axis_format="$.5f", mode=mode
        )
    )
    right.altair_chart(
        charts.trend_chart(
            trend.dropna(subset=["p95"]),
            value="p95",
            title="p95 latency / turn (s)",
            axis_format=".2f",
            mode=mode,
        )
    )
    st.caption(
        "The band around the pass rate is its 95% confidence interval. Runs recorded"
        " before tracing have no latency percentiles and are left out of the last chart."
    )

    st.subheader("All runs")
    table = pd.DataFrame(
        [
            {
                "name": r["name"],
                "model": r["model"],
                "thinking": r["thinking"],
                "cases": f"{r['unique_cases']} x {r['repeats']}",
                "passed": report.fmt_rate(r["cases"]),
                "cost / conv.": report.fmt_usd(r["cost_per_conversation_usd"]),
                "traces": "yes" if r["traced"] else "no",
                "run at (UTC)": r["run_at"],
            }
            for r in reversed(runs)
        ]
    )
    selection = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", key="runs_table"
    )
    st.caption("Select a run to open it.")
    if selection.selection.rows:
        st.switch_page(
            PAGES["run"], query_params={"run": table.iloc[selection.selection.rows[0]]["name"]}
        )


def _card(
    column: Any,
    label: str,
    value: str,
    change: float | None,
    unit: str,
    *,
    lower_is_better: bool = False,
    help: str | None = None,
) -> None:
    """A metric tile; an unchanged value gets a neutral delta instead of a colored arrow."""
    if change is None:
        column.metric(label, value, help=help, border=True)
    elif round(change, 1) == 0:
        column.metric(
            label, value, "no change", delta_color="off", delta_arrow="off", help=help, border=True
        )
    else:
        column.metric(
            label,
            value,
            f"{change:+.1f} {unit}",
            delta_color="inverse" if lower_is_better else "normal",
            help=help,
            border=True,
        )


def _headline(latest: dict[str, Any], previous: dict[str, Any] | None) -> None:
    def pts(key: str) -> float | None:
        a = rate_value(latest[key])
        b = rate_value(previous[key]) if previous else None
        return None if a is None or b is None else (a - b) * 100

    def pct(get: Any) -> float | None:
        a, b = get(latest), (get(previous) if previous else None)
        return None if a is None or not b else (a - b) / b * 100

    cols = st.columns(4)
    _card(
        cols[0],
        "Pass rate",
        report.fmt_percent(rate_value(latest["cases"])),
        pts("cases"),
        "pts",
        help=f"Conversations passing every check: {report.fmt_rate(latest['cases'])}",
    )
    _card(
        cols[1],
        "Wrong submits",
        report.fmt_percent(rate_value(latest["wrong_submit"])),
        pts("wrong_submit"),
        "pts",
        lower_is_better=True,
        help="Submitted without confirmation, after a prompt injection or with invalid data: "
        + report.fmt_rate(latest["wrong_submit"]),
    )
    cost = lambda m: m["cost_usd"]["per_conversation"]  # noqa: E731
    _card(
        cols[2],
        "Cost / conversation",
        report.fmt_usd(cost(latest)),
        pct(cost),
        "%",
        lower_is_better=True,
        help="Estimated from token counts and the provider's price table",
    )
    p95 = lambda m: m["latency_ms"]["turn_p95"]  # noqa: E731
    _card(
        cols[3],
        "p95 latency / turn",
        report.fmt_seconds(p95(latest)),
        pct(p95),
        "%",
        lower_is_better=True,
        help="95th percentile of the LLM wait per user message (from traces)",
    )
    if previous:
        st.caption(f"Changes are against the previous run, {previous['run']['name']}.")


def page_run() -> None:
    st.title("Run details")
    conn = get_conn()
    runs = metrics.list_runs(conn)
    if not runs:
        no_runs_message()
        return
    by_name = {r["name"]: r for r in runs}
    names = [r["name"] for r in reversed(runs)]
    name = url_selectbox("Run", names, "run", format_func=lambda n: run_label(by_name[n]))
    m = metrics.run_metrics(conn, by_name[name]["id"])

    cols = st.columns(4)
    for column, label, rate in [
        (cols[0], "Pass rate", m["cases"]),
        (cols[1], "Wrong submits", m["wrong_submit"]),
        (cols[2], "Missed submits", m["missed_submit"]),
    ]:
        _card(
            column,
            label,
            report.fmt_percent(rate_value(rate)),
            None,
            "",
            help=report.fmt_rate(rate),
        )
    _card(
        cols[3], "Cost / conversation", report.fmt_usd(m["cost_usd"]["per_conversation"]), None, ""
    )
    low, high = m["cases_ci95"] or (None, None)
    notes = [
        f"{m['unique_cases']} cases x {m['run']['repeats']} repeats, run at {m['run']['run_at']}."
    ]
    if low is not None:
        notes.append(f"95% confidence interval for the pass rate: {low:.1%} - {high:.1%}.")
    if not m["run"]["traced"]:
        notes.append("This run has no trace events: no latency percentiles or timelines.")
    st.caption(" ".join(notes))

    left, right = st.columns(2)
    with left:
        st.subheader("By category")
        st.dataframe(_rate_table(m["by_category"], "category"), hide_index=True)
    with right:
        st.subheader("By check type")
        st.dataframe(_rate_table(m["checks"], "check"), hide_index=True)
        st.subheader("Latency and tokens")
        st.dataframe(
            pd.DataFrame(
                [
                    [label, fmt(get(m))]
                    for label, get, fmt, _ in report.metric_rows(m)
                    if "atency" in label or "oken" in label or "cache" in label
                ],
                columns=["metric", "value"],
            ),
            hide_index=True,
        )

    cases = conn.execute(
        "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id, repeat_index",
        (m["run"]["id"],),
    ).fetchall()
    failed = [c for c in cases if not c["passed"]]
    st.subheader(f"Failed conversations ({len(failed)})")
    if not failed:
        st.success("Every conversation passed all of its checks.")
    for case in failed:
        failing = conn.execute(
            "SELECT description, turn FROM checks"
            " WHERE case_result_id = ? AND passed = 0 ORDER BY position",
            (case["id"],),
        ).fetchall()
        with st.expander(f"{case['case_id']}  (repeat {case['repeat_index'] + 1})"):
            for check in failing:
                where = (
                    "end of conversation" if check["turn"] is None else f"turn {check['turn'] + 1}"
                )
                st.markdown(f"- **{where}:** {check['description']}")
            if case["error"]:
                st.error(case["error"])
            _transcript(json.loads(case["transcript_json"] or "[]"))
            if case["conversation_id"]:
                st.page_link(
                    PAGES["conversation"],
                    label="Open trace",
                    query_params={"conversation": case["conversation_id"]},
                )

    st.subheader("All conversations")
    table = pd.DataFrame(
        [
            {
                "case": c["case_id"],
                "repeat": c["repeat_index"] + 1,
                "category": c["category"],
                "passed": bool(c["passed"]),
                "user turns": c["user_turns"],
                "LLM calls": c["llm_calls"],
                "cost": report.fmt_usd(c["cost_usd"]),
                "LLM wait": report.fmt_seconds(c["latency_ms"]),
                "conversation_id": c["conversation_id"],
            }
            for c in cases
        ]
    )
    selection = st.dataframe(
        table.drop(columns=["conversation_id"]),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cases_table",
    )
    if selection.selection.rows:
        row = selection.selection.rows[0]
        conversation_id = table.iloc[row]["conversation_id"]
        if conversation_id:
            st.switch_page(PAGES["conversation"], query_params={"conversation": conversation_id})
        st.markdown(f"**{table.iloc[row]['case']}**, repeat {table.iloc[row]['repeat']} (no trace)")
        _transcript(json.loads(cases[row]["transcript_json"] or "[]"))
    else:
        st.caption(
            "Select a conversation to see its trace (or its transcript, for runs without traces)."
        )


def _rate_table(rates: dict[str, dict[str, Any]], label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{label: key, "passed": report.fmt_rate(value)} for key, value in rates.items()]
    )


def page_compare() -> None:
    st.title("Compare runs")
    conn = get_conn()
    runs = metrics.list_runs(conn)
    if len(runs) < 2:
        st.info("Comparing needs at least two runs.")
        return
    by_name = {r["name"]: r for r in runs}
    names = [r["name"] for r in reversed(runs)]
    left, right = st.columns(2)
    baseline = url_selectbox(
        "Baseline",
        names,
        "baseline",
        default=1,
        format_func=lambda n: run_label(by_name[n]),
        container=left,
    )
    candidate = url_selectbox(
        "Candidate",
        names,
        "candidate",
        format_func=lambda n: run_label(by_name[n]),
        container=right,
    )
    if baseline == candidate:
        st.warning("Pick two different runs.")
        return
    comparison = metrics.compare_runs(conn, by_name[baseline]["id"], by_name[candidate]["id"])

    common = comparison["common_cases"]
    message = report.verdict(comparison)
    if not common["count"] or common["baseline"]["rate"] == common["candidate"]["rate"]:
        st.info(message)
    elif common["candidate"]["rate"] < common["baseline"]["rate"]:
        # A significant drop is an error; a drop that may be noise is still a warning.
        (st.error if common["p_worse"] < report.SIGNIFICANCE else st.warning)(message)
    else:
        # Green only when the improvement is unlikely to be noise.
        (st.success if common["p_better"] < report.SIGNIFICANCE else st.info)(message)

    base, cand = comparison["baseline"], comparison["candidate"]
    st.subheader("Metrics")
    st.dataframe(
        pd.DataFrame(
            [
                [
                    label,
                    fmt(get(base)),
                    fmt(get(cand)),
                    report.delta(get(base), get(cand), fmt, better),
                ]
                for label, get, fmt, better in report.metric_rows(cand)
            ],
            columns=["metric", "baseline", "candidate", "change"],
        ),
        hide_index=True,
    )
    st.caption(
        "`!` marks a change for the worse:"
        " any pass-rate drop, or 10% or more for cost, tokens and latency."
    )

    categories = sorted(base["by_category"].keys() | cand["by_category"].keys())
    st.subheader("By category")
    st.dataframe(
        pd.DataFrame(
            [
                [
                    cat,
                    report.fmt_rate(base["by_category"].get(cat)),
                    report.fmt_rate(cand["by_category"].get(cat)),
                    report.delta(
                        base["by_category"].get(cat),
                        cand["by_category"].get(cat),
                        report.fmt_rate,
                        True,
                    ),
                ]
                for cat in categories
            ],
            columns=["category", "baseline", "candidate", "change"],
        ),
        hide_index=True,
    )

    shown = [
        c for c in comparison["cases"] if c["status"] != "unchanged" or c["flaky_in_candidate"]
    ]
    st.subheader(f"Changed or flaky cases ({len(shown)})")
    if not shown:
        st.caption("Every case has the same pass rate in both runs.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "case": c["case_id"],
                    "category": c["category"],
                    "status": "flaky" if c["status"] == "unchanged" else c["status"],
                    "baseline": report.fmt_rate(c["baseline"]),
                    "candidate": report.fmt_rate(c["candidate"]),
                    "failed checks": "; ".join(c["failures"][:3]),
                }
                for c in shown
            ]
        ),
        hide_index=True,
    )


def page_traces() -> None:
    st.title("Traces")
    conn = get_conn()
    sources = traces.list_trace_sources(conn)
    if not sources:
        st.info("No trace events yet. Load JSONL traces with `evalplat ingest <traces folder>`.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "source": s["source"],
                    "eval run": s["run_name"] or "-",
                    "conversations": s["conversations"],
                    "events": s["events"],
                    "first event (UTC)": s["first_event"],
                    "last event (UTC)": s["last_event"],
                }
                for s in sources
            ]
        ),
        hide_index=True,
    )
    options = ["all"] + [s["source"] for s in sources]
    source = url_selectbox("Source", options, "source")
    conversations = traces.list_conversations(conn, source=None if source == "all" else source)
    table = pd.DataFrame(
        [
            {
                "started (UTC)": c["started_at"],
                "case": c["case_id"] or "-",
                "passed": None if c["passed"] is None else bool(c["passed"]),
                "turns": c["turns"],
                "LLM calls": c["llm_calls"],
                "failed tools": c["failed_tool_calls"],
                "cost": report.fmt_usd(c["cost_usd"]),
                "duration": report.fmt_seconds(c["duration_ms"]),
                "submitted": bool(c["submitted"]),
                "source": c["source"],
                "conversation_id": c["conversation_id"],
            }
            for c in conversations
        ]
    )
    st.caption(f"{len(table)} conversations. Select one to open its trace.")
    selection = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="conversations_table",
    )
    if selection.selection.rows:
        st.switch_page(
            PAGES["conversation"],
            query_params={
                "conversation": table.iloc[selection.selection.rows[0]]["conversation_id"]
            },
        )


def page_conversation() -> None:
    st.title("Conversation trace")
    conn = get_conn()
    known = {c["conversation_id"]: c for c in traces.list_conversations(conn)}
    if not known:
        st.info("No traced conversations yet.")
        return

    def label(conversation_id: str) -> str:
        c = known[conversation_id]
        case = f"{c['case_id']} #{c['repeat_index'] + 1}" if c["case_id"] else "app session"
        return f"{case} - {c['started_at'][:19]} - {conversation_id[:8]}"

    conversation_id = url_selectbox("Conversation", list(known), "conversation", format_func=label)
    trace = traces.conversation_trace(conn, conversation_id)
    case, totals = trace["case"], trace["totals"]

    if case:
        status = "passed" if case["passed"] else "failed"
        st.markdown(
            f"Eval case **{case['case_id']}** (repeat {case['repeat_index'] + 1}) in run "
            f"`{case['run_name']}`: **{status}**"
        )
    duration = sum((t["summary"] or {}).get("duration_ms") or 0 for t in trace["turns"])
    cols = st.columns(4)
    _card(cols[0], "User turns", str(len([t for t in trace["turns"] if t["summary"]])), None, "")
    _card(
        cols[1],
        "LLM calls",
        str(totals["llm_calls"]),
        None,
        "",
        help=f"Tokens in / out: {totals['input_tokens']:,} / {totals['output_tokens']:,}",
    )
    _card(cols[2], "Cost", report.fmt_usd(totals["cost_usd"]), None, "")
    _card(cols[3], "Time in agent turns", report.fmt_seconds(duration), None, "")

    spans = sorted(
        [span for turn in trace["turns"] for span in turn["spans"]],
        key=lambda s: (s["start_ms"], s["type"] != "agent_turn"),
    )
    if spans:
        st.subheader("Timeline")
        st.altair_chart(charts.waterfall_chart(spans, theme_mode()))
        st.caption(
            "Each bar is one event; an agent turn spans the LLM and tool calls it made."
            " Events shorter than the thinnest visible bar (most tool calls) are drawn at"
            " that minimum width; hover for the real duration."
        )

    if not totals["has_content"]:
        st.info(
            "Message content is not in this trace (the app logs it only with "
            "TRACE_INCLUDE_CONTENT=true); metrics and timing are still complete."
        )
    st.subheader("Turns")
    for turn in trace["turns"]:
        summary = turn["summary"] or {}
        user_text = summary.get("user_text")
        if (
            user_text is None
            and turn["turn_index"] is not None
            and turn["turn_index"] < len(trace["transcript"])
        ):
            user_text = trace["transcript"][turn["turn_index"]]["user"]
        if user_text:
            st.chat_message("user").write(user_text)
        reply = summary.get("reply")
        if reply:
            st.chat_message("assistant").write(reply)
        with st.expander(
            f"Turn {(turn['turn_index'] or 0) + 1}: {summary.get('llm_calls', '?')} LLM calls, "
            f"{report.fmt_seconds(summary.get('duration_ms'))}, "
            f"fields {summary.get('fields_filled', '?')}/{summary.get('fields_total', '?')}"
            + (", submitted" if summary.get("submitted") else "")
        ):
            for span in turn["spans"]:
                if span["type"] == "agent_turn":
                    continue
                event = span["event"]
                st.markdown(f"**{span['label']}** ({span['duration_ms']:.0f} ms, ok={span['ok']})")
                details = {
                    k: event[k]
                    for k in (
                        "arguments",
                        "result",
                        "tool_call_arguments",
                        "output_text",
                        "error",
                        "invalid_fields",
                    )
                    if event.get(k) not in (None, "", [])
                }
                if details:
                    st.json(details, expanded=False)

    if trace["checks"]:
        st.subheader("Eval checks")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "passed": bool(c["passed"]),
                        "type": c["kind"],
                        "when": "end" if c["turn"] is None else f"turn {c['turn'] + 1}",
                        "check": c["description"],
                    }
                    for c in trace["checks"]
                ]
            ),
            hide_index=True,
        )
    if trace["final_values"]:
        st.subheader("Final form values")
        st.json(trace["final_values"])


def _transcript(transcript: list[dict[str, Any]]) -> None:
    for turn in transcript:
        st.chat_message("user").write(turn["user"])
        st.chat_message("assistant").write(turn["agent"])
        tools = ", ".join(
            f"{c['name']}{'' if c.get('ok') else ' (rejected)'}" for c in turn["tool_calls"]
        )
        if tools:
            st.caption(f"Tools: {tools}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="LLM Eval Platform", page_icon=":material/monitoring:", layout="wide"
    )
    PAGES.update(
        runs=st.Page(
            page_runs, title="Runs", icon=":material/monitoring:", url_path="runs", default=True
        ),
        run=st.Page(page_run, title="Run details", icon=":material/fact_check:", url_path="run"),
        compare=st.Page(
            page_compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"
        ),
        traces=st.Page(page_traces, title="Traces", icon=":material/timeline:", url_path="traces"),
        conversation=st.Page(
            page_conversation,
            title="Conversation",
            icon=":material/forum:",
            url_path="conversation",
        ),
    )
    st.navigation(list(PAGES.values())).run()
