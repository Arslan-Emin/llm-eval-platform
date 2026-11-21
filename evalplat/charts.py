"""Altair charts for the dashboard.

Colors: span types use the first three slots of a categorical palette that was
checked for colour-vision deficiency (all pairs separable in light and dark mode).
The order is fixed, so a span type keeps its color on every page. Each mode has
its own steps (dark steps are lighter, for contrast on a dark surface).
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

SERIES_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
SPAN_TYPES = ["agent_turn", "llm_call", "tool_call"]
SPAN_NAMES = {"agent_turn": "Agent turn", "llm_call": "LLM call", "tool_call": "Tool call"}


def _mode(mode: str | None) -> str:
    return "dark" if mode == "dark" else "light"


def trend_chart(
    data: pd.DataFrame,
    *,
    value: str,
    title: str,
    axis_format: str,
    mode: str | None = None,
    low: str | None = None,
    high: str | None = None,
    zero: bool = True,
) -> alt.Chart:
    """One metric across runs: a 2px line with point markers, optionally with an
    interval band (e.g. the 95% confidence interval of a pass rate).

    The y axis starts at zero unless `zero=False`: a cost moving from $0.00065 to
    $0.00068 is a 5% change and should not look like a cliff."""
    color = SERIES_COLORS[_mode(mode)][0]
    order = list(data["run"])
    x = alt.X(
        "run:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=80, labelFlush=True)
    )
    y_scale = alt.Scale(zero=zero, nice=True)
    tooltip = [
        alt.Tooltip("name:N", title="Run"),
        alt.Tooltip(f"{value}:Q", title=title, format=axis_format),
    ]
    if low and high:
        tooltip += [
            alt.Tooltip(f"{low}:Q", title="95% CI low", format=axis_format),
            alt.Tooltip(f"{high}:Q", title="95% CI high", format=axis_format),
        ]
    layers = []
    if low and high:
        layers.append(
            alt.Chart(data)
            .mark_area(opacity=0.15, color=color)
            .encode(x=x, y=alt.Y(f"{low}:Q", scale=y_scale), y2=f"{high}:Q")
        )
    base = alt.Chart(data).encode(
        x=x,
        y=alt.Y(
            f"{value}:Q", title=None, scale=y_scale, axis=alt.Axis(format=axis_format, tickCount=4)
        ),
        tooltip=tooltip,
    )
    layers.append(base.mark_line(strokeWidth=2, color=color))
    # Hit target bigger than the marker, so hovering anywhere near a point works.
    layers.append(base.mark_point(size=80, filled=True, color=color, opacity=1))
    layers.append(base.mark_point(size=600, opacity=0))
    return alt.layer(*layers).properties(title=title, height=200)


def waterfall_chart(spans: list[dict[str, Any]], mode: str | None = None) -> alt.Chart:
    """Where a conversation spent its time: one row per event, bars on a shared ms axis.

    Tool calls often take well under a millisecond, which would be an invisible bar
    next to second-long LLM calls; such events are drawn at a minimum visible width
    (the tooltip shows the real duration)."""
    total = max((s["end_ms"] for s in spans), default=0.0)
    min_width = total * 0.006
    rows = []
    for index, span in enumerate(spans):
        event = span["event"]
        rows.append(
            {
                "order": index,
                "row": f"{index + 1}. T{event.get('turn_index', '?')} {span['label']}",
                "type": SPAN_NAMES[span["type"]],
                "start_ms": span["start_ms"],
                "end_ms": span["end_ms"],
                "draw_end_ms": max(span["end_ms"], span["start_ms"] + min_width),
                "duration_ms": span["duration_ms"],
                "ok": "yes" if span["ok"] else "no",
                "tokens": (event.get("input_tokens") or 0) + (event.get("output_tokens") or 0)
                if span["type"] == "llm_call"
                else None,
            }
        )
    data = pd.DataFrame(rows)
    colors = SERIES_COLORS[_mode(mode)]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadius=3, height=12)
        .encode(
            x=alt.X("start_ms:Q", title="ms since the conversation started"),
            x2="draw_end_ms:Q",
            y=alt.Y(
                "row:N",
                sort=list(data["row"]),
                title=None,
                # Vega hides labels it thinks overlap; every row needs its name.
                axis=alt.Axis(labelLimit=260, labelOverlap=False),
            ),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=[SPAN_NAMES[t] for t in SPAN_TYPES], range=colors),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("row:N", title="Event"),
                alt.Tooltip("duration_ms:Q", title="Duration (ms)", format=",.1f"),
                alt.Tooltip("start_ms:Q", title="Start (ms)", format=",.0f"),
                alt.Tooltip("tokens:Q", title="Tokens", format=","),
                alt.Tooltip("ok:N", title="OK"),
            ],
        )
        .properties(height=alt.Step(24))
    )
