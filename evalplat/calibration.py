"""Calibrating the LLM judge against human labels.

A human answers the same yes/no question the judge got, without seeing the judge's
verdict. Comparing the two tells how far the judge can be trusted, and the cases
where they disagree show what to fix (the judge's prompt, or the rubric itself).

Labels have a third option, "not applicable": the human says the question should
not have been asked about this reply at all. That is not a judge mistake but a
rubric mistake (its `applies` rule in code), so it is counted separately and left
out of the judge-vs-human agreement.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from evalplat.judge import DEFAULT_JUDGE_MODEL, items_for_conversation
from evalplat.rubrics import RUBRICS, JudgeItem
from evalplat.store import transaction

LABELER_ENV = "EVALPLAT_LABELER"
DEFAULT_LABELER = "human-1"
LABELS = ("pass", "fail", "not_applicable")


def labeler_name() -> str:
    return os.environ.get(LABELER_ENV) or DEFAULT_LABELER


# --------------------------------------------------------------------------- #
# Choosing what to label
# --------------------------------------------------------------------------- #


def build_label_queue(
    conn: sqlite3.Connection,
    *,
    size: int = 60,
    seed: int = 7,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> int:
    """Pick the replies to label and store them in a fixed, shuffled order.

    Every distinct question the judge failed is included (that is where a judge is
    most often wrong, and they are rare), and the rest is filled with questions it
    passed, spread evenly over the rubrics. The order is shuffled so the labeler
    cannot guess the judge's verdict from an item's position. Same seed, same queue.
    """
    rows = conn.execute(
        """
        SELECT prompt_hash, rubric, verdict, min(conversation_id) AS conversation_id,
               min(turn_index) AS turn_index
        FROM judgments WHERE judge_model = ? GROUP BY prompt_hash ORDER BY prompt_hash
        """,
        (judge_model,),
    ).fetchall()
    rng = random.Random(seed)
    failed = [r for r in rows if r["verdict"] == "fail"]
    passed_by_rubric: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        if row["verdict"] == "pass":
            passed_by_rubric[row["rubric"]].append(row)

    chosen = list(failed[:size])
    rubrics = sorted(passed_by_rubric)
    remaining = size - len(chosen)
    while remaining > 0 and any(passed_by_rubric[r] for r in rubrics):
        for rubric in rubrics:  # round robin keeps the rubrics balanced
            pool = passed_by_rubric[rubric]
            if remaining and pool:
                chosen.append(pool.pop(rng.randrange(len(pool))))
                remaining -= 1
    rng.shuffle(chosen)

    now = datetime.now(UTC).isoformat(timespec="seconds")
    with transaction(conn):
        conn.execute("DELETE FROM label_queue")
        conn.executemany(
            "INSERT INTO label_queue (prompt_hash, rubric, conversation_id, turn_index, position,"
            " added_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (r["prompt_hash"], r["rubric"], r["conversation_id"], r["turn_index"], i, now)
                for i, r in enumerate(chosen)
            ],
        )
    return len(chosen)


def queue(conn: sqlite3.Connection, labeler: str) -> list[dict[str, Any]]:
    """The label queue with this labeler's answers so far (label is None if open)."""
    rows = conn.execute(
        """
        SELECT q.*, l.label, l.note FROM label_queue q
        LEFT JOIN labels l ON l.prompt_hash = q.prompt_hash AND l.labeler = ?
        ORDER BY q.position
        """,
        (labeler,),
    ).fetchall()
    return [dict(row) for row in rows]


def save_label(
    conn: sqlite3.Connection,
    entry: dict[str, Any],
    label: str,
    *,
    labeler: str,
    note: str | None = None,
) -> None:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO labels (prompt_hash, rubric, conversation_id, turn_index, label, note,
                                labeler, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (prompt_hash, labeler) DO UPDATE SET
                label = excluded.label, note = excluded.note, created_at = excluded.created_at
            """,
            (
                entry["prompt_hash"],
                entry["rubric"],
                entry["conversation_id"],
                entry["turn_index"],
                label,
                note or None,
                labeler,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )


def delete_label(conn: sqlite3.Connection, prompt_hash: str, labeler: str) -> None:
    with transaction(conn):
        conn.execute(
            "DELETE FROM labels WHERE prompt_hash = ? AND labeler = ?", (prompt_hash, labeler)
        )


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement between two raters beyond what their label frequencies alone would
    give by chance: 1 = perfect, 0 = chance level. None when it is undefined (e.g.
    both raters always said the same single label)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
        for label in labels
    )
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(
    conn: sqlite3.Connection,
    *,
    labeler: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    rubric_names: list[str] | None = None,
) -> dict[str, Any]:
    """Judge verdicts vs human labels on the same questions."""
    rows = conn.execute(
        """
        SELECT l.prompt_hash, l.rubric, l.conversation_id, l.turn_index, l.label, l.note,
               (SELECT j.verdict FROM judgments j WHERE j.prompt_hash = l.prompt_hash
                AND j.judge_model = ? LIMIT 1) AS verdict,
               (SELECT j.reason FROM judgments j WHERE j.prompt_hash = l.prompt_hash
                AND j.judge_model = ? LIMIT 1) AS reason
        FROM labels l WHERE l.labeler = coalesce(?, l.labeler) ORDER BY l.rubric, l.id
        """,
        (judge_model, judge_model, labeler),
    ).fetchall()
    rows = [dict(r) for r in rows if r["verdict"] is not None]
    if rubric_names:
        rows = [r for r in rows if r["rubric"] in rubric_names]

    def stats(selected: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in selected if r["label"] != "not_applicable"]
        pairs = [(r["verdict"], r["label"]) for r in scored]
        confusion = {
            f"judge_{j}_human_{h}": sum(1 for a, b in pairs if (a, b) == (j, h))
            for j in ("pass", "fail")
            for h in ("pass", "fail")
        }
        judge_fails = confusion["judge_fail_human_fail"] + confusion["judge_fail_human_pass"]
        human_fails = confusion["judge_fail_human_fail"] + confusion["judge_pass_human_fail"]
        return {
            "labeled": len(selected),
            "not_applicable": sum(r["label"] == "not_applicable" for r in selected),
            "scored": len(scored),
            "agree": sum(a == b for a, b in pairs),
            "accuracy": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
            "kappa": cohen_kappa(pairs),
            "confusion": confusion,
            # Of the replies the judge failed, how many really were wrong.
            "fail_precision": confusion["judge_fail_human_fail"] / judge_fails
            if judge_fails
            else None,
            # Of the replies that really were wrong, how many the judge caught.
            "fail_recall": confusion["judge_fail_human_fail"] / human_fails
            if human_fails
            else None,
        }

    by_rubric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rubric[row["rubric"]].append(row)
    return {
        "overall": stats(rows),
        "rubrics": {name: stats(selected) for name, selected in sorted(by_rubric.items())},
        "disagreements": [
            r for r in rows if r["label"] != "not_applicable" and r["label"] != r["verdict"]
        ],
        "not_applicable": [r for r in rows if r["label"] == "not_applicable"],
    }


def rubric_title(name: str) -> str:
    return RUBRICS[name].title if name in RUBRICS else name


def item_with_history(
    conn: sqlite3.Connection, conversation_id: str, turn_index: int
) -> tuple[JudgeItem, list[JudgeItem]] | None:
    """The reply as the judge saw it, plus the turns before it (for the labeler)."""
    rows = conn.execute(
        "SELECT data_json FROM events WHERE conversation_id = ? ORDER BY timestamp, rowid",
        (conversation_id,),
    ).fetchall()
    items = items_for_conversation([json.loads(r["data_json"]) for r in rows])
    for index, item in enumerate(items):
        if item["turn_index"] == turn_index:
            return item, items[:index]
    return None


def export_labels(conn: sqlite3.Connection, labeler: str | None = None) -> dict[str, Any]:
    """Human labels in the file format `evalplat ingest` reads back."""
    from evalplat.ingest import LABEL_COLUMNS

    rows = conn.execute(
        f"SELECT {', '.join(LABEL_COLUMNS)} FROM labels WHERE labeler = coalesce(?, labeler)"
        " ORDER BY id",
        (labeler,),
    ).fetchall()
    return {"labels": [dict(row) for row in rows]}
