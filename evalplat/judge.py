"""LLM-as-judge: score individual agent replies against rubrics.

Keyword checks answer "does the reply contain *24 saat*?"; a judge can answer "is
everything this reply says about the clinic supported by the FAQ passages it was
given?". The judge sees the same data the agent had in that turn, taken from the
trace: the user message, the tool calls and their results (including retrieved
passages) and the saved form values. Runs whose traces have no message content
(the app's own traces) cannot be judged.

Verdicts are stored per (prompt hash, judge model): re-running `evalplat judge`
only pays for replies that were not judged yet with exactly this prompt.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from evalplat import pricing
from evalplat.rubrics import RUBRICS, SYSTEM_PROMPT, JudgeItem, Rubric, user_prompt
from evalplat.store import transaction

DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Rough characters-per-token for the dry-run estimate (Turkish and JSON tokenize
# worse than English prose, so this errs on the expensive side).
CHARS_PER_TOKEN = 3.0
EXPECTED_OUTPUT_TOKENS = 60


# --------------------------------------------------------------------------- #
# Judge client
# --------------------------------------------------------------------------- #


@dataclass
class JudgeResponse:
    text: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class JudgeClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> JudgeResponse: ...


class DeepSeekJudge:
    """DeepSeek through its OpenAI-compatible Chat Completions API, in JSON mode.

    Thinking is explicitly disabled: it is on by default for DeepSeek models and
    would multiply output tokens for a one-sentence verdict.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        base_url: str = DEEPSEEK_BASE_URL,
        max_tokens: int = 300,
        timeout_s: float = 60.0,
    ) -> None:
        from openai import OpenAI  # optional dependency: pip install "evalplat[judge]"

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=2)

    def complete(self, system: str, user: str) -> JudgeResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        usage = response.usage
        details = usage.prompt_tokens_details if usage else None
        return JudgeResponse(
            text=response.choices[0].message.content or "",
            model=response.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            cached_input_tokens=(details.cached_tokens or 0) if details else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class JudgeError(RuntimeError):
    pass


def parse_verdict(text: str) -> tuple[str, str]:
    """('pass' | 'fail', reason) from the judge's JSON, or JudgeError."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise JudgeError(f"not JSON: {text[:80]!r}") from None
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in {"pass", "fail"}:
        raise JudgeError(f"no pass/fail verdict in {text[:80]!r}")
    return verdict, str(data.get("reason", "")).strip()


def ask(judge: JudgeClient, user: str, attempts: int = 2) -> tuple[str, str, JudgeResponse]:
    """Ask once more when the answer is unusable: in JSON mode DeepSeek can return
    blank content (documented) and a model can still drift from the format."""
    error: JudgeError | None = None
    total = JudgeResponse(text="", model=judge.model)
    for _ in range(attempts):
        response = judge.complete(SYSTEM_PROMPT, user)
        # Every attempt is billed, so token counts add up across retries.
        total.model = response.model
        total.input_tokens += response.input_tokens
        total.cached_input_tokens += response.cached_input_tokens
        total.output_tokens += response.output_tokens
        total.latency_ms += response.latency_ms
        try:
            verdict, reason = parse_verdict(response.text)
        except JudgeError as problem:
            error = problem
            continue
        return verdict, reason, total
    raise JudgeError(str(error))


# --------------------------------------------------------------------------- #
# Items: one per agent reply, rebuilt from the trace
# --------------------------------------------------------------------------- #

# Fields of a tool result that are shown to the judge separately or are noise.
_HIDDEN_RESULT_KEYS = {"current_values", "results"}


def items_for_conversation(events: list[dict[str, Any]]) -> list[JudgeItem]:
    """Turn a conversation's events (in order) into one item per agent turn."""
    items: list[JudgeItem] = []
    saved: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    for event in events:
        if event["type"] == "tool_call":
            result = event.get("result") or {}
            if isinstance(result.get("current_values"), dict):
                saved = dict(result["current_values"])
            if event.get("name") == "search_knowledge_base":
                passages += [
                    {"title": r.get("title"), "text": r.get("text")}
                    for r in result.get("results") or []
                ]
            calls.append(
                {
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                    "ok": event.get("ok"),
                    "result": {k: v for k, v in result.items() if k not in _HIDDEN_RESULT_KEYS},
                }
            )
        elif event["type"] == "agent_turn":
            if "user_text" in event and "reply" in event:
                items.append(
                    {
                        "conversation_id": event.get("conversation_id"),
                        "turn_index": event.get("turn_index"),
                        "user_text": event["user_text"],
                        "reply": event["reply"] or "",
                        "tool_calls": calls,
                        "passages": passages,
                        "saved_values": {k: v for k, v in saved.items() if v is not None},
                        "fields_filled": event.get("fields_filled"),
                        "fields_total": event.get("fields_total"),
                        "submitted": bool(event.get("submitted")),
                    }
                )
            calls, passages = [], []
    return items


def run_items(conn: sqlite3.Connection, run_id: int) -> list[JudgeItem]:
    cases = conn.execute(
        """
        SELECT id, case_id, repeat_index, conversation_id FROM case_results
        WHERE run_id = ? AND conversation_id IS NOT NULL ORDER BY case_id, repeat_index
        """,
        (run_id,),
    ).fetchall()
    items = []
    for case in cases:
        rows = conn.execute(
            "SELECT data_json FROM events WHERE conversation_id = ? ORDER BY timestamp, rowid",
            (case["conversation_id"],),
        ).fetchall()
        for item in items_for_conversation([json.loads(r["data_json"]) for r in rows]):
            item.update(
                case_result_id=case["id"],
                case_id=case["case_id"],
                repeat_index=case["repeat_index"],
            )
            items.append(item)
    return items


# --------------------------------------------------------------------------- #
# Running the judge
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    rubric: Rubric
    item: JudgeItem
    prompt: str
    prompt_hash: str


@dataclass
class JudgeReport:
    planned: int = 0  # rubric checks that apply to this run's replies
    already_judged: int = 0  # have a stored verdict for this exact prompt and model
    reused: int = 0  # identical question judged before: verdict copied, nothing sent
    questions: int = 0  # distinct prompts to send (several replies can share one)
    judged: int = 0  # questions answered by the judge
    stale: int = 0  # stored verdicts whose question no longer applies (removed)
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_input_tokens: int = 0


def plan_tasks(items: Iterable[JudgeItem], rubrics: Iterable[Rubric]) -> list[Task]:
    tasks = []
    for item in items:
        for rubric in rubrics:
            if rubric.applies(item):
                prompt = user_prompt(rubric, item)
                digest = hashlib.sha256(f"{SYSTEM_PROMPT}\n\n{prompt}".encode()).hexdigest()
                tasks.append(Task(rubric, item, prompt, digest))
    return tasks


def estimate(prompts: list[str], model: str) -> tuple[int, float]:
    """Input tokens and an upper-bound cost (no prompt cache) for the given prompts."""
    tokens = sum(int(len(SYSTEM_PROMPT + prompt) / CHARS_PER_TOKEN) for prompt in prompts)
    cost = pricing.cost_usd(model, tokens, 0, EXPECTED_OUTPUT_TOKENS * len(prompts))
    return tokens, cost or 0.0


def run_judge(
    conn: sqlite3.Connection,
    run_id: int,
    judge: JudgeClient | None,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    rubric_names: Iterable[str] | None = None,
    limit: int | None = None,
    workers: int = 4,
    dry_run: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> JudgeReport:
    """Judge the replies of one run. With `dry_run` nothing is sent or stored: the
    report only counts what would be judged and estimates the cost.

    Every applicable (reply, rubric) pair gets its own verdict row. Identical
    questions (same prompt, e.g. a reply repeated word for word across repeats) are
    sent once; the other rows copy that verdict and are marked as cached."""
    rubrics = [RUBRICS[name] for name in (rubric_names or RUBRICS)]
    tasks = plan_tasks(run_items(conn, run_id), rubrics)
    report = JudgeReport(planned=len(tasks))

    rows = conn.execute("SELECT * FROM judgments WHERE judge_model = ?", (model,)).fetchall()
    stored = {(r["case_result_id"], r["turn_index"], r["rubric"], r["prompt_hash"]) for r in rows}
    known = {r["prompt_hash"]: (r["verdict"], r["reason"]) for r in rows}
    todo = [t for t in tasks if _key(t) not in stored]
    report.already_judged = len(tasks) - len(todo)

    # Verdicts for questions that are no longer asked (a rubric's rule or text
    # changed) are stale; still valid ones get the current rubric version.
    current = {_key(t) for t in tasks}
    run_rows = conn.execute(
        """
        SELECT j.id, j.case_result_id, j.turn_index, j.rubric, j.prompt_hash FROM judgments j
        JOIN case_results c ON c.id = j.case_result_id WHERE c.run_id = ? AND j.judge_model = ?
        """,
        (run_id, model),
    ).fetchall()
    stale = [
        r["id"]
        for r in run_rows
        if r["rubric"] in {rubric.name for rubric in rubrics}
        and (r["case_result_id"], r["turn_index"], r["rubric"], r["prompt_hash"]) not in current
    ]
    report.stale = len(stale)
    if not dry_run:
        with transaction(conn):
            conn.executemany("DELETE FROM judgments WHERE id = ?", [(i,) for i in stale])
            for rubric in rubrics:
                conn.execute(
                    """
                    UPDATE judgments SET rubric_version = ? WHERE rubric = ? AND judge_model = ?
                    AND case_result_id IN (SELECT id FROM case_results WHERE run_id = ?)
                    """,
                    (rubric.version, rubric.name, model, run_id),
                )

    groups: dict[str, list[Task]] = defaultdict(list)
    for task in todo:
        groups[task.prompt_hash].append(task)
    reusable = [digest for digest in groups if digest in known]
    report.reused = sum(len(groups[digest]) for digest in reusable)
    to_send = [digest for digest in groups if digest not in known]
    if limit is not None:
        to_send = to_send[:limit]
    report.questions = len(to_send)
    report.estimated_input_tokens, report.estimated_cost_usd = estimate(
        [groups[digest][0].prompt for digest in to_send], model
    )
    if dry_run:
        return report
    for digest in reusable:
        verdict, reason = known[digest]
        for task in groups[digest]:
            _store(conn, task, model, verdict, reason, None, None)
    if not to_send:
        return report
    if judge is None:
        raise ValueError("a judge client is needed unless dry_run is set")

    # API calls in parallel; the SQLite writes stay on this thread.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask, judge, groups[d][0].prompt): d for d in to_send}
        for done, future in enumerate(as_completed(futures), start=1):
            first, *copies = groups[futures[future]]
            try:
                verdict, reason, response = future.result()
            except Exception as error:  # noqa: BLE001 - one bad call must not stop the run
                report.errors.append(
                    f"{first.item['case_id']} turn {first.item['turn_index']} "
                    f"{first.rubric.name}: {error}"
                )
            else:
                cost = pricing.cost_usd(
                    response.model,
                    response.input_tokens,
                    response.cached_input_tokens,
                    response.output_tokens,
                )
                _store(conn, first, response.model, verdict, reason, response, cost)
                for task in copies:
                    _store(conn, task, response.model, verdict, reason, None, None)
                report.judged += 1
                report.reused += len(copies)
                report.cost_usd += cost or 0.0
            if progress:
                progress(done, len(to_send))
    return report


def _key(task: Task) -> tuple[int, int, str, str]:
    item = task.item
    return (item["case_result_id"], item["turn_index"], task.rubric.name, task.prompt_hash)


def _store(
    conn: sqlite3.Connection,
    task: Task,
    model: str,
    verdict: str,
    reason: str,
    response: JudgeResponse | None,
    cost: float | None,
) -> None:
    """One verdict row; without `response` it copies an earlier identical question."""
    with transaction(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO judgments (
                case_result_id, conversation_id, turn_index, rubric, rubric_version,
                prompt_hash, judge_model, verdict, reason, input_tokens,
                cached_input_tokens, output_tokens, cost_usd, latency_ms, cached, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.item["case_result_id"],
                task.item["conversation_id"],
                task.item["turn_index"],
                task.rubric.name,
                task.rubric.version,
                task.prompt_hash,
                model,
                verdict,
                reason,
                response.input_tokens if response else 0,
                response.cached_input_tokens if response else 0,
                response.output_tokens if response else 0,
                cost if response else 0.0,
                round(response.latency_ms, 1) if response else None,
                0 if response else 1,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )


# --------------------------------------------------------------------------- #
# Reading verdicts back
# --------------------------------------------------------------------------- #


def run_judgments(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT j.*, c.case_id, c.repeat_index, c.category FROM judgments j
        JOIN case_results c ON c.id = j.case_result_id
        WHERE c.run_id = ? ORDER BY c.case_id, c.repeat_index, j.turn_index, j.rubric
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def judge_summary(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Pass rate per rubric, failing replies, and where the judge and the keyword
    RAG checks disagree about the same turn."""
    judgments = run_judgments(conn, run_id)
    per_rubric: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for j in judgments:
        per_rubric[j["rubric"]][0] += j["verdict"] == "pass"
        per_rubric[j["rubric"]][1] += 1

    # Keyword RAG checks per (case result, turn): passed only if all of them passed.
    keyword: dict[tuple[int, int], bool] = {}
    for row in conn.execute(
        """
        SELECT ch.case_result_id, ch.turn, min(ch.passed) AS passed FROM checks ch
        JOIN case_results c ON c.id = ch.case_result_id
        WHERE c.run_id = ? AND ch.kind = 'rag' AND ch.turn IS NOT NULL
        GROUP BY ch.case_result_id, ch.turn
        """,
        (run_id,),
    ):
        keyword[(row["case_result_id"], row["turn"])] = bool(row["passed"])
    compared = [
        (j, keyword[(j["case_result_id"], j["turn_index"])])
        for j in judgments
        if j["rubric"] == "grounded_answer" and (j["case_result_id"], j["turn_index"]) in keyword
    ]
    return {
        "judged": len(judgments),
        "judge_models": sorted({j["judge_model"] for j in judgments}),
        "cost_usd": sum(j["cost_usd"] or 0 for j in judgments),
        "rubrics": {
            name: {"passed": p, "total": t, "rate": p / t if t else None}
            for name, (p, t) in sorted(per_rubric.items())
        },
        "failures": [j for j in judgments if j["verdict"] == "fail"],
        "rag_keyword_vs_judge": {
            "compared": len(compared),
            "agree": sum((j["verdict"] == "pass") == passed for j, passed in compared),
            "disagreements": [
                {**j, "keyword_passed": passed}
                for j, passed in compared
                if (j["verdict"] == "pass") != passed
            ],
        },
    }


def export_judgments(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Verdicts of a run in the file format `evalplat ingest` reads back."""
    from evalplat.ingest import JUDGMENT_COLUMNS

    rows = conn.execute(
        f"""
        SELECT {", ".join("j." + c for c in JUDGMENT_COLUMNS)} FROM judgments j
        JOIN case_results c ON c.id = j.case_result_id
        WHERE c.run_id = ? ORDER BY j.conversation_id, j.turn_index, j.rubric
        """,
        (run_id,),
    ).fetchall()
    return {"judgments": [dict(row) for row in rows]}
