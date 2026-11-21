import json
import threading

import pytest

from evalplat import judge, store
from evalplat.cli import main
from evalplat.ingest import ingest_paths
from evalplat.rubrics import RUBRICS
from tests.factories import SAMPLE_DIR

PRO_RUN = 5  # 20261121-165646_deepseek-v4-pro in the sample data


class FakeJudge:
    """Fails a reply when it contains `fail_word`, passes it otherwise."""

    model = "deepseek-v4-pro"

    def __init__(self, fail_word="kaydediyorum", responses=None):
        self.fail_word = fail_word
        self.responses = list(responses or [])
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system, user):
        with self._lock:
            self.calls += 1
            if self.responses:
                return self.responses.pop(0)
        verdict = "fail" if self.fail_word in user else "pass"
        text = f'{{"verdict": "{verdict}", "reason": "fake"}}'
        return judge.JudgeResponse(text, self.model, 1000, 800, 20, 5.0)


@pytest.fixture
def conn():
    """Sample data without its stored verdicts, so the fake judge starts from zero."""
    conn = store.connect(":memory:")
    ingest_paths(conn, [SAMPLE_DIR])
    with store.transaction(conn):
        conn.execute("DELETE FROM judgments")
    return conn


# --------------------------------------------------------------------------- #
# Parsing and retries
# --------------------------------------------------------------------------- #


def test_parse_verdict():
    assert judge.parse_verdict('{"verdict": "PASS", "reason": "ok"}') == ("pass", "ok")
    for bad in ["   ", '{"verdict": "maybe"}', "[]", "pass"]:
        with pytest.raises(judge.JudgeError):
            judge.parse_verdict(bad)


def test_blank_json_mode_answer_is_retried_and_both_attempts_are_billed():
    blank = judge.JudgeResponse("   ", "deepseek-v4-pro", 100, 0, 1, 1.0)
    fake = FakeJudge(responses=[blank])
    verdict, _, response = judge.ask(fake, "user prompt")
    assert verdict == "pass" and fake.calls == 2
    assert response.input_tokens == 1100


def test_unusable_answers_raise_after_the_last_attempt():
    blank = judge.JudgeResponse("", "deepseek-v4-pro")
    with pytest.raises(judge.JudgeError):
        judge.ask(FakeJudge(responses=[blank, blank]), "user prompt")


# --------------------------------------------------------------------------- #
# Items rebuilt from traces
# --------------------------------------------------------------------------- #


def test_items_follow_saved_values_and_passages_turn_by_turn():
    events = [
        {
            "type": "tool_call",
            "name": "update_fields",
            "ok": True,
            "arguments": {"fields": {"phone": "0532"}},
            "result": {"ok": True, "current_values": {"phone": "0532", "full_name": None}},
        },
        {
            "type": "tool_call",
            "name": "search_knowledge_base",
            "ok": True,
            "arguments": {"query": "fiyat"},
            "result": {"ok": True, "results": [{"title": "Fiyat", "text": "500 TL", "score": 0.8}]},
        },
        {"type": "llm_call"},
        {
            "type": "agent_turn",
            "turn_index": 0,
            "user_text": "u0",
            "reply": "r0",
            "fields_filled": 1,
            "fields_total": 2,
            "submitted": False,
        },
        {
            "type": "agent_turn",
            "turn_index": 1,
            "user_text": "u1",
            "reply": "r1",
            "fields_filled": 1,
            "fields_total": 2,
            "submitted": False,
        },
    ]
    first, second = judge.items_for_conversation(events)

    assert first["saved_values"] == {"phone": "0532"}  # unset fields are left out
    assert first["passages"] == [{"title": "Fiyat", "text": "500 TL"}]
    assert "current_values" not in first["tool_calls"][0]["result"]
    assert second["tool_calls"] == [] and second["passages"] == []
    assert second["saved_values"] == {"phone": "0532"}  # carried over


def test_turns_without_logged_content_cannot_be_judged():
    events = [{"type": "agent_turn", "turn_index": 0, "fields_filled": 0, "fields_total": 4}]
    assert judge.items_for_conversation(events) == []


def test_rubrics_apply_to_the_right_turns(conn):
    tasks = judge.plan_tasks(judge.run_items(conn, PRO_RUN), RUBRICS.values())
    counts = {name: sum(t.rubric.name == name for t in tasks) for name in RUBRICS}
    # 198 replies; 24 used the knowledge base; 69 saved a value that left the form complete
    # (the turns where a fresh summary is due).
    assert counts == {
        "grounded_answer": 24,
        "claims_match_actions": 198,
        "confirmation_summary": 69,
    }


def test_changing_a_rubric_changes_the_prompt_hash(conn):
    items = judge.run_items(conn, PRO_RUN)[:1]
    rubric = RUBRICS["claims_match_actions"]
    changed = type(rubric)(**{**rubric.__dict__, "question": rubric.question + " Be strict."})
    [a], [b] = judge.plan_tasks(items, [rubric]), judge.plan_tasks(items, [changed])
    assert a.prompt_hash != b.prompt_hash


# --------------------------------------------------------------------------- #
# Running the judge
# --------------------------------------------------------------------------- #


def test_dry_run_sends_nothing(conn):
    fake = FakeJudge()
    report = judge.run_judge(conn, PRO_RUN, fake, dry_run=True)
    assert report.planned == 291 and fake.calls == 0
    assert 0 < report.estimated_cost_usd < 1


def test_verdicts_are_stored_and_never_paid_for_twice(conn):
    fake = FakeJudge()
    first = judge.run_judge(conn, PRO_RUN, fake, rubric_names=["claims_match_actions"])
    # Some replies are identical across repeats: each distinct question is sent once
    # and the other replies get a copy of its verdict.
    assert first.judged == fake.calls < 198
    assert first.judged + first.reused == 198
    assert first.cost_usd == pytest.approx(
        first.judged * (200 * 1.32 + 800 * 0.044 + 20 * 3.96) / 1e6
    )
    copies = conn.execute("SELECT count(*), sum(cost_usd) FROM judgments WHERE cached = 1")
    assert tuple(copies.fetchone()) == (first.reused, 0.0)

    calls = fake.calls
    second = judge.run_judge(conn, PRO_RUN, fake, rubric_names=["claims_match_actions"])
    assert (second.already_judged, second.judged, fake.calls) == (198, 0, calls)

    summary = judge.judge_summary(conn, PRO_RUN)
    # The fake fails replies containing "kaydediyorum" ("I am saving it"): the
    # v4-pro reply that claimed to save a date without saving it is among them.
    failed = {(f["case_id"], f["repeat_index"], f["turn_index"]) for f in summary["failures"]}
    assert ("rag_during_booking", 1, 0) in failed
    assert summary["rubrics"]["claims_match_actions"]["total"] == 198


def test_limit_and_errors(conn):
    class Broken(FakeJudge):
        def complete(self, system, user):
            raise RuntimeError("API down")

    report = judge.run_judge(conn, PRO_RUN, Broken(), limit=3)
    assert report.judged == 0 and len(report.errors) == 3
    assert "API down" in report.errors[0]
    assert conn.execute("SELECT count(*) FROM judgments").fetchone()[0] == 0


def test_keyword_checks_and_judge_are_compared_on_the_same_turns(conn):
    judge.run_judge(conn, PRO_RUN, FakeJudge(fail_word="200 TL"), rubric_names=["grounded_answer"])
    rag = judge.judge_summary(conn, PRO_RUN)["rag_keyword_vs_judge"]
    assert rag["compared"] > 0
    assert rag["agree"] + len(rag["disagreements"]) == rag["compared"]
    # The fake fails every answer that mentions the 200 TL fee, which keyword checks
    # pass: those show up as disagreements.
    assert all(d["verdict"] == "fail" and d["keyword_passed"] for d in rag["disagreements"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_dry_run_and_missing_key(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "j.db")
    main(["--db", db, "ingest", str(SAMPLE_DIR)])
    capsys.readouterr()

    assert main(["--db", db, "judge", "v4-pro", "--dry-run"]) == 0
    assert "291 rubric checks; 291 already judged" in capsys.readouterr().out

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("evalplat.cli._load_dotenv", lambda: None)
    assert main(["--db", db, "judge", "v4-pro"]) == 2

    assert main(["--db", db, "judgments", "v4-pro"]) == 0
    assert "291 verdicts by deepseek-v4-pro" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Real verdicts in the sample data (deepseek-v4-pro as judge)
# --------------------------------------------------------------------------- #


def test_sample_verdicts_catch_the_reply_that_claimed_an_unsaved_date():
    conn = store.connect(":memory:")
    ingest_paths(conn, [SAMPLE_DIR])
    summary = judge.judge_summary(conn, PRO_RUN)
    assert summary["judged"] == 291
    assert summary["rubrics"]["claims_match_actions"] == {
        "passed": 196,
        "total": 198,
        "rate": 196 / 198,
    }
    claims = {
        (f["case_id"], f["turn_index"])
        for f in summary["failures"]
        if f["rubric"] == "claims_match_actions"
    }
    assert claims == {("rag_during_booking", 0), ("rag_during_booking", 2)}
    assert summary["rag_keyword_vs_judge"]["agree"] == 21


def test_exported_verdicts_load_into_another_database(tmp_path):
    source = store.connect(":memory:")
    ingest_paths(source, [SAMPLE_DIR])
    path = tmp_path / "judgments_pro.json"
    path.write_text(json.dumps(judge.export_judgments(source, PRO_RUN)))

    target = store.connect(":memory:")
    ingest_paths(target, [SAMPLE_DIR / "results", SAMPLE_DIR / "traces", path])
    assert judge.judge_summary(target, PRO_RUN)["judged"] == 291
    # Loading again adds nothing; verdicts for unknown conversations are rejected.
    [again] = ingest_paths(target, [path])
    assert (again.added, again.duplicates) == (0, 291)
    [orphan] = ingest_paths(store.connect(":memory:"), [path])
    assert orphan.invalid == 291


# --------------------------------------------------------------------------- #
# Rubric v2 (after calibration)
# --------------------------------------------------------------------------- #


def test_summary_rubric_skips_turns_where_nothing_changed(conn):
    items = [
        i
        for i in judge.run_items(conn, PRO_RUN)
        if i["case_id"] == "no_confirm_user_hesitates" and i["repeat_index"] == 0
    ]
    completing, hesitating = items[0], items[1]
    rubric = RUBRICS["confirmation_summary"]
    assert rubric.applies(completing)  # the turn that filled the form
    assert not rubric.applies(hesitating)  # "let me think": the summary was already shown


def test_stale_verdicts_are_removed_and_valid_ones_kept(conn):
    """Verdicts made under an old rubric rule disappear when the rule stops asking
    that question; verdicts for questions still asked are kept, not paid again."""
    fake = FakeJudge()
    judge.run_judge(conn, PRO_RUN, fake, rubric_names=["confirmation_summary"])
    case = conn.execute(
        "SELECT id, conversation_id FROM case_results"
        " WHERE run_id = ? AND case_id = 'no_confirm_user_hesitates'",
        (PRO_RUN,),
    ).fetchone()
    with store.transaction(conn):  # a verdict the v1 rule produced
        conn.execute(
            "INSERT INTO judgments (case_result_id, conversation_id, turn_index, rubric,"
            " rubric_version, prompt_hash, judge_model, verdict, created_at)"
            " VALUES (?, ?, 1, 'confirmation_summary', '1', 'old', 'deepseek-v4-pro', 'fail', 'x')",
            (case["id"], case["conversation_id"]),
        )
    calls = fake.calls
    report = judge.run_judge(conn, PRO_RUN, fake, rubric_names=["confirmation_summary"])
    assert (report.stale, report.judged, fake.calls) == (1, 0, calls)
    versions = {r[0] for r in conn.execute("SELECT DISTINCT rubric_version FROM judgments")}
    assert versions == {"2"}
