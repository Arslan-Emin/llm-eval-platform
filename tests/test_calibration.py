import pytest

from evalplat import calibration, store
from evalplat.ingest import ingest_paths
from tests.factories import SAMPLE_DIR


@pytest.fixture
def conn():
    conn = store.connect(":memory:")
    ingest_paths(conn, [SAMPLE_DIR])
    return conn


def verdicts(conn):
    return dict(conn.execute("SELECT prompt_hash, verdict FROM judgments"))


def test_queue_contains_every_judge_failure_and_balanced_passes(conn):
    assert calibration.build_label_queue(conn, size=60, seed=7) == 60
    queue = calibration.queue(conn, "h")
    verdict = verdicts(conn)
    distinct_fails = {h for h, v in verdict.items() if v == "fail"}

    assert {e["prompt_hash"] for e in queue} >= distinct_fails
    assert len({e["prompt_hash"] for e in queue}) == 60  # no question twice
    assert {e["rubric"] for e in queue} == {
        "claims_match_actions",
        "confirmation_summary",
        "grounded_answer",
    }
    # Shuffled: the failures are not all at the front.
    first = [verdict[e["prompt_hash"]] for e in queue[: len(distinct_fails)]]
    assert first.count("fail") < len(distinct_fails)


def test_same_seed_gives_the_same_queue(conn):
    calibration.build_label_queue(conn, seed=3)
    first = [e["prompt_hash"] for e in calibration.queue(conn, "h")]
    calibration.build_label_queue(conn, seed=3)
    assert [e["prompt_hash"] for e in calibration.queue(conn, "h")] == first


def test_labels_can_be_saved_changed_and_undone(conn):
    calibration.build_label_queue(conn, size=5)
    entry = calibration.queue(conn, "h")[0]

    calibration.save_label(conn, entry, "fail", labeler="h", note="claims a save")
    calibration.save_label(conn, entry, "pass", labeler="h")  # changed my mind
    assert calibration.queue(conn, "h")[0]["label"] == "pass"
    assert calibration.queue(conn, "someone else")[0]["label"] is None

    calibration.delete_label(conn, entry["prompt_hash"], "h")
    assert calibration.queue(conn, "h")[0]["label"] is None
    with pytest.raises(ValueError):
        calibration.save_label(conn, entry, "maybe", labeler="h")


def test_item_with_history_returns_earlier_turns(conn):
    conversation_id = conn.execute(
        "SELECT conversation_id FROM case_results"
        " WHERE case_id = 'no_confirm_user_hesitates' AND run_id = 5 LIMIT 1"
    ).fetchone()[0]
    item, earlier = calibration.item_with_history(conn, conversation_id, 1)
    assert "düşüneyim" in item["user_text"]
    assert len(earlier) == 1 and earlier[0]["turn_index"] == 0
    assert calibration.item_with_history(conn, conversation_id, 9) is None


def test_cohen_kappa():
    # Observed agreement 0.75, chance agreement 0.5 -> kappa 0.5.
    pairs = [("pass", "pass"), ("pass", "pass"), ("fail", "fail"), ("pass", "fail")]
    assert calibration.cohen_kappa(pairs) == pytest.approx(0.5)
    assert calibration.cohen_kappa([("pass", "pass")] * 3) is None  # undefined
    assert calibration.cohen_kappa([]) is None


def test_agreement_leaves_not_applicable_out_of_the_score(conn):
    calibration.build_label_queue(conn, size=60)
    verdict = verdicts(conn)
    entries = calibration.queue(conn, "h")
    fails = [e for e in entries if verdict[e["prompt_hash"]] == "fail"][:2]
    passes = [e for e in entries if verdict[e["prompt_hash"]] == "pass"][:3]
    calibration.save_label(conn, fails[0], "fail", labeler="h")  # agree
    calibration.save_label(conn, fails[1], "pass", labeler="h")  # judge too strict
    calibration.save_label(conn, passes[0], "not_applicable", labeler="h")
    calibration.save_label(conn, passes[1], "pass", labeler="h")  # agree
    calibration.save_label(conn, passes[2], "fail", labeler="h")  # judge missed it

    overall = calibration.agreement(conn, labeler="h")["overall"]
    assert (overall["labeled"], overall["scored"], overall["not_applicable"]) == (5, 4, 1)
    assert overall["accuracy"] == pytest.approx(0.5)
    assert overall["fail_precision"] == pytest.approx(0.5)  # 1 of 2 judge fails right
    assert overall["fail_recall"] == pytest.approx(0.5)  # caught 1 of 2 real fails
    assert sum(overall["confusion"].values()) == 4
    assert len(calibration.agreement(conn, labeler="h")["disagreements"]) == 2


def test_labels_export_and_load_into_another_database(conn, tmp_path, capsys):
    import json

    from evalplat.cli import main

    calibration.build_label_queue(conn, size=3)
    for entry in calibration.queue(conn, "h"):
        calibration.save_label(conn, entry, "pass", labeler="h")
    path = tmp_path / "labels_h.json"
    path.write_text(json.dumps(calibration.export_labels(conn, "h")), encoding="utf-8")

    target = store.connect(":memory:")
    reports = ingest_paths(target, [SAMPLE_DIR, path])
    assert reports[-1].kind == "labels" and reports[-1].added == 3
    assert calibration.agreement(target, labeler="h")["overall"]["labeled"] == 3

    db = str(tmp_path / "cli.db")
    main(["--db", db, "ingest", str(SAMPLE_DIR), str(path)])
    capsys.readouterr()
    assert main(["--db", db, "calibration", "--labeler", "h"]) == 0
    assert "judge pass" in capsys.readouterr().out
    assert main(["--db", db, "label-sample", "--size", "10"]) == 0
    assert "10 replies" in capsys.readouterr().out
