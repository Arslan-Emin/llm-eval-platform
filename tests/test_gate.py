import json

import pytest

from evalplat import gate
from evalplat.cli import main
from tests.factories import SAMPLE_DIR, make_case, make_results

RESULTS = SAMPLE_DIR / "results"
BASELINE = RESULTS / "20261119-225827_deepseek-flash_baseline.json"
FIX_1 = RESULTS / "20261119-225955_deepseek-flash.json"
FLASH = RESULTS / "20261121-152221_deepseek-flash.json"
PRO = RESULTS / "20261121-165646_deepseek-v4-pro.json"


def statuses(result):
    return {c.name: c.status for c in result.checks}


# --------------------------------------------------------------------------- #
# Real runs from the agent project
# --------------------------------------------------------------------------- #


def test_the_fix_passes_the_gate():
    result = gate.run_gate(BASELINE, FIX_1)
    assert result.status == "ok" and result.exit_code == 0


def test_undoing_the_fix_warns_but_does_not_fail():
    # 75/75 -> 72/75: three cases each lost one of three repeats. Not significant
    # (p = 0.12) and no case broke in most repeats, so CI stays green with warnings.
    result = gate.run_gate(FIX_1, BASELINE)
    assert result.status == "warn" and result.exit_code == 0
    assert statuses(result)["Pass rate"] == "warn"
    flaky = next(c for c in result.checks if c.name == "Flaky cases")
    assert "missing_last_name 3/3 -> 2/3" in flaky.detail


def test_switching_to_the_bigger_model_shows_cost_and_guard_warnings():
    result = gate.run_gate(FLASH, PRO)
    s = statuses(result)
    assert result.exit_code == 0
    assert s["Cost per conversation"] == "warn"
    assert s["Latency per user turn"] == "warn"
    assert s["Submits blocked by the code guard"] == "warn"


def test_a_folder_means_its_newest_run():
    assert gate.resolve_input(RESULTS).name == "20261121-165926_deepseek-flash.json"
    with pytest.raises(gate.GateInputError, match="same run"):
        gate.run_gate(RESULTS / "20261121-165926_deepseek-flash.json", RESULTS)


# --------------------------------------------------------------------------- #
# Synthetic runs: the failing rules
# --------------------------------------------------------------------------- #


def write_run(path, cases, run_at):
    path.write_text(json.dumps(make_results(cases, run_at=run_at)), encoding="utf-8")
    return path


def repeats(case_id, passed, total=3, **extra):
    return [{**make_case(case_id, passed=i < passed), **extra} for i in range(total)]


@pytest.fixture
def base(tmp_path):
    cases = [c for i in range(30) for c in repeats(f"case{i}", 3)]
    return write_run(tmp_path / "base.json", cases, "2026-01-01T00:00:00+00:00")


def candidate(tmp_path, cases):
    return write_run(tmp_path / "cand.json", cases, "2026-01-02T00:00:00+00:00")


def test_a_case_broken_in_most_repeats_fails(tmp_path, base):
    cases = [c for i in range(29) for c in repeats(f"case{i}", 3)] + repeats("case29", 1)
    result = gate.run_gate(base, candidate(tmp_path, cases))
    assert result.exit_code == 1
    assert statuses(result)["Broken cases"] == "fail"
    assert statuses(result)["Pass rate"] == "warn"  # 88/90 alone is not significant


def test_a_significant_drop_fails(tmp_path, base):
    # Eight cases each lose one repeat: no single case is broken, together they are.
    cases = [c for i in range(30) for c in repeats(f"case{i}", 2 if i < 8 else 3)]
    result = gate.run_gate(base, candidate(tmp_path, cases))
    assert statuses(result)["Pass rate"] == "fail"
    assert statuses(result)["Broken cases"] == "ok"
    assert result.exit_code == 1


def test_a_wrong_submit_always_fails(tmp_path, base):
    cases = [c for i in range(30) for c in repeats(f"case{i}", 3)]
    cases[0] = {**cases[0], "expected_submit": False, "submitted": True}
    result = gate.run_gate(base, candidate(tmp_path, cases))
    assert statuses(result)["Wrong or premature submits"] == "fail"


def test_a_new_case_that_mostly_fails_fails(tmp_path, base):
    cases = [c for i in range(30) for c in repeats(f"case{i}", 3)] + repeats("new_case", 0)
    result = gate.run_gate(base, candidate(tmp_path, cases))
    assert statuses(result)["Broken cases"] == "fail"
    assert "new_case (new) 0/3" in next(c.detail for c in result.checks if c.name == "Broken cases")


# --------------------------------------------------------------------------- #
# CLI and CI integration
# --------------------------------------------------------------------------- #


def test_cli_exit_codes_and_github_summary(tmp_path, base, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv(gate.STEP_SUMMARY_ENV, str(summary))

    assert main(["gate", str(BASELINE), str(FIX_1)]) == 0
    assert "Gate PASSED." in capsys.readouterr().out
    assert "### Eval regression gate passed" in summary.read_text(encoding="utf-8")

    broken = [c for i in range(29) for c in repeats(f"case{i}", 3)] + repeats("case29", 0)
    assert main(["gate", str(base), str(candidate(tmp_path, broken))]) == 1
    assert "Gate FAILED." in capsys.readouterr().out

    assert main(["gate", str(BASELINE), str(tmp_path / "missing.json")]) == 2
