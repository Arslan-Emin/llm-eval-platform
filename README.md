# LLM Eval & Observability Platform

Ingests traces and eval results from an LLM agent, compares eval runs, flags
regressions, scores replies with an LLM judge calibrated against human labels, and
gates CI on eval results.

Built to evaluate [conversational-form-agent](https://github.com/Arslan-Emin/conversational-form-agent).

> Work in progress.

## Usage

```bash
evalplat ingest data/sample      # eval results (.json) and traces (.jsonl) into evalplat.db
evalplat runs                    # list runs
evalplat show latest             # metrics of one run
evalplat compare baseline latest # candidate vs baseline, case by case (add --json for tooling)
```

### Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run streamlit_app.py   # reads evalplat.db (or $EVALPLAT_DB)
```

Pages: run history with pass-rate (95% confidence band), cost and latency trends;
one run (categories, check types, failed conversations with their transcripts); a
comparison of two runs; trace sources; and one conversation with a timeline of
every LLM and tool call. The selected run or conversation is part of the URL, so
any view can be shared as a link. With `EVALPLAT_SEED_DIR=data/sample` an empty
database is filled from the sample data on start (used by the public demo).

### Live traces

```bash
pip install -e ".[server]"
evalplat serve                   # POST /v1/events on http://127.0.0.1:8700
```

The agent sends its events while it runs by setting
`TRACE_HTTP_URL=http://127.0.0.1:8700/v1/events` (its `HttpExporter` queues events
and posts them in batches from a background thread). Live events show up in the
dashboard under the source `live`. Posting an event twice is harmless, so clients
can retry. Set `EVALPLAT_INGEST_TOKEN` to require `Authorization: Bearer <token>`.
The server binds to localhost and is not part of the public demo.

### LLM judge

```bash
pip install -e ".[judge]"        # needs DEEPSEEK_API_KEY in the environment or .env
evalplat judge v4-pro --dry-run  # what would be judged, and an upper-bound cost
evalplat judge v4-pro            # score the run's replies
evalplat judgments v4-pro        # verdicts, failures, judge vs keyword checks
evalplat judgments v4-pro --export data/sample/judgments   # save verdicts for ingest
```

Keyword checks only see the end state and fixed phrases. The judge
(`deepseek-v4-pro` in JSON mode, thinking off) answers one yes/no question per agent
reply, with the data the agent had in that turn, rebuilt from the trace:

| Rubric | Applies to | Question |
|---|---|---|
| `grounded_answer` | replies after a knowledge base search | Is every clinic fact supported by the retrieved passages (or does it say it doesn't know)? |
| `claims_match_actions` | every reply | Is what the reply says about the form true (saved values, submission)? |
| `confirmation_summary` | replies once every field is filled | Does it show every saved value correctly and ask for confirmation? |

Whether a rubric applies is decided in code, not by the judge. Verdicts are stored
per reply with the hash of the exact prompt: re-running never pays twice, identical
questions (a reply repeated word for word across repeats) are sent once, and editing
a rubric makes its old verdicts stale automatically.

### Calibrating the judge with human labels

```bash
evalplat label-sample --size 60   # pick verdicts to label (all judge fails + balanced passes)
# label them on the dashboard's "Calibrate judge" page, then:
evalplat calibration              # agreement, kappa, confusion matrix, disagreements
```

A person answers the judge's questions blind (the verdict is shown only after
answering), with a third option for "this question should not have been asked
about this reply". 60 verdicts from three runs were labeled
([labels](data/sample/labels/labels_human-1.json)):

| | Judge `confirmation_summary` v1 | v2 |
|---|---|---|
| Judge vs human on the questions that applied | 51 / 51 agree | 51 / 51 agree |
| Labeled "question doesn't fit" | 9 | 0 |
| `confirmation_summary` failures across the three runs | 9 (all false alarms) | 0 |

([v1](data/sample/calibration/rubric_v1.json), [v2](data/sample/calibration/rubric_v2.json))

The judge itself answered every question the way the human did; what the labels
exposed was a rubric rule. v1 expected a summary in *every* turn once the form was
complete, so when a user said "let me think about it" right after seeing the
summary, the judge (correctly, by the letter of the question) failed the reply.
v2 asks for a summary only in a turn that saved a value. Re-judging cost nothing:
the questions that were still asked had the same prompts, so their stored
verdicts were reused, and the 9 stale ones were removed.

Two caveats: the labels contain only two real failures, so the judge's recall is
measured on very few positives; and v2 was fixed using the same labels it is
evaluated on. Labeling also surfaced an error no check measures: one reply mixed
a Cyrillic word into Turkish ("телефонla").

### Regression gate

```bash
evalplat gate evals/results/<baseline>.json evals/results/   # newest run in the folder
```

Exit code 1 fails the CI job; on GitHub Actions the report also goes to the run's
summary page. LLMs are not deterministic, so the gate is a hybrid of hard limits
and statistics:

| Fails | Warns |
|---|---|
| any wrong or premature submit, any crashed case | a case that lost one repeat (flaky) |
| a significant pass-rate drop on the shared cases (one-sided Fisher, p < 0.05) | a smaller, non-significant drop |
| a case that newly fails in most of its repeats, or a new case that mostly fails | cost or latency up 25% or more; more submits blocked by the code guard |

On the agent's real runs: the fix passes; undoing the fix (75/75 → 72/75, three
cases each losing one of three repeats, p = 0.12) passes with warnings; switching to
`deepseek-v4-pro` passes with cost, latency and guard warnings. The gate only uses
the standard library, so `pip install` in another project's CI takes seconds.

### Notes

Re-running `ingest` is safe: runs and events that are already stored are skipped.
Runs can be referred to by id, name, a unique part of the name, `latest` or `previous`.

Runs are compared on the cases both contain (new and removed cases are listed
separately), and the change in pass rate gets a one-sided Fisher exact p-value: with
3 repeats per case, a few flipped results are often noise. Latency percentiles come
from trace events, so they are only shown for runs that have traces.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,dashboard,server,judge]"
pytest
ruff check . && ruff format --check .
```
