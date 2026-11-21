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
pip install -e ".[dev,dashboard]"
pytest
ruff check . && ruff format --check .
```
