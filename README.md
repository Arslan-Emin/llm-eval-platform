# LLM Eval & Observability Platform

Ingests traces and eval results from an LLM agent, compares eval runs, flags
regressions, scores replies with an LLM judge calibrated against human labels, and
gates CI on eval results.

Built to evaluate [conversational-form-agent](https://github.com/Arslan-Emin/conversational-form-agent).

> Work in progress.

## Usage

```bash
evalplat ingest data/sample      # eval results (.json) and traces (.jsonl) into evalplat.db
```

Re-running `ingest` is safe: runs and events that are already stored are skipped.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```
