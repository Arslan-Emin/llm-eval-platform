# Sample data

Real eval runs and traces from
[conversational-form-agent](https://github.com/Arslan-Emin/conversational-form-agent)
(commit `f62f0df`), used by the tests and the public demo.

| File | What it is |
|---|---|
| `results/20261119-225827_deepseek-flash_baseline.json` | Baseline run, 25 cases x 3, 72/75 passed |
| `results/20261119-225955_deepseek-flash.json` | After fix 1, 75/75 |
| `results/20261119-231456_deepseek-flash.json` | After fix 2, 26 cases x 3, 78/78 |
| `results/20261121-152221_deepseek-flash.json` | After fix 3, 30 cases x 3, 90/90 |
| `traces/eval_*.jsonl` | Full trace events of the last two runs (the first two predate tracing) |
| `traces/2026-11-*.jsonl` | Traces of the app itself; message content is not logged there |

All names, phone numbers and messages in the eval data are fictional.
