# Sample data

Real eval runs and traces from
[conversational-form-agent](https://github.com/Arslan-Emin/conversational-form-agent)
(runs up to commit `cb331aa`), used by the tests and the public demo.

| File | What it is |
|---|---|
| `results/20261119-225827_deepseek-flash_baseline.json` | Baseline run, 25 cases x 3, 72/75 passed |
| `results/20261119-225955_deepseek-flash.json` | After fix 1, 75/75 |
| `results/20261119-231456_deepseek-flash.json` | After fix 2, 26 cases x 3, 78/78 |
| `results/20261121-152221_deepseek-flash.json` | After fix 3, 30 cases x 3, 90/90 |
| `results/20261121-165646_deepseek-v4-pro.json` | Same suite on `deepseek-v4-pro`, thinking off: 89/90 |
| `results/20261121-165926_deepseek-flash.json` | Same suite on `deepseek-flash` with thinking on: 90/90 |
| `traces/eval_*.jsonl` | Full trace events of every run since tracing was added (the first two predate it) |
| `traces/2026-11-*.jsonl` | Traces of the app itself; message content is not logged there |
| `judgments/judgments_*.json` | LLM judge verdicts (`deepseek-v4-pro`) for the last three runs |
| `labels/labels_human-1.json` | 60 human labels of judge questions, for calibration |
| `calibration/rubric_v1.json`, `rubric_v2.json` | Judge vs human agreement before and after the rubric fix |

All names, phone numbers and messages in the eval data are fictional.
