"""Command-line entry point: `evalplat <command>` (or `python -m evalplat <command>`).

Each command registers a handler with `set_defaults(func=...)`; the handler gets
the parsed arguments and returns the process exit code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evalplat import __version__, metrics, report, store
from evalplat.ingest import ingest_paths


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        reports = ingest_paths(conn, args.paths)
    except FileNotFoundError as error:
        print(f"No such file or folder: {error}")
        return 2
    for file_report in reports:
        print(file_report.line())
    added = sum(r.added for r in reports if r.kind == "results")
    events = sum(r.added for r in reports if r.kind == "traces")
    invalid = sum(r.invalid for r in reports)
    db_path = store.resolve_db_path(args.db)
    print(f"\n{added} case result(s) and {events} trace event(s) added to {db_path}")
    # Partly invalid input is reported, but still a failure for scripts and CI.
    return 1 if invalid else 0


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_runs(args: argparse.Namespace) -> int:
    runs = metrics.list_runs(store.connect(args.db))
    if args.json:
        _print_json(runs)
    else:
        print(report.format_runs(runs))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        result = metrics.run_metrics(conn, metrics.resolve_run(conn, args.run)["id"])
    except metrics.RunNotFound as error:
        print(error)
        return 2
    if args.json:
        _print_json(result)
    else:
        print(report.format_run(result))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        baseline = metrics.resolve_run(conn, args.baseline)
        candidate = metrics.resolve_run(conn, args.candidate)
    except metrics.RunNotFound as error:
        print(error)
        return 2
    result = metrics.compare_runs(conn, baseline["id"], candidate["id"])
    if args.json:
        _print_json(result)
    else:
        print(report.format_comparison(result))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    try:
        from evalplat import judge
    except ImportError:
        print('The judge needs extra packages: pip install "evalplat[judge]"')
        return 2
    conn = store.connect(args.db)
    try:
        run = metrics.resolve_run(conn, args.run)
    except metrics.RunNotFound as error:
        print(error)
        return 2
    client = None
    if not args.dry_run:
        _load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set (environment or .env).")
            return 2
        client = judge.DeepSeekJudge(api_key, args.model)

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    result = judge.run_judge(
        conn,
        run["id"],
        client,
        model=args.model,
        rubric_names=args.rubric,
        limit=args.limit,
        workers=args.workers,
        dry_run=args.dry_run,
        progress=progress,
    )
    print(
        f"Run {run['name']}: {result.planned} rubric checks; {result.already_judged} already "
        f"judged, {result.reused} answered by an identical earlier question, "
        f"{result.stale} stale verdicts {'to remove' if args.dry_run else 'removed'}."
    )
    print(
        f"To send to {args.model}: {result.questions} distinct questions, "
        f"~{result.estimated_input_tokens:,} input tokens, at most "
        f"{report.fmt_usd(result.estimated_cost_usd)} (peak price, no prompt cache)."
    )
    if args.dry_run:
        return 0
    print(f"Sent {result.judged}, cost {report.fmt_usd(result.cost_usd)}.")
    for error in result.errors:
        print(f"  error: {error}")
    return 1 if result.errors else 0


def cmd_judgments(args: argparse.Namespace) -> int:
    from evalplat import judge

    conn = store.connect(args.db)
    try:
        run = metrics.resolve_run(conn, args.run)
    except metrics.RunNotFound as error:
        print(error)
        return 2
    if args.export:
        path = Path(args.export)
        if path.is_dir():
            path = path / f"judgments_{run['name']}.json"
        data = judge.export_judgments(conn, run["id"])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {len(data['judgments'])} verdicts to {path}")
        return 0
    summary = judge.judge_summary(conn, run["id"])
    if args.json:
        _print_json(summary)
    else:
        print(f"Run {run['name']}")
        print(report.format_judgments(summary))
    return 0


def cmd_label_sample(args: argparse.Namespace) -> int:
    from evalplat import calibration

    conn = store.connect(args.db)
    count = calibration.build_label_queue(conn, size=args.size, seed=args.seed)
    print(f"Label queue: {count} replies. Label them on the dashboard's 'Calibrate judge' page.")
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    from evalplat import calibration

    conn = store.connect(args.db)
    if args.export:
        path = Path(args.export)
        if path.is_dir():
            path = path / f"labels_{args.labeler or 'all'}.json"
        data = calibration.export_labels(conn, args.labeler)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {len(data['labels'])} labels to {path}")
        return 0
    result = calibration.agreement(conn, labeler=args.labeler)
    if args.json:
        _print_json(result)
    else:
        print(report.format_calibration(result))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from evalplat import gate

    rules = gate.GateRules(
        alpha=args.alpha,
        max_case_drop=args.max_case_drop,
        max_cost_increase=args.max_cost_increase,
        max_latency_increase=args.max_latency_increase,
    )
    try:
        result = gate.run_gate(args.baseline, args.candidate, rules)
    except gate.GateInputError as error:
        print(f"Gate input error: {error}")
        return 2
    if args.json:
        _print_json(gate.to_dict(result))
    else:
        print(gate.format_text(result))
    gate.write_step_summary(result)
    return result.exit_code


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def cmd_serve(args: argparse.Namespace) -> int:
    # Imported here: the server needs the optional `server` extra, the rest of the
    # CLI (and the CI regression gate) must work without it.
    try:
        import uvicorn

        from evalplat.server import create_app
    except ImportError:
        print('The ingest server needs extra packages: pip install "evalplat[server]"')
        return 2
    uvicorn.run(create_app(args.db), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalplat",
        description="Eval and observability platform for LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--db",
        help=f"SQLite database file (default: ${store.DB_PATH_ENV} or {store.DEFAULT_DB_PATH})",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    ingest = commands.add_parser(
        "ingest", help="load eval result files (.json) and trace files (.jsonl)"
    )
    ingest.add_argument(
        "paths", nargs="+", type=Path, help="files or folders (searched recursively)"
    )
    ingest.set_defaults(func=cmd_ingest)

    run_help = "run id, name, unique name prefix, 'latest' or 'previous'"
    runs = commands.add_parser("runs", help="list eval runs")
    runs.add_argument("--json", action="store_true", help="print JSON instead of a table")
    runs.set_defaults(func=cmd_runs)

    show = commands.add_parser("show", help="metrics of one eval run")
    show.add_argument("run", nargs="?", default="latest", help=run_help)
    show.add_argument("--json", action="store_true", help="print JSON instead of a table")
    show.set_defaults(func=cmd_show)

    compare = commands.add_parser("compare", help="compare a candidate run against a baseline")
    compare.add_argument("baseline", help=run_help)
    compare.add_argument("candidate", nargs="?", default="latest", help=run_help)
    compare.add_argument("--json", action="store_true", help="print JSON instead of a table")
    compare.set_defaults(func=cmd_compare)

    judge = commands.add_parser("judge", help="score a run's replies with the LLM judge")
    judge.add_argument("run", nargs="?", default="latest", help=run_help)
    judge.add_argument("--rubric", action="append", help="only this rubric (repeatable)")
    judge.add_argument("--limit", type=int, help="judge at most this many replies")
    judge.add_argument("--workers", type=int, default=4, help="parallel API calls")
    judge.add_argument("--model", default="deepseek-v4-pro", help="judge model")
    judge.add_argument(
        "--dry-run", action="store_true", help="only count and estimate the cost, send nothing"
    )
    judge.set_defaults(func=cmd_judge)

    judgments = commands.add_parser("judgments", help="judge verdicts of a run")
    judgments.add_argument("run", nargs="?", default="latest", help=run_help)
    judgments.add_argument("--json", action="store_true", help="print JSON instead of text")
    judgments.add_argument(
        "--export", metavar="PATH", help="write the verdicts to a file (or folder) for ingest"
    )
    judgments.set_defaults(func=cmd_judgments)

    label_sample = commands.add_parser(
        "label-sample", help="pick judge verdicts for a human to label"
    )
    label_sample.add_argument("--size", type=int, default=60)
    label_sample.add_argument("--seed", type=int, default=7, help="same seed, same sample")
    label_sample.set_defaults(func=cmd_label_sample)

    calibration = commands.add_parser("calibration", help="judge vs human labels")
    calibration.add_argument("--labeler", help="only this labeler's labels")
    calibration.add_argument("--json", action="store_true", help="print JSON instead of text")
    calibration.add_argument(
        "--export", metavar="PATH", help="write the labels to a file or folder"
    )
    calibration.set_defaults(func=cmd_calibration)

    gate = commands.add_parser(
        "gate", help="fail (exit 1) if a candidate eval run regressed against a baseline"
    )
    gate.add_argument("baseline", type=Path, help="baseline results file")
    gate.add_argument(
        "candidate", type=Path, help="candidate results file, or a folder (its newest run)"
    )
    gate.add_argument("--alpha", type=float, default=0.05, help="significance level")
    gate.add_argument(
        "--max-case-drop",
        type=float,
        default=0.5,
        help="fail a case whose pass rate drops by more than this (default 0.5)",
    )
    gate.add_argument("--max-cost-increase", type=float, default=0.25, help="warn above this")
    gate.add_argument("--max-latency-increase", type=float, default=0.25, help="warn above this")
    gate.add_argument("--json", action="store_true", help="print JSON instead of text")
    gate.set_defaults(func=cmd_gate)

    serve = commands.add_parser("serve", help="receive trace events over HTTP (POST /v1/events)")
    serve.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: local only)"
    )
    serve.add_argument("--port", type=int, default=8700)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
