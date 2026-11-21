"""Command-line entry point: `evalplat <command>` (or `python -m evalplat <command>`).

Each command registers a handler with `set_defaults(func=...)`; the handler gets
the parsed arguments and returns the process exit code.
"""

from __future__ import annotations

import argparse
import json
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
