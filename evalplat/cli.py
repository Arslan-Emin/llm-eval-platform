"""Command-line entry point: `evalplat <command>` (or `python -m evalplat <command>`).

Each command registers a handler with `set_defaults(func=...)`; the handler gets
the parsed arguments and returns the process exit code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evalplat import __version__, store
from evalplat.ingest import ingest_paths


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        reports = ingest_paths(conn, args.paths)
    except FileNotFoundError as error:
        print(f"No such file or folder: {error}")
        return 2
    for report in reports:
        print(report.line())
    added = sum(r.added for r in reports if r.kind == "results")
    events = sum(r.added for r in reports if r.kind == "traces")
    invalid = sum(r.invalid for r in reports)
    db_path = store.resolve_db_path(args.db)
    print(f"\n{added} case result(s) and {events} trace event(s) added to {db_path}")
    # Partly invalid input is reported, but still a failure for scripts and CI.
    return 1 if invalid else 0


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
