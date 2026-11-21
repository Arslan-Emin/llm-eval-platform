"""Command-line entry point: `evalplat <command>` (or `python -m evalplat <command>`).

Each command registers a handler with `set_defaults(func=...)`; the handler gets
the parsed arguments and returns the process exit code.
"""

from __future__ import annotations

import argparse

from evalplat import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalplat",
        description="Eval and observability platform for LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
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
