"""Command-line entry point: ``visionbench``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _cmd_env(_: argparse.Namespace) -> int:
    from visionbench.core.env_info import collect_env_info

    print(json.dumps(collect_env_info(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from visionbench import __version__

    parser = argparse.ArgumentParser(prog="visionbench", description="VisionBench CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="print environment information as JSON")
    env.set_defaults(func=_cmd_env)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
