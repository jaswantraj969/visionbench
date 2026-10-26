"""Command-line entry point: ``visionbench``.

Expected errors print ``error: <message>`` and exit 1 without a traceback; ``--debug`` shows it.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from visionbench.core.errors import VisionBenchError

if TYPE_CHECKING:
    from visionbench.core.config import Settings


def _settings() -> Settings:
    from visionbench.core.config import Settings

    return Settings()


def _pick(value: Any, default: Any) -> Any:
    return default if value is None else value


def _cmd_env(_: argparse.Namespace) -> int:
    from visionbench.core.env_info import collect_env_info

    print(json.dumps(collect_env_info(), indent=2))
    return 0


def _print_issue_counts(issue_counts: dict[str, int]) -> None:
    nonzero = {k: v for k, v in issue_counts.items() if v}
    if nonzero:
        print("  issues: " + ", ".join(f"{k}={v}" for k, v in sorted(nonzero.items())))
    else:
        print("  issues: none")


def _print_report_summary(report: dict[str, Any]) -> None:
    dup = report["duplicates"]
    counts = ", ".join(f"{c}={n}" for c, n in report["class_counts"].items())
    print(
        f"  classes: {len(report['classes'])} | images: {report['total_images']} | "
        f"bytes: {report['total_bytes']}"
    )
    print(f"  class counts: {counts} | imbalance ratio: {report['imbalance_ratio']}")
    _print_issue_counts({k: v["count"] for k, v in report["issues"].items()})
    print(
        f"  duplicate groups (>1 member): {dup['multi_member_groups']} "
        f"({dup['files_in_multi_member_groups']} files) | exact duplicate files: "
        f"{dup['exact_duplicate_files']} | label-conflict groups: "
        f"{len(dup['label_conflict_groups'])}"
    )
    print(
        f"  perceptual check skipped (low information): "
        f"{dup['perceptual_check_skipped_low_information']['count']} "
        f"(std < {dup['low_info_std_threshold']} or degenerate dHash; "
        f"dHash threshold {dup['dhash_threshold']})"
    )


def _cmd_dataset_import(args: argparse.Namespace) -> int:
    from visionbench.data.scan import ScanConfig
    from visionbench.data.store import import_dataset

    s = _settings()
    config = ScanConfig(
        max_image_pixels=_pick(args.max_image_pixels, s.max_image_pixels),
        min_images_per_class=_pick(args.min_images_per_class, s.min_images_per_class),
        dhash_threshold=_pick(args.dhash_threshold, s.dhash_threshold),
        low_info_std=_pick(args.low_info_std, s.low_info_std_threshold),
        samples_per_class=s.report_samples_per_class,
    )
    result = import_dataset(Path(args.src), args.name, s.data_dir("datasets"), config)
    print(f"imported dataset {result.dataset_id} ({result.dataset['name']})")
    print(f"  path: {result.path}")
    print(f"  fingerprint: {result.dataset['fingerprint']}")
    _print_report_summary(result.report)
    return 0


def _cmd_dataset_report(args: argparse.Namespace) -> int:
    from visionbench.data.store import load_report

    report = load_report(_settings().data_dir("datasets"), args.dataset_id)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print(f"dataset {report['dataset_id']} ({report['name']})")
    print(f"  fingerprint: {report['fingerprint']}")
    _print_report_summary(report)
    w, h = report["width"], report["height"]
    print(
        f"  width min/median/max: {w['min']}/{w['median']}/{w['max']} | "
        f"height: {h['min']}/{h['median']}/{h['max']}"
    )
    print(f"  modes: {report['mode_counts']} | formats: {report['format_counts']}")
    return 0


def _cmd_dataset_list(_: argparse.Namespace) -> int:
    from visionbench.data.store import list_datasets

    datasets = list_datasets(_settings().data_dir("datasets"))
    if not datasets:
        print("no datasets")
    for d in datasets:
        print(
            f"{d['id']}  {d['name']}  classes={len(d['classes'])}  "
            f"images={d['total_images']}  created={d['created_at']}"
        )
    return 0


def _cmd_dataset_split(args: argparse.Namespace) -> int:
    from visionbench.data.split import SPLIT_NAMES, SplitConfig, make_split

    s = _settings()
    config = SplitConfig(
        seed=args.seed,
        fractions=tuple(args.fractions),
        dhash_threshold=_pick(args.dhash_threshold, s.dhash_threshold),
        low_info_std=_pick(args.low_info_std, s.low_info_std_threshold),
        conflict_policy=_pick(args.conflict_policy, s.conflict_policy),
        allow_missing_classes=args.allow_missing_classes,
    )
    result = make_split(s.data_dir("datasets"), args.dataset_id, config)
    m = result.manifest
    state = "already existed (recomputed manifest identical)" if result.already_existed else "new"
    print(f"split {result.split_id} for dataset {m['dataset_id']}: {state}")
    print(f"  manifest: {result.path / 'manifest.json'}")
    print(f"  manifest sha256: {result.manifest_sha256}")
    print(
        f"  seed={config.seed} fractions={list(config.fractions)} "
        f"conflict_policy={config.conflict_policy}"
    )
    print("  counts (train/val/test/total):")
    for label, c in m["counts"].items():
        print(f"    {label}: " + "/".join(str(c[k]) for k in (*SPLIT_NAMES, "total")))
    totals = m["split_totals"]
    print("  totals: " + ", ".join(f"{k}={totals[k]}" for k in SPLIT_NAMES))
    print(
        f"  excluded (label conflicts): {len(m['excluded'])} files in {m['excluded_groups']} groups"
    )
    for warning in m["warnings"]:
        print(f"  warning: {warning}")
    lc = m["leakage_check"]
    print(
        f"  leakage check: groups_in_multiple_splits={lc['groups_in_multiple_splits']} "
        f"sha256_in_multiple_splits={lc['sha256_in_multiple_splits']} "
        f"(checked {lc['groups_checked']} groups, {lc['sha256_checked']} sha256 values)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    from visionbench import __version__

    # `--debug` is accepted before or after subcommands. Subparsers get their own action with a
    # SUPPRESS default so they never overwrite a root-level `--debug` (parents share action
    # objects, so the root must not reuse this one).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug", action="store_true", default=argparse.SUPPRESS, help="show tracebacks for errors"
    )

    parser = argparse.ArgumentParser(prog="visionbench", description="VisionBench CLI")
    parser.add_argument("--debug", action="store_true", help="show tracebacks for errors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", parents=[common], help="print environment information as JSON")
    env.set_defaults(func=_cmd_env)

    dataset = sub.add_parser("dataset", parents=[common], help="dataset snapshots and splits")
    dsub = dataset.add_subparsers(dest="dataset_command", required=True)

    imp = dsub.add_parser("import", parents=[common], help="validate and snapshot a folder")
    imp.add_argument("src", help="source folder: <src>/<class>/<images>")
    imp.add_argument("--name", required=True)
    imp.add_argument("--max-image-pixels", type=int)
    imp.add_argument("--min-images-per-class", type=int)
    imp.add_argument("--dhash-threshold", type=int)
    imp.add_argument("--low-info-std", type=float)
    imp.set_defaults(func=_cmd_dataset_import)

    rep = dsub.add_parser("report", parents=[common], help="show a dataset report")
    rep.add_argument("dataset_id")
    rep.add_argument("--json", action="store_true", help="print the full report.json")
    rep.set_defaults(func=_cmd_dataset_report)

    lst = dsub.add_parser("list", parents=[common], help="list imported datasets")
    lst.set_defaults(func=_cmd_dataset_list)

    spl = dsub.add_parser("split", parents=[common], help="create a train/val/test manifest")
    spl.add_argument("dataset_id")
    spl.add_argument("--seed", type=int, required=True)
    spl.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    spl.add_argument("--dhash-threshold", type=int)
    spl.add_argument("--low-info-std", type=float)
    spl.add_argument("--conflict-policy", choices=["exclude", "error"])
    spl.add_argument("--allow-missing-classes", action="store_true")
    spl.set_defaults(func=_cmd_dataset_split)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on non-ASCII names in a legacy console
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        if isinstance(exc, VisionBenchError):
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
