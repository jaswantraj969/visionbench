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
    from visionbench.tracking.store import ExperimentStore


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
        f"(std < {dup['low_info_std_threshold']}, degenerate dHash, or popcount outside "
        f"[{dup.get('low_info_popcount_min', 0)}, {dup.get('low_info_popcount_max', 64)}]; "
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
        low_info_popcount_min=_pick(args.low_info_popcount_min, s.low_info_popcount_min),
        low_info_popcount_max=_pick(args.low_info_popcount_max, s.low_info_popcount_max),
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


def _cmd_dataset_verify(args: argparse.Namespace) -> int:
    from visionbench.data.store import verify_dataset

    result = verify_dataset(_settings().data_dir("datasets"), args.dataset_id, full=args.full)
    mode = "full (sizes + sha256)" if args.full else "fast (existence + sizes + fingerprint)"
    print(f"verify {args.dataset_id}: {mode}; {result.files_checked} files checked")
    if result.ok:
        print("  OK: snapshot matches its records")
        return 0
    for relpath, problem in result.problems:
        print(f"  MISMATCH {relpath}: {problem}")
    print(f"  {len(result.problems)} problem(s) found", file=sys.stderr)
    return 1


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
        low_info_popcount_min=_pick(args.low_info_popcount_min, s.low_info_popcount_min),
        low_info_popcount_max=_pick(args.low_info_popcount_max, s.low_info_popcount_max),
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


_TRAIN_FLAGS: tuple[tuple[str, type], ...] = (
    ("--dataset-id", str),
    ("--split-id", str),
    ("--model", str),
    ("--weights", str),
    ("--finetune", str),
    ("--input-size", int),
    ("--epochs", int),
    ("--max-steps", int),
    ("--batch-size", int),
    ("--optimizer", str),
    ("--lr", float),
    ("--weight-decay", float),
    ("--scheduler", str),
    ("--seed", int),
    ("--device", str),
    ("--num-workers", int),
)


def _fmt(value: Any, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def _cmd_train(args: argparse.Namespace) -> int:
    from visionbench.training.config import load_train_config, parse_override
    from visionbench.training.runner import run_training

    overrides: dict[str, Any] = {
        flag[2:].replace("-", "_"): getattr(args, flag[2:].replace("-", "_"))
        for flag, _ in _TRAIN_FLAGS
    }
    overrides["amp"] = args.amp
    overrides["allow_weight_fallback"] = args.allow_weight_fallback
    for item in args.set:
        key, value = parse_override(item)
        overrides[key] = value
    config = load_train_config(args.config, overrides)

    def on_start(exp_id: str, run_dir: Path) -> None:
        print(f"experiment {exp_id} running ({config.model}, seed={config.seed})")
        print(f"  run dir: {run_dir}")

    def on_epoch(r: dict[str, Any]) -> None:
        print(
            f"  epoch {r['epoch']:3d}  train_loss={_fmt(r['train_loss'])}  "
            f"val_loss={_fmt(r['val_loss'])}  val_acc={_fmt(r['val_accuracy'])}  "
            f"val_bal_acc={_fmt(r['val_balanced_accuracy'])}  lr={_fmt(r['lr'], '.2e')}  "
            f"{r['epoch_seconds']:.2f}s{'  *best' if r.get('is_best') else ''}"
        )

    result = run_training(config, _settings(), on_start=on_start, on_epoch=on_epoch)
    print(
        f"experiment {result.experiment_id}: {result.status} "
        f"(stopped: {result.stopped_reason}; best epoch {result.best_epoch}, "
        f"best {config.early_stopping.monitor}={_fmt(result.best_value)})"
    )
    return 0


def _open_store() -> ExperimentStore:
    from visionbench.tracking.store import DB_FILENAME, ExperimentStore

    return ExperimentStore(_settings().data_dir("db") / DB_FILENAME)


def _cmd_experiments_list(_: argparse.Namespace) -> int:
    with _open_store() as store:
        rows = store.list()
    if not rows:
        print("no experiments")
    for r in rows:
        weights = r["weights_source"] or "-"
        best_epoch = r["best_epoch"] or "-"
        print(
            f"{r['id']}  {r['status']:9s}  {r['model']:17s}  weights={weights:8s}  "
            f"seed={r['seed']}  best_epoch={best_epoch}  best={_fmt(r['best_metric'])}  "
            f"{r['dataset_id']}/{r['split_id']}  {r['created_at']}"
        )
    return 0


def _cmd_experiments_show(args: argparse.Namespace) -> int:
    with _open_store() as store:
        record = store.get(args.experiment_id)
        epochs = store.epochs(args.experiment_id)
        artifacts = store.artifacts(args.experiment_id)
    if args.json:
        print(
            json.dumps(
                {"experiment": record, "epochs": epochs, "artifacts": artifacts},
                indent=2,
                default=str,
            )
        )
        return 0
    skip = {"config_json", "env_json"}
    for key, value in record.items():
        if key not in skip:
            print(f"{key}: {value}")
    print("epochs:")
    for e in epochs:
        print(
            f"  {e['epoch']:3d}  train_loss={_fmt(e['train_loss'])}  "
            f"val_loss={_fmt(e['val_loss'])}  val_bal_acc={_fmt(e['val_balanced_accuracy'])}"
        )
    print("artifacts:")
    for a in artifacts:
        print(f"  {a['kind']:15s} {a['relpath']}  {a['size_bytes']} B  sha256={a['sha256']}")
    return 0


def _ci_text(value: Any, ci: dict[str, Any]) -> str:
    return f"{_fmt(value)} [{_fmt(ci.get('low'))}, {_fmt(ci.get('high'))}]"


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from visionbench.evaluation.evaluate import evaluate_experiment

    def warn(message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr)

    r = evaluate_experiment(
        args.experiment_id,
        args.split,
        _settings(),
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap,
        seed=args.seed,
        force=args.force,
        reason=args.reason,
        on_warning=warn,
    )
    doc, m = r.metrics, r.metrics["metrics"]
    ci = doc["bootstrap"]["metrics"]
    ck = doc["checkpoint"]
    tu = doc["test_use"]
    print(
        f"evaluation {r.eval_id}: experiment {r.experiment_id} ({doc['model']}) on "
        f"{r.split.upper()} with checkpoint '{ck['name']}' (epoch {ck['epoch']}, "
        f"sha256 {ck['sha256'][:12]}...)"
    )
    print(f"  n={m['n']}  accuracy={_ci_text(m['accuracy'], ci['accuracy'])}")
    print(
        f"  balanced_accuracy={_ci_text(m['balanced_accuracy'], ci['balanced_accuracy'])}  "
        f"(excluded absent classes: {m['balanced_accuracy_excluded_classes'] or 'none'})"
    )
    print(
        f"  macro_f1={_ci_text(m['macro_f1'], ci['macro_f1'])}  "
        f"weighted_f1={_fmt(m['weighted_f1'])}  log_loss={_fmt(m['log_loss'])}"
    )
    print(
        f"  roc_auc_ovr_macro={_fmt(m['roc_auc_ovr_macro'])}  "
        f"ap_macro={_fmt(m['average_precision_macro'])}  "
        f"ece={_fmt(m['calibration']['ece'] if m['calibration'] else None)}"
    )
    if m["undefined"]:
        print("  undefined (null): " + ", ".join(f"{k}={v}" for k, v in m["undefined"].items()))
    print(
        f"  CIs: {doc['bootstrap']['resamples']} bootstrap resamples, seed "
        f"{doc['bootstrap']['seed']}"
    )
    print(
        f"  test evaluations of this experiment+checkpoint before this one: "
        f"{tu['prior_test_evaluations_same_checkpoint']}"
    )
    print(f"  outputs: {r.eval_dir}")
    return 0


def _cmd_latency(args: argparse.Namespace) -> int:
    from visionbench.evaluation.latency import measure_experiment_latency

    r = measure_experiment_latency(
        args.experiment_id,
        _settings(),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    m = r.measurement
    print(
        f"latency {r.latency_id}: experiment {r.experiment_id} ({m['model']}) on {m['device']} "
        f"({m['device_name']}, {m['torch_threads']} threads, {m['input_size']} px, "
        f"warmup {m['warmup']}, {m['iterations']} iterations; {m['scope']})"
    )
    for bs, v in m["batch_sizes"].items():
        print(
            f"  batch {bs:>3}: median {v['median_ms']:.3f} ms  p95 {v['p95_ms']:.3f} ms  "
            f"({v['median_ms_per_image']:.3f} ms/image)"
        )
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    import logging

    from visionbench.jobs.worker import WorkerConfig, run_worker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    s = _settings()
    wc = WorkerConfig.from_settings(s)
    if args.poll_seconds is not None:
        wc = WorkerConfig(
            args.poll_seconds,
            wc.heartbeat_seconds,
            wc.stale_after_seconds,
            wc.cancel_timeout_seconds,
        )
    run_worker(s, wc, max_jobs=args.max_jobs)
    return 0


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _cmd_serve(args: argparse.Namespace) -> int:
    """API + one worker subprocess. Refuses non-loopback hosts: the API has no auth."""
    import os
    import subprocess

    import uvicorn

    from visionbench.api.main import create_app

    s = _settings()
    host = args.host or s.api_host
    port = args.port or s.api_port
    if host not in LOOPBACK_HOSTS:
        print(
            f"error: refusing to bind to {host!r}: the API has no authentication; "
            f"use one of {LOOPBACK_HOSTS}",
            file=sys.stderr,
        )
        return 2
    worker = None
    if not args.no_worker:
        env = os.environ | {"VISIONBENCH_DATA_ROOT": str(s.resolved_data_root)}
        worker = subprocess.Popen([sys.executable, "-m", "visionbench.cli", "worker"], env=env)
        print(f"worker started (pid {worker.pid})")
    try:
        uvicorn.run(create_app(s), host=host, port=port, log_level="info")
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(15)
            except subprocess.TimeoutExpired:
                worker.kill()
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from visionbench.evaluation.compare import compare_experiments

    report = compare_experiments(
        args.experiment_ids,
        args.split,
        _settings(),
        reference=args.reference,
        resamples=args.bootstrap,
        seed=args.seed,
        note=args.note,
        eval_ids=args.eval_ids,
        latency_device=args.latency_device,
    )
    print(
        f"comparison {report['report_id']} on {args.split.upper()}: "
        f"{len(report['rows'])} experiments, reference {report['reference_experiment']}"
    )
    print(f"  json: {report['paths']['json']}")
    print(f"  markdown: {report['paths']['markdown']}")
    print(f"  confound flags: {', '.join(c['code'] for c in report['confounds']) or 'none'}")
    return 0


def _cmd_evaluations_list(args: argparse.Namespace) -> int:
    with _open_store() as store:
        rows = store.evaluations(args.experiment_id)
    if not rows:
        print("no evaluations")
    for r in rows:
        print(
            f"{r['id']}  {r['experiment_id']}  split={r['split']:4s}  "
            f"checkpoint={r['checkpoint']:8s}  {r['created_at']}  {r['metrics_path']}"
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
    imp.add_argument("--low-info-popcount-min", type=int)
    imp.add_argument("--low-info-popcount-max", type=int)
    imp.set_defaults(func=_cmd_dataset_import)

    rep = dsub.add_parser("report", parents=[common], help="show a dataset report")
    rep.add_argument("dataset_id")
    rep.add_argument("--json", action="store_true", help="print the full report.json")
    rep.set_defaults(func=_cmd_dataset_report)

    ver = dsub.add_parser("verify", parents=[common], help="check a snapshot against its records")
    ver.add_argument("dataset_id")
    ver.add_argument("--full", action="store_true", help="also re-hash every file (SHA-256)")
    ver.set_defaults(func=_cmd_dataset_verify)

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
    spl.add_argument("--low-info-popcount-min", type=int)
    spl.add_argument("--low-info-popcount-max", type=int)
    spl.add_argument("--conflict-policy", choices=["exclude", "error"])
    spl.add_argument("--allow-missing-classes", action="store_true")
    spl.set_defaults(func=_cmd_dataset_split)

    trn = sub.add_parser("train", parents=[common], help="train one model on a split (explicit)")
    trn.add_argument("--config", type=Path, help="YAML training config")
    trn.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key (dotted, YAML value), e.g. "
        "early_stopping.patience=3; repeatable",
    )
    for flag, kind in _TRAIN_FLAGS:
        trn.add_argument(flag, type=kind)
    trn.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    trn.add_argument(
        "--allow-weight-fallback",
        action="store_true",
        default=None,
        help="if pretrained weights cannot be obtained, train from scratch "
        "(recorded as weights=scratch) instead of failing",
    )
    trn.set_defaults(func=_cmd_train)

    exps = sub.add_parser("experiments", parents=[common], help="list/show tracked experiments")
    esub = exps.add_subparsers(dest="experiments_command", required=True)
    elist = esub.add_parser("list", parents=[common], help="list experiments")
    elist.set_defaults(func=_cmd_experiments_list)
    eshow = esub.add_parser("show", parents=[common], help="show one experiment")
    eshow.add_argument("experiment_id")
    eshow.add_argument("--json", action="store_true", help="print record, epochs, artifacts")
    eshow.set_defaults(func=_cmd_experiments_show)

    ev = sub.add_parser(
        "evaluate", parents=[common], help="evaluate a trained experiment on val or test (recorded)"
    )
    ev.add_argument("experiment_id")
    ev.add_argument("--split", choices=["val", "test"], required=True)
    ev.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="best = val-selected (default); last is recorded as such",
    )
    ev.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs")
    ev.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ev.add_argument(
        "--force",
        action="store_true",
        help="allow a repeated TEST evaluation of the same experiment+checkpoint (needs --reason)",
    )
    ev.add_argument("--reason", help="why the TEST evaluation is repeated (stored and reported)")
    ev.set_defaults(func=_cmd_evaluate)

    lat = sub.add_parser(
        "latency", parents=[common], help="measure forward-pass latency (random input, no split)"
    )
    lat.add_argument("experiment_id")
    lat.add_argument("--device", choices=["cpu", "cuda"], required=True)
    lat.add_argument("--warmup", type=int, default=10)
    lat.add_argument("--iterations", type=int, default=50)
    lat.set_defaults(func=_cmd_latency)

    wrk = sub.add_parser(
        "worker", parents=[common], help="run queued jobs one at a time (single worker)"
    )
    wrk.add_argument("--poll-seconds", type=float, help="queue poll interval")
    wrk.add_argument(
        "--max-jobs", type=int, help="exit after this many jobs (default: run forever)"
    )
    wrk.set_defaults(func=_cmd_worker)

    srv = sub.add_parser(
        "serve", parents=[common], help="run the API (localhost only) plus one worker"
    )
    srv.add_argument("--host", help="loopback host (default 127.0.0.1)")
    srv.add_argument("--port", type=int, help="port (default 8000)")
    srv.add_argument("--no-worker", action="store_true", help="API only (run a worker yourself)")
    srv.set_defaults(func=_cmd_serve)

    cmp_ = sub.add_parser("compare", parents=[common], help="compare evaluated experiments")
    cmp_.add_argument("experiment_ids", nargs="+")
    cmp_.add_argument("--split", choices=["val", "test"], required=True)
    cmp_.add_argument(
        "--reference",
        help="experiment for paired differences "
        "(default: majority baseline if present, else first)",
    )
    cmp_.add_argument("--bootstrap", type=int, default=1000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--note", help="text shown at the top of the report")
    cmp_.add_argument(
        "--eval-ids", nargs="+", default=[], help="pin specific evaluations (one per experiment)"
    )
    cmp_.add_argument(
        "--latency-device", choices=["cpu", "cuda"], help="report latency measured on this device"
    )
    cmp_.set_defaults(func=_cmd_compare)

    evs = sub.add_parser("evaluations", parents=[common], help="list recorded evaluations")
    evsub = evs.add_subparsers(dest="evaluations_command", required=True)
    evl = evsub.add_parser("list", parents=[common], help="list evaluations")
    evl.add_argument("experiment_id", nargs="?")
    evl.set_defaults(func=_cmd_evaluations_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on non-ASCII names in a legacy console
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("cancelled (KeyboardInterrupt)", file=sys.stderr)
        return 130
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
