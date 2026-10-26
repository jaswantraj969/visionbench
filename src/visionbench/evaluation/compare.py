"""`visionbench compare`: side-by-side report of evaluated experiments with confound flags.

Rules:
- All experiments must share the dataset fingerprint and split_id (no override).
- Each experiment contributes its latest recorded evaluation on the chosen split with the
  val-selected checkpoint ("best", or "baseline"); evaluations are never run implicitly here.
- Any ranking / "best model" statement uses VALIDATION metrics only (balanced accuracy at the
  val-selected epoch). Chosen-split metrics (e.g. test) are reported for every model with CIs and
  paired differences, without a winner label.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from visionbench import __version__
from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json, write_json
from visionbench.evaluation.data import EVAL_SPLITS, EvaluationError
from visionbench.evaluation.metrics import DEFAULT_BOOTSTRAP, paired_bootstrap_difference
from visionbench.models.pretrained import file_sha256
from visionbench.tracking.store import DB_FILENAME, ExperimentStore

REPORT_FORMAT_VERSION = 1
SELECTED_CHECKPOINTS = ("best", "baseline")


def _read_predictions(path: Path) -> tuple[list[str], list[str], list[str]]:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    return (
        [r["relpath"] for r in rows],
        [r["true_label"] for r in rows],
        [r["pred_label"] for r in rows],
    )


def _entry(store: ExperimentStore, data_root: Path, exp_id: str, split: str) -> dict[str, Any]:
    exp = store.get(exp_id)
    evals = [
        e for e in store.evaluations(exp_id, split=split) if e["checkpoint"] in SELECTED_CHECKPOINTS
    ]
    if not evals:
        raise EvaluationError(
            f"experiment {exp_id} has no '{split}' evaluation of its val-selected checkpoint; "
            f"run `visionbench evaluate {exp_id} --split {split}` first"
        )
    ev = evals[-1]
    metrics_path = data_root / ev["metrics_path"]
    doc = read_json(metrics_path)
    pred_path = metrics_path.parent / "predictions.csv"
    if file_sha256(pred_path) != ev["predictions_sha256"]:
        raise EvaluationError(f"predictions for {ev['id']} do not match their recorded sha256")
    relpaths, y_true, y_pred = _read_predictions(pred_path)
    run_config = read_json(data_root / exp["run_dir"] / "config.json")
    eff = read_json(metrics_path.parent / "efficiency.json")
    epochs = store.epochs(exp_id)
    best = next((e for e in epochs if e["epoch"] == exp["best_epoch"]), None)
    return {
        "experiment": exp,
        "evaluation": ev,
        "metrics_doc": doc,
        "efficiency": eff,
        "config": run_config["config"],
        "resolved": run_config["resolved"],
        "relpaths": relpaths,
        "y_true": y_true,
        "y_pred": y_pred,
        "val_bal_acc_at_selected_epoch": best["val_balanced_accuracy"] if best else None,
        "epochs_trained": len(epochs),
        "test_evaluations": len(store.evaluations(exp_id, split="test")),
    }


def competition_ranks(values: dict[str, float]) -> dict[str, int]:
    """Higher is better; equal values share a rank ("1224" ranking)."""
    ordered = sorted(values.values(), reverse=True)
    return {k: ordered.index(v) + 1 for k, v in values.items()}


def _confounds(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    trained = [e for e in entries if e["experiment"]["model"] != "majority_baseline"]

    def values(items: list[dict[str, Any]], fn: Any) -> dict[str, Any]:
        return {e["experiment"]["id"]: fn(e) for e in items}

    def flag(code: str, message: str, vals: dict[str, Any]) -> None:
        flags.append({"code": code, "message": message, "values": vals})

    res = values(trained, lambda e: e["resolved"].get("input_size"))
    if len(set(res.values())) > 1:
        flag(
            "input_resolution_differs",
            "models see different input resolutions; architecture and resolution are confounded",
            res,
        )
    weights = values(trained, lambda e: e["experiment"]["weights_source"])
    if len(set(weights.values())) > 1:
        flag(
            "pretrained_vs_scratch",
            "ImageNet-pretrained and scratch models are compared; pretraining is a confound",
            weights,
        )
    monitor = values(trained, lambda e: e["config"]["early_stopping"]["monitor"])
    if len(set(monitor.values())) > 1:
        flag(
            "early_stopping_monitor_differs",
            "checkpoints were selected with different monitors",
            monitor,
        )
    budget = values(
        trained,
        lambda e: {
            "epochs": e["config"]["epochs"],
            "max_steps": e["config"]["max_steps"],
            "epochs_trained": e["epochs_trained"],
        },
    )
    if len({(b["epochs"], b["max_steps"]) for b in budget.values()}) > 1:
        flag("training_budget_differs", "configured epoch/step budgets differ", budget)
    seeds = values(entries, lambda e: e["experiment"]["seed"])
    if len(set(seeds.values())) > 1:
        flag("seeds_differ", "experiments used different seeds", seeds)
    per_model = Counter(e["experiment"]["model"] for e in entries)
    if all(n == 1 for n in per_model.values()):
        flag(
            "single_seed_per_model",
            "one run (one seed) per model: seed-to-seed variance is unmeasured, so differences "
            "may be within run-to-run noise",
            dict(per_model),
        )
    weighted = values(trained, lambda e: e["config"]["class_weighted_loss"])
    if len(set(weighted.values())) > 1:
        flag("class_weighting_differs", "some runs used class-weighted loss", weighted)
    devices = {
        e["experiment"]["id"]: e["efficiency"]["latency"].get("device")
        for e in trained
        if e["efficiency"]["latency"].get("applicable")
    }
    if len(set(devices.values())) > 1:
        flag("latency_devices_differ", "latency was measured on different devices", devices)
    test_uses = values(entries, lambda e: e["test_evaluations"])
    if any(n > 1 for n in test_uses.values()):
        flag(
            "repeated_test_evaluation",
            "at least one experiment has been evaluated on TEST more than once",
            test_uses,
        )
    return flags


def compare_experiments(
    exp_ids: list[str],
    split: str,
    settings: Settings,
    *,
    reference: str | None = None,
    resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    note: str | None = None,
) -> dict[str, Any]:
    if split not in EVAL_SPLITS:
        raise EvaluationError(f"split must be one of {EVAL_SPLITS}")
    if len(exp_ids) < 2 or len(set(exp_ids)) != len(exp_ids):
        raise EvaluationError("compare needs at least two distinct experiment ids")
    data_root = settings.resolved_data_root
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        exps = [store.get(x) for x in exp_ids]
        fps = {e["dataset_fingerprint"] for e in exps}
        splits = {e["split_id"] for e in exps}
        if len(fps) > 1 or len(splits) > 1:
            raise EvaluationError(
                "refusing to compare experiments on different data: "
                f"dataset fingerprints {sorted(fps)}, split_ids {sorted(splits)}"
            )
        entries = [_entry(store, data_root, x, split) for x in exp_ids]

    base = entries[0]
    for e in entries[1:]:
        if e["relpaths"] != base["relpaths"] or e["y_true"] != base["y_true"]:
            raise EvaluationError("evaluated samples/labels differ between experiments")
    class_names = base["metrics_doc"]["metrics"]["class_names"]
    index = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([index[c] for c in base["y_true"]])

    if reference is None:
        baselines = [e for e in entries if e["experiment"]["model"] == "majority_baseline"]
        reference = (baselines[0] if baselines else entries[0])["experiment"]["id"]
    if reference not in exp_ids:
        raise EvaluationError(f"reference {reference} is not among the compared experiments")
    ref = next(e for e in entries if e["experiment"]["id"] == reference)
    ref_pred = np.array([index[c] for c in ref["y_pred"]])

    rows = []
    for e in entries:
        exp, m, eff = e["experiment"], e["metrics_doc"]["metrics"], e["efficiency"]
        ci = e["metrics_doc"]["bootstrap"]["metrics"]
        lat = eff["latency"]
        row = {
            "experiment_id": exp["id"],
            "model": exp["model"],
            "weights_source": exp["weights_source"],
            "input_size": e["resolved"].get("input_size"),
            "params_total": eff["params"]["total"],
            "checkpoint_size_bytes": eff["checkpoint_size_bytes"],
            "latency_bs1_median_ms": (
                lat["batch_sizes"]["1"]["median_ms"] if lat.get("applicable") else None
            ),
            "latency_device": lat.get("device") if lat.get("applicable") else None,
            "latency_note": None if lat.get("applicable") else lat.get("reason"),
            "train_seconds": exp["train_seconds"],
            "best_epoch": exp["best_epoch"],
            "selection_monitor": e["config"]["early_stopping"]["monitor"],
            "selection_value": exp["best_metric"],
            "val_balanced_accuracy_at_selected_epoch": e["val_bal_acc_at_selected_epoch"],
            "checkpoint": e["evaluation"]["checkpoint"],
            "eval_id": e["evaluation"]["id"],
            "test_evaluations": e["test_evaluations"],
            "bootstrap": {k: e["metrics_doc"]["bootstrap"][k] for k in ("resamples", "seed")},
            split: {
                "accuracy": m["accuracy"],
                "accuracy_ci": ci["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "balanced_accuracy_ci": ci["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "macro_f1_ci": ci["macro_f1"],
            },
        }
        if exp["id"] != reference:
            pred = np.array([index[c] for c in e["y_pred"]])
            row["paired_difference_vs_reference"] = paired_bootstrap_difference(
                y_true, pred, ref_pred, len(class_names), resamples=resamples, seed=seed
            )
        rows.append(row)

    ranked = sorted(
        (r for r in rows if r["val_balanced_accuracy_at_selected_epoch"] is not None),
        key=lambda r: -r["val_balanced_accuracy_at_selected_epoch"],
    )
    report_id = f"rp-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    report = {
        "format_version": REPORT_FORMAT_VERSION,
        "report_id": report_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "visionbench_version": __version__,
        "note": note,
        "split": split,
        "dataset_id": exps[0]["dataset_id"],
        "dataset_fingerprint": exps[0]["dataset_fingerprint"],
        "split_id": exps[0]["split_id"],
        "n_samples": len(y_true),
        "class_names": class_names,
        "reference_experiment": reference,
        "paired_bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "difference": "experiment minus reference",
        },
        "rows": rows,
        "val_ranking": {
            "basis": "validation balanced accuracy at the val-selected epoch (VAL only)",
            "order": [r["experiment_id"] for r in ranked],
            "ranks": competition_ranks(
                {r["experiment_id"]: r["val_balanced_accuracy_at_selected_epoch"] for r in ranked}
            ),
        },
        "confounds": _confounds(entries),
    }
    out_dir = settings.data_dir("reports") / report_id
    out_dir.mkdir(parents=True)
    write_json(out_dir / "comparison.json", report)
    (out_dir / "comparison.md").write_bytes(render_markdown(report).encode("utf-8"))
    report["paths"] = {
        "json": str(out_dir / "comparison.json"),
        "markdown": str(out_dir / "comparison.md"),
    }
    return report


def _f(v: Any, spec: str = ".4f") -> str:
    return "n/a" if v is None else format(v, spec)


def _ci(value: Any, ci: dict[str, Any]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} [{_f(ci.get('low'))}, {_f(ci.get('high'))}]"


def render_markdown(report: dict[str, Any]) -> str:
    split = report["split"]
    lines = [f"# Model comparison ({split} split)", ""]
    if report.get("note"):
        lines += [f"> **{report['note']}**", ""]
    lines += [
        f"- Report: `{report['report_id']}` ({report['created_at']}, visionbench "
        f"{report['visionbench_version']})",
        f"- Dataset: `{report['dataset_id']}` (fingerprint `{report['dataset_fingerprint'][:16]}…`)"
        f", split `{report['split_id']}`, {report['n_samples']} {split} samples, "
        f"classes: {', '.join(report['class_names'])}",
        f"- Reference for paired differences: `{report['reference_experiment']}`",
        "",
    ]
    if split == "test":
        lines += [
            "Test-set metrics are reported for every model **without a winner label**. "
            "Model ranking uses validation metrics only (see *Validation ranking*).",
            "",
        ]
    lines += [
        "## Models",
        "",
        "| experiment | model | weights | input px | params | ckpt bytes | latency bs1 "
        "(median ms, device) | train s | best epoch | val selection | val bal. acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        latency = (
            f"{r['latency_bs1_median_ms']:.3f} ({r['latency_device']})"
            if r["latency_bs1_median_ms"] is not None
            else f"n/a: {r['latency_note']}"
        )
        lines.append(
            f"| `{r['experiment_id']}` | {r['model']} | {r['weights_source']} | "
            f"{r['input_size'] if r['input_size'] is not None else 'n/a'} | "
            f"{r['params_total'] if r['params_total'] is not None else 'n/a'} | "
            f"{r['checkpoint_size_bytes']} | {latency} | {_f(r['train_seconds'], '.1f')} | "
            f"{r['best_epoch']} | {r['selection_monitor']}={_f(r['selection_value'])} | "
            f"{_f(r['val_balanced_accuracy_at_selected_epoch'])} |"
        )
    boots = {(r["bootstrap"]["resamples"], r["bootstrap"]["seed"]) for r in report["rows"]}
    boot_text = ", ".join(f"{n} resamples, seed {sd}" for n, sd in sorted(boots))
    lines += [
        "",
        f"## {split.capitalize()} metrics",
        "",
        f"Point value [95% percentile bootstrap CI] from each evaluation ({boot_text}).",
        "",
        "| experiment | model | checkpoint | accuracy | balanced accuracy | macro F1 | "
        "test evaluations so far |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        s = r[split]
        lines.append(
            f"| `{r['experiment_id']}` | {r['model']} | {r['checkpoint']} | "
            f"{_ci(s['accuracy'], s['accuracy_ci'])} | "
            f"{_ci(s['balanced_accuracy'], s['balanced_accuracy_ci'])} | "
            f"{_ci(s['macro_f1'], s['macro_f1_ci'])} | {r['test_evaluations']} |"
        )
    pb = report["paired_bootstrap"]
    lines += [
        "",
        f"## Paired differences vs reference `{report['reference_experiment']}`",
        "",
        f"Paired percentile bootstrap: {pb['resamples']} resamples, seed {pb['seed']}, the "
        "same resampled indices for both models; difference = experiment − reference.",
        "",
        "| experiment | model | Δ balanced accuracy [95% CI] | Δ macro F1 [95% CI] |",
        "|---|---|---|---|",
    ]
    for r in report["rows"]:
        d = r.get("paired_difference_vs_reference")
        if d is None:
            continue
        ba, f1 = d["metrics"]["balanced_accuracy"], d["metrics"]["macro_f1"]
        lines.append(
            f"| `{r['experiment_id']}` | {r['model']} | {_ci(ba['point'], ba)} | "
            f"{_ci(f1['point'], f1)} |"
        )
    vr = report["val_ranking"]
    by_id = {r["experiment_id"]: r for r in report["rows"]}
    lines += ["", "## Validation ranking", "", f"Basis: {vr['basis']}.", ""]
    tied = Counter(vr["ranks"].values())
    for exp_id in vr["order"]:
        r, rank = by_id[exp_id], vr["ranks"][exp_id]
        label = f"{rank}=" if tied[rank] > 1 else f"{rank}"
        lines.append(
            f"- rank {label}: `{exp_id}` {r['model']}: "
            f"{_f(r['val_balanced_accuracy_at_selected_epoch'])}"
        )
    if any(n > 1 for n in tied.values()):
        lines.append("")
        lines.append("`=` marks tied validation values; tied models are not ordered.")
    lines += ["", "## Confound flags", ""]
    if not report["confounds"]:
        lines.append("None detected.")
    for c in report["confounds"]:
        lines.append(
            f"- **{c['code']}**: {c['message']}. Values: "
            f"`{json.dumps(c['values'], sort_keys=True)}`"
        )
    return "\n".join(lines) + "\n"
