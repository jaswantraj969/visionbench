"""Robustness report over the runs of a benchmark (``runs.json``): Markdown + JSON.

Reads recorded results only (no model is run). Every number is labelled
"VALIDATION (selection-biased) — not test results" for split=val. No model is called robust; a
difference is called distinguishable from zero only where its paired bootstrap CI excludes zero.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json
from visionbench.robustness.suite import RobustnessError, Suite, TestSplitLockedError
from visionbench.tracking.store import DB_FILENAME, ExperimentStore

VAL_LABEL = "VALIDATION (selection-biased) — not test results"
LOW_INFO = {"SeaLake": "78 % low-information", "Forest": "34 % low-information"}
SCOPE = (
    "Scope: synthetic corruptions at 3 fixed severities on one dataset (EuroSAT RGB, one split). "
    "The numbers describe sensitivity to THESE corruptions only; they do not show general "
    "robustness, robustness to real sensor degradation, or robustness to distribution shift. "
    "No model is called robust."
)
CONFOUNDS = [
    (
        "resolution_upsampling",
        "Corruptions are applied at 64 px. Pretrained models see an upsampled version (resize 256, "
        "centre crop 224) in which corruption and content are scaled together (a blur of 1 px "
        "becomes about 4 px); scratch models see both at native resolution. This is not a "
        "smoothing of the corruption relative to the content (robustness protocol, amendment 1); "
        "pretraining and resolution stay confounded, so group differences have no attributed "
        "cause.",
    ),
    (
        "pretrained_vs_scratch",
        "ImageNet pretraining and input resolution are confounded with each other.",
    ),
    ("centre_crop", "Pretrained models never see the 4 px border of a tile."),
    ("single_split", "One fixed split: no split-to-split variance."),
    ("three_seeds", "n = 3 seeds: the sd is itself uncertain; per-seed values are shown."),
    (
        "selection_bias",
        "VAL chose each checkpoint; clean and corrupted VAL numbers share that bias.",
    ),
]


def _ms(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.4f}" if values else "-"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def _ci(d: dict[str, Any]) -> str:
    return f"{d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"


def _excludes_zero(d: dict[str, Any]) -> bool:
    return d["low"] > 0 or d["high"] < 0


def collect(settings: Settings, runs_path: Path, suite: Suite, split: str) -> dict[str, Any]:
    if split == "test" and not settings.test_images_unlocked:
        raise TestSplitLockedError("robustness on the TEST split is locked until M8")
    runs = json.loads(runs_path.read_text(encoding="utf-8"))["runs"]
    final = {}
    for r in runs:  # last attempt per (model, seed)
        final[(r["model"], r["seed"])] = r
    data_root = settings.resolved_data_root
    out: dict[str, Any] = {"suite": suite.name, "suite_sha256": suite.sha256, "split": split}
    rows = []
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        for (model, seed), r in sorted(final.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if r["status"] != "completed":
                continue
            if split == "val":
                clean_id = r["val_eval_id"]
            else:
                raise RobustnessError("test reports are produced at M8")
            clean = read_json(data_root / store.get_evaluation(clean_id)["metrics_path"])
            recorded = {
                (x["corruption"], x["severity"]): x
                for x in store.robustness_evaluations(
                    r["experiment_id"], split=split, suite_sha256=suite.sha256
                )
            }
            conds = {}
            for c in suite.conditions:
                x = recorded.get((c.corruption, c.severity_str))
                if x is None:
                    conds[c.key] = None
                    continue
                if x["clean_evaluation_id"] != clean_id:
                    raise RobustnessError(f"{x['id']} is paired with another clean evaluation")
                m = read_json(data_root / x["metrics_path"])
                conds[c.key] = {
                    "robustness_id": x["id"],
                    "condition_hash": x["condition_hash"],
                    "balanced_accuracy": m["metrics"]["balanced_accuracy"],
                    "macro_f1": m["metrics"]["macro_f1"],
                    "accuracy": m["metrics"]["accuracy"],
                    "bootstrap": {
                        k: m["bootstrap"]["metrics"][k] for k in ("balanced_accuracy", "macro_f1")
                    },
                    "paired": m["paired_difference_corrupted_minus_clean"]["metrics"],
                    "per_class_recall": {
                        cls: v["recall"] for cls, v in m["metrics"]["per_class"].items()
                    },
                }
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "experiment_id": r["experiment_id"],
                    "clean_evaluation_id": clean_id,
                    "clean_balanced_accuracy": clean["metrics"]["balanced_accuracy"],
                    "clean_macro_f1": clean["metrics"]["macro_f1"],
                    "clean_per_class_recall": {
                        cls: v["recall"] for cls, v in clean["metrics"]["per_class"].items()
                    },
                    "conditions": conds,
                }
            )
    out["runs"] = rows
    return out


def _aggregates(run: dict[str, Any], suite: Suite) -> dict[str, Any]:
    clean = run["clean_balanced_accuracy"]
    vals = {k: v["balanced_accuracy"] for k, v in run["conditions"].items() if v is not None}
    corruptions = sorted({c.corruption for c in suite.conditions})
    per_corruption = {}
    for name in corruptions:
        keys = [c.key for c in suite.conditions if c.corruption == name]
        if all(k in vals for k in keys):
            per_corruption[name] = statistics.mean(vals[k] for k in keys)
    complete = len(vals) == len(suite.conditions)
    mean_all = statistics.mean(vals.values()) if complete else None
    return {
        "complete": complete,
        "mean_corrupted_balanced_accuracy": mean_all,
        "mean_per_corruption": per_corruption,
        "relative_drop": None if mean_all is None else (clean - mean_all) / clean,
    }


def build_report(data: dict[str, Any], suite: Suite) -> tuple[str, dict[str, Any]]:
    label = VAL_LABEL if data["split"] == "val" else "TEST"
    runs = data["runs"]
    for r in runs:
        r["aggregates"] = _aggregates(r, suite)
    models = list(dict.fromkeys(r["model"] for r in runs))
    by_model = {
        m: sorted((r for r in runs if r["model"] == m), key=lambda r: r["seed"]) for m in models
    }
    keys = [c.key for c in suite.conditions]
    missing = [
        f"{r['model']} seed {r['seed']}: {k}"
        for r in runs
        for k in keys
        if r["conditions"][k] is None
    ]
    lines = [
        f"# Robustness ({suite.name}) on {data['split'].upper()} — {label}",
        "",
        f"**{label}.** {SCOPE}",
        "",
        f"Suite `{suite.name}` (robustness_suite.yaml sha256 `{suite.sha256}`), 12 conditions, "
        "applied to the 64x64 originals before each model's val transform. Clean numbers are the "
        "pinned B1 clean VAL evaluations; corrupted and clean predictions cover the same images. "
        "Drop = clean - corrupted balanced accuracy; relative drop = drop / clean. No reference-"
        "model normalization (this is not ImageNet-C's mCE). Mean ± sample sd over seeds 0/1/2.",
        "",
    ]
    if missing:
        lines += ["**Missing evaluations:** " + "; ".join(missing), ""]
    lines += [
        "## Summary per model (balanced accuracy)",
        "",
        "| model | clean (mean ± sd) | mean over 12 conditions (mean ± sd) | drop (mean ± sd) | "
        "relative drop (mean ± sd) | per seed: clean / mean corrupted |",
        "|---|---|---|---|---|---|",
    ]
    summary = {}
    for m in models:
        rs = by_model[m]
        clean = [r["clean_balanced_accuracy"] for r in rs]
        if not all(r["aggregates"]["complete"] for r in rs):
            lines.append(f"| {m} | {_ms(clean)} | incomplete | | | |")
            continue
        corr = [r["aggregates"]["mean_corrupted_balanced_accuracy"] for r in rs]
        drop = [c - x for c, x in zip(clean, corr, strict=True)]
        rel = [r["aggregates"]["relative_drop"] for r in rs]
        seeds = "; ".join(
            f"s{r['seed']} {c:.4f} / {x:.4f}" for r, c, x in zip(rs, clean, corr, strict=True)
        )
        lines.append(f"| {m} | {_ms(clean)} | {_ms(corr)} | {_ms(drop)} | {_ms(rel)} | {seeds} |")
        summary[m] = {"clean": clean, "mean_corrupted": corr, "drop": drop, "relative_drop": rel}
    corruptions = list(dict.fromkeys(c.corruption for c in suite.conditions))
    lines += [
        "",
        "## Mean corrupted balanced accuracy per corruption "
        "(over its 3 severities; mean ± sd over seeds)",
        "",
        "| model | " + " | ".join(corruptions) + " |",
        "|---|" + "---|" * len(corruptions),
    ]
    for m in models:
        cells = []
        for name in corruptions:
            vals = [r["aggregates"]["mean_per_corruption"].get(name) for r in by_model[m]]
            cells.append(_ms(vals) if all(v is not None for v in vals) else "-")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per condition: corrupted balanced accuracy and drop (mean ± sd over seeds)",
        "",
        "| model | condition | corrupted bal. acc. | drop | per seed corrupted (s0 / s1 / s2) |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        rs = by_model[m]
        for k in keys:
            if any(r["conditions"][k] is None for r in rs):
                lines.append(f"| {m} | {k} | missing | | |")
                continue
            corr = [r["conditions"][k]["balanced_accuracy"] for r in rs]
            drop = [r["clean_balanced_accuracy"] - c for r, c in zip(rs, corr, strict=True)]
            lines.append(
                f"| {m} | {k} | {_ms(corr)} | {_ms(drop)} | "
                + " / ".join(f"{c:.4f}" for c in corr)
                + " |"
            )
    lines += [
        "",
        "## Seed 0: paired difference corrupted - clean [95 % paired bootstrap CI]",
        "",
        "Paired percentile bootstrap, 1000 resamples, seed 0, the same image indices for clean and "
        "corrupted predictions. ✓ = the CI excludes zero (the change is distinguishable from zero "
        "for this run, for this corruption only).",
        "",
        "| model | condition | Δ balanced accuracy | Δ macro F1 | CI excludes 0 (bal. acc.) |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        r0 = next((r for r in by_model[m] if r["seed"] == 0), None)
        if r0 is None:
            continue
        for k in keys:
            c = r0["conditions"][k]
            if c is None:
                continue
            pb, pf = c["paired"]["balanced_accuracy"], c["paired"]["macro_f1"]
            lines.append(
                f"| {m} | {k} | {_ci(pb)} | {_ci(pf)} | {'✓' if _excludes_zero(pb) else '—'} |"
            )
    lines += [
        "",
        "## Seed 0: per-class recall in the worst condition per model",
        "",
        "Worst condition = lowest corrupted balanced accuracy of the seed-0 run. Classes marked † "
        "are largely low-information (near-uniform tiles; DATASETS.md).",
        "",
    ]
    worst_out = {}
    for m in models:
        r0 = next((r for r in by_model[m] if r["seed"] == 0), None)
        if r0 is None or not r0["aggregates"]["complete"]:
            continue
        k = min(keys, key=lambda key: r0["conditions"][key]["balanced_accuracy"])
        c = r0["conditions"][k]
        lines += [
            f"**{m}** (seed 0, {r0['experiment_id']}): worst condition `{k}`, balanced accuracy "
            f"{r0['clean_balanced_accuracy']:.4f} clean → {c['balanced_accuracy']:.4f}.",
            "",
            "| class | clean recall | corrupted recall | drop |",
            "|---|---|---|---|",
        ]
        rows_w = {}
        for cls, clean_r in r0["clean_per_class_recall"].items():
            cor = c["per_class_recall"][cls]
            mark = f" † ({LOW_INFO[cls]})" if cls in LOW_INFO else ""
            if clean_r is None or cor is None:
                lines.append(f"| {cls}{mark} | {clean_r} | {cor} | undefined |")
                continue
            lines.append(f"| {cls}{mark} | {clean_r:.4f} | {cor:.4f} | {clean_r - cor:+.4f} |")
            rows_w[cls] = {"clean": clean_r, "corrupted": cor}
        lines.append("")
        worst_out[m] = {"condition": k, "per_class_recall": rows_w}
    baseline = [r for r in runs if r["model"] == "majority_baseline"]
    if baseline:
        exact = all(
            c is not None
            and c["balanced_accuracy"] == r["clean_balanced_accuracy"]
            and c["macro_f1"] == r["clean_macro_f1"]
            for r in baseline
            for c in r["conditions"].values()
        )
        pairs = len(baseline) * len(keys)
        verdict = f"equal for all {pairs} run x condition pairs" if exact else "NOT EQUAL"
        lines += [
            "## Pipeline check: majority baseline",
            "",
            "The majority baseline does not look at the image, so every corrupted metric "
            f"must equal its clean metric exactly: **{verdict}**.",
            "",
        ]
    lines += ["## Confounds", ""]
    lines += [f"- **{code}**: {text}" for code, text in CONFOUNDS]
    lines += [
        "",
        "## Runs",
        "",
        "| model | seed | experiment | clean evaluation (pinned) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['model']} | {r['seed']} | {r['experiment_id']} | {r['clean_evaluation_id']} |"
        for r in runs
    ]
    doc = {
        "label": label,
        "scope": SCOPE,
        "confounds": dict(CONFOUNDS),
        "suite": suite.name,
        "suite_sha256": suite.sha256,
        "split": data["split"],
        "summary": summary,
        "worst_condition_seed0": worst_out,
        "missing": missing,
        "runs": runs,
    }
    return "\n".join(lines) + "\n", doc


def write_report(
    settings: Settings, runs_path: Path, suite: Suite, split: str, out_md: Path
) -> dict[str, Any]:
    text, doc = build_report(collect(settings, runs_path, suite, split), suite)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8", newline="\n")
    out_md.with_suffix(".json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return doc
