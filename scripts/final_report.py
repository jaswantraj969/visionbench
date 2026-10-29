"""Final report of a frozen benchmark (M8), as pre-registered in ``FINAL_ANALYSIS_PLAN.md``.

Reads recorded evaluations only; no model is run. Writes ``results/FINAL_REPORT.md`` and
``results/FINAL_REPORT.json`` (``--split test``), plus the TEST robustness report
(``results/robustness_test.md/.json``).

- **Gates:** it refuses to run unless the protocol and plan hashes verify.
- **Runs:** each of the 15 runs uses its single clean evaluation of the val-selected checkpoint on
  the chosen split. On TEST that is the one evaluation recorded by
  ``run_benchmark.py evaluate-test``; more than one is an error.
- **CIs:** the per-run CIs are the ones stored by ``evaluate``. The plan requires 1000 resamples,
  seed 0, and this is checked.
- **Seed-matched paired comparisons:** these are paired percentile bootstraps of A - B on the same
  images. They use 1000 resamples, seed 0, and indices shared by the two models.
- **Ranking and rank groups:**
  - Models are ranked by their 3-seed mean balanced accuracy.
  - Two neighbours in that order are in different rank groups only if all three seed-matched
    paired CIs of the balanced-accuracy difference exclude zero in the same direction.
  - Rank groups are descriptive. Only the three pre-registered comparisons are confirmatory.

``--split val --out-dir <dir>`` is a dry run of the same analysis on VALIDATION, labelled as such,
for checking the script before the TEST evaluation. It is never written to ``results/``.

Usage:
    python scripts/final_report.py benchmarks/eurosat-v1 --split test
    python scripts/final_report.py benchmarks/eurosat-v1 --split val --out-dir <scratch dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Any

MODELS = (
    "majority_baseline",
    "resnet18_scratch_64",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
)
SEEDS = (0, 1, 2)
METRICS = ("balanced_accuracy", "macro_f1")
PRIMARY = "balanced_accuracy"
RESAMPLES, BOOT_SEED = 1000, 0
# Pre-registered, seed-matched paired comparisons (A - B), in this order.
COMPARISONS = (
    ("efficientnet_b0_imagenet_224", "resnet18_imagenet_224"),
    ("small_cnn_scratch_64", "resnet18_scratch_64"),
    ("resnet18_imagenet_224", "resnet18_scratch_64"),
)
VAL_LABEL = "VALIDATION (selection-biased) — not test results"
TEST_LABEL = "TEST"
CONSTRAINTS = (
    (
        "epoch_cap",
        "Every trainable run had a fixed epoch cap (25) and early stopping on validation loss; "
        "several runs stopped at the cap. The numbers describe these training budgets, not each "
        "architecture's best achievable accuracy.",
    ),
    (
        "untuned_learning_rates",
        "Learning rates and other hyperparameters were fixed in the protocol, not tuned per "
        "model. A different setting could change any ranking.",
    ),
    (
        "single_split",
        "One fixed train/val/test split of one dataset: no split-to-split variance is included, "
        "and nothing here generalises beyond EuroSAT RGB.",
    ),
    (
        "three_seeds",
        "n = 3 seeds per model: the seed sd is itself very uncertain, and the seed-matched "
        "comparison rule is strict by design.",
    ),
    (
        "low_information_classes",
        "Many SeaLake (about 78 %) and Forest (about 34 %) tiles are near-uniform; their "
        "near-duplicates cannot be detected by hashing, so leakage for those classes is bounded "
        "less tightly (DATASETS.md).",
    ),
    (
        "resolution_pretraining_confound",
        "The two ImageNet-pretrained models use 224 px inputs (upsampled from 64 px) and the "
        "scratch models 64 px: pretraining and input resolution are confounded, so a difference "
        "between the groups has no attributed cause.",
    ),
    (
        "corruption_scaling",
        "Corruptions are applied at 64 px before each model's transform; for the 224 px models "
        "the corruption is upsampled with the content (a 1 px blur becomes about 4 px; robustness "
        "protocol amendment 1). This scales corruption and content together; it is not a "
        "smoothing of the corruption relative to the content.",
    ),
    (
        "no_robustness_claim",
        "The robustness numbers describe sensitivity to 4 synthetic corruptions at 3 fixed "
        "severities only. No model is called robust.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(dir_: Path, hash_file: str) -> dict[str, str]:
    out = {}
    for line in (dir_ / hash_file).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            actual = sha256(dir_ / name)
            if actual != digest:
                raise SystemExit(f"{name}: sha256 {actual} != {hash_file}; refusing to run")
            out[name] = digest
    return out


def ms(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "per_seed": values,
    }


def fmt_ms(d: dict[str, Any]) -> str:
    return f"{d['mean']:.4f} ± {d['sd']:.4f}" if d["sd"] is not None else f"{d['mean']:.4f}"


def fmt_ci(d: dict[str, Any]) -> str:
    return f"{d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"


def direction(cis: list[dict[str, Any]]) -> int:
    """+1 / -1 if every CI excludes zero on that side, else 0."""
    if all(c["low"] is not None and c["low"] > 0 for c in cis):
        return 1
    if all(c["high"] is not None and c["high"] < 0 for c in cis):
        return -1
    return 0


def read_predictions(path: Path) -> dict[str, tuple[str, str]]:
    rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    return {r["relpath"]: (r["true_label"], r["pred_label"]) for r in rows}


def load_runs(protocol_dir: Path, split: str, settings: Any, store: Any) -> list[dict[str, Any]]:
    from visionbench.core.jsonio import read_json
    from visionbench.robustness.report import single_test_evaluation

    runs = json.loads((protocol_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    final = {(r["model"], r["seed"]): r for r in runs}
    root = settings.resolved_data_root
    out = []
    for model in MODELS:
        for seed in SEEDS:
            r = final[(model, seed)]
            exp = r["experiment_id"]
            eval_id = r["val_eval_id"] if split == "val" else single_test_evaluation(store, exp)
            ev = store.get_evaluation(eval_id)
            metrics_path = root / ev["metrics_path"]
            doc = read_json(metrics_path)
            boot = doc["bootstrap"]
            if (boot["resamples"], boot["seed"]) != (RESAMPLES, BOOT_SEED):
                raise SystemExit(
                    f"{eval_id}: CIs use {boot['resamples']}/{boot['seed']}, not 1000/0"
                )
            if doc["split"] != split or doc["checkpoint"]["name"] not in ("best", "baseline"):
                raise SystemExit(
                    f"{eval_id} is not a {split} evaluation of the selected checkpoint"
                )
            pred_path = metrics_path.parent / "predictions.csv"
            if sha256(pred_path) != ev["predictions_sha256"]:
                raise SystemExit(f"{eval_id}: predictions do not match their recorded sha256")
            val_doc = read_json(root / store.get_evaluation(r["val_eval_id"])["metrics_path"])
            out.append(
                {
                    "model": model,
                    "seed": seed,
                    "experiment_id": exp,
                    "evaluation_id": eval_id,
                    "val_evaluation_id": r["val_eval_id"],
                    "n": doc["metrics"]["n"],
                    "class_names": doc["metrics"]["class_names"],
                    **{m: doc["metrics"][m] for m in (*METRICS, "accuracy")},
                    "ci": {m: doc["bootstrap"]["metrics"][m] for m in METRICS},
                    "val": {m: val_doc["metrics"][m] for m in METRICS},
                    "predictions": read_predictions(pred_path),
                }
            )
    return out


def paired(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    from visionbench.evaluation.metrics import paired_bootstrap_difference

    if a["class_names"] != b["class_names"]:
        raise SystemExit(f"{a['evaluation_id']} and {b['evaluation_id']} have different classes")
    if set(a["predictions"]) != set(b["predictions"]):
        raise SystemExit(f"{a['evaluation_id']} and {b['evaluation_id']} cover different images")
    index = {c: i for i, c in enumerate(a["class_names"])}
    keys = sorted(a["predictions"])
    y_true = [index[a["predictions"][k][0]] for k in keys]
    if y_true != [index[b["predictions"][k][0]] for k in keys]:
        raise SystemExit("true labels differ between paired evaluations")
    pa = [index[a["predictions"][k][1]] for k in keys]
    pb = [index[b["predictions"][k][1]] for k in keys]
    d = paired_bootstrap_difference(
        y_true, pa, pb, len(index), metrics=METRICS, resamples=RESAMPLES, seed=BOOT_SEED
    )
    return d["metrics"]


def seed_matched(by: dict[tuple[str, int], dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    per_seed = {s: paired(by[(a, s)], by[(b, s)]) for s in SEEDS}
    out: dict[str, Any] = {"a": a, "b": b, "per_seed": per_seed, "verdict": {}}
    for m in METRICS:
        dirn = direction([per_seed[s][m] for s in SEEDS])
        out["verdict"][m] = (
            f"distinguishable: {a} > {b}"
            if dirn > 0
            else f"distinguishable: {a} < {b}"
            if dirn < 0
            else "not distinguishable"
        )
    return out


def analyse(runs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by = {(r["model"], r["seed"]): r for r in runs}
    models: dict[str, Any] = {}
    for m in MODELS:
        rs = [by[(m, s)] for s in SEEDS]
        models[m] = {
            **{k: ms([r[k] for r in rs]) for k in METRICS},
            "val": {k: ms([r["val"][k] for r in rs]) for k in METRICS},
        }
    order = sorted(MODELS, key=lambda m: -models[m][PRIMARY]["mean"])
    groups, neighbours, g = {order[0]: 1}, [], 1
    for hi, lo in zip(order, order[1:], strict=False):
        cmp = seed_matched(by, hi, lo)
        apart = direction([cmp["per_seed"][s][PRIMARY] for s in SEEDS]) > 0
        neighbours.append({"higher": hi, "lower": lo, "separated": apart, **cmp})
        g += 1 if apart else 0
        groups[lo] = g
    ranks: dict[str, int] = {}
    for i, m in enumerate(order):  # competition ranks on the mean
        prev = order[i - 1] if i else None
        same = prev is not None and models[m][PRIMARY]["mean"] == models[prev][PRIMARY]["mean"]
        ranks[m] = ranks[prev] if same and prev is not None else i + 1
    ranking = [{"model": m, "rank": ranks[m], "rank_group": groups[m]} for m in order]
    comparisons = [seed_matched(by, a, b) for a, b in COMPARISONS]
    return {
        "split": split,
        "models": models,
        "ranking": ranking,
        "neighbours": neighbours,
        "comparisons": comparisons,
    }


def render(doc: dict[str, Any], runs: list[dict[str, Any]], rob: dict[str, Any] | None) -> str:
    split, label = doc["split"], doc["label"]
    other = "VALIDATION" if split == "test" else "VALIDATION (again)"
    title = "Final report" if split == "test" else "DRY RUN of the final report on VALIDATION"
    lines = [
        f"# {title}: {doc['benchmark']} — {label}",
        "",
        f"**{label}.** "
        + (
            "One clean TEST evaluation per run (15 runs), following "
            f"`FINAL_ANALYSIS_PLAN.md` (sha256 `{doc['final_analysis_plan_sha256']}`). The "
            "checkpoints were chosen on VALIDATION before any TEST access."
            if split == "test"
            else "This dry run checks the analysis code on VALIDATION numbers; they are "
            "selection-biased and are NOT test results."
        ),
        "",
        "Balanced accuracy is primary, macro F1 secondary. Per-run CIs: 95 % percentile bootstrap, "
        "1000 resamples, seed 0. Mean ± sample sd over seeds 0/1/2 (the split is fixed, so the sd "
        "reflects training variance only).",
        "",
        "## Per model (mean ± sd over 3 seeds)",
        "",
        f"| model | balanced accuracy ({label}) | macro F1 ({label}) | "
        f"per-seed balanced accuracy ({label}) |",
        "|---|---|---|---|",
    ]
    for m in MODELS:
        d = doc["models"][m]
        per = " / ".join(f"{v:.4f}" for v in d[PRIMARY]["per_seed"])
        lines.append(f"| {m} | {fmt_ms(d[PRIMARY])} | {fmt_ms(d['macro_f1'])} | {per} |")
    lines += [
        "",
        "## Per run",
        "",
        f"| model | seed | experiment | evaluation | n | balanced accuracy [95 % CI] ({label}) | "
        f"macro F1 [95 % CI] ({label}) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        cb, cf = r["ci"][PRIMARY], r["ci"]["macro_f1"]
        lines.append(
            f"| {r['model']} | {r['seed']} | {r['experiment_id']} | {r['evaluation_id']} | "
            f"{r['n']} | {r[PRIMARY]:.4f} [{cb['low']:.4f}, {cb['high']:.4f}] | "
            f"{r['macro_f1']:.4f} [{cf['low']:.4f}, {cf['high']:.4f}] |"
        )
    lines += [
        "",
        "## Ranking by 3-seed mean balanced accuracy, with rank groups",
        "",
        "Neighbours are in different rank groups only if all three seed-matched paired CIs of the "
        "balanced-accuracy difference exclude zero in the same direction. Rank groups are "
        "descriptive; only the three pre-registered comparisons below are confirmatory.",
        "",
        "| rank | model | mean balanced accuracy | rank group |",
        "|---|---|---|---|",
    ]
    for x in doc["ranking"]:
        mean = doc["models"][x["model"]][PRIMARY]["mean"]
        lines.append(f"| {x['rank']} | {x['model']} | {mean:.4f} | {x['rank_group']} |")
    lines += [
        "",
        "| higher | lower | Δ seed 0 | Δ seed 1 | Δ seed 2 | separated |",
        "|---|---|---|---|---|---|",
    ]
    for nb in doc["neighbours"]:
        cells = " | ".join(fmt_ci(nb["per_seed"][s][PRIMARY]) for s in SEEDS)
        lines.append(
            f"| {nb['higher']} | {nb['lower']} | {cells} | {'yes' if nb['separated'] else 'no'} |"
        )
    lines += [
        "",
        "## Pre-registered comparisons (seed-matched, A - B)",
        "",
        "Paired percentile bootstrap of A - B on the same images (1000 resamples, seed 0, shared "
        "indices), one per seed. **Distinguishable** only if all three CIs exclude zero in the "
        "same direction.",
        "",
        "| # | A - B | metric | Δ seed 0 | Δ seed 1 | Δ seed 2 | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(doc["comparisons"], 1):
        for m in METRICS:
            cells = " | ".join(fmt_ci(c["per_seed"][s][m]) for s in SEEDS)
            lines.append(f"| {i} | {c['a']} - {c['b']} | {m} | {cells} | {c['verdict'][m]} |")
    lines += [
        "",
        f"## {label} next to VALIDATION (selection-biased)",
        "",
        f"| model | balanced accuracy: {other} | balanced accuracy: {label} | "
        f"macro F1: {other} | macro F1: {label} |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        d = doc["models"][m]
        lines.append(
            f"| {m} | {fmt_ms(d['val'][PRIMARY])} | {fmt_ms(d[PRIMARY])} | "
            f"{fmt_ms(d['val']['macro_f1'])} | {fmt_ms(d['macro_f1'])} |"
        )
    if rob is not None:
        lines += [
            "",
            f"## Robustness ({rob['suite']}) on {split.upper()} — {rob['label']}",
            "",
            f"Full tables: `results/robustness_{split}.md`. Mean balanced accuracy over the 12 "
            "conditions; drop = clean - corrupted (mean ± sd over seeds).",
            "",
            "| model | clean | mean over 12 conditions | drop | relative drop |",
            "|---|---|---|---|---|",
        ]
        for m in MODELS:
            s = rob["summary"].get(m)
            if s is None:
                lines.append(f"| {m} | incomplete | | | |")
                continue
            lines.append(
                f"| {m} | {fmt_ms(ms(s['clean']))} | {fmt_ms(ms(s['mean_corrupted']))} | "
                f"{fmt_ms(ms(s['drop']))} | {fmt_ms(ms(s['relative_drop']))} |"
            )
        if rob["missing"]:
            lines += ["", f"**Missing robustness evaluations: {len(rob['missing'])}.**"]
    lines += ["", "## Interpretation constraints (fixed in the plan)", ""]
    lines += [f"- **{code}**: {text}" for code, text in CONSTRAINTS]
    lines += [
        "",
        "No model is called robust or state-of-the-art. A difference is called distinguishable "
        "only under the rule above.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out-dir", type=Path, help="val dry run only; default for test: results/")
    args = ap.parse_args()
    from visionbench.core.config import Settings
    from visionbench.robustness.report import write_report
    from visionbench.robustness.suite import load_suite
    from visionbench.tracking.store import DB_FILENAME, ExperimentStore

    d = args.protocol_dir
    if args.split == "val":
        if args.out_dir is None or args.out_dir.resolve().is_relative_to((d / "results").resolve()):
            raise SystemExit("the VALIDATION dry run needs --out-dir outside results/")
        out_dir = args.out_dir
    else:
        if args.out_dir is not None:
            raise SystemExit("the TEST report is always written to results/")
        out_dir = d / "results"
    protocol = verify(d, "PROTOCOL_HASHES.txt")
    verify(d / "robustness", "ROBUSTNESS_HASHES.txt")
    plan = verify(d, "FINAL_ANALYSIS_PLAN_HASHES.txt")
    if "FINAL_ANALYSIS_PLAN.md" not in plan:
        raise SystemExit("FINAL_ANALYSIS_PLAN_HASHES.txt does not cover FINAL_ANALYSIS_PLAN.md")
    settings = Settings()
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        runs = load_runs(d, args.split, settings, store)
    doc: dict[str, Any] = {
        "benchmark": d.name,
        "label": TEST_LABEL if args.split == "test" else VAL_LABEL,
        "final_analysis_plan_sha256": plan["FINAL_ANALYSIS_PLAN.md"],
        "protocol_hashes": protocol,
        "bootstrap": {"resamples": RESAMPLES, "seed": BOOT_SEED, "level": 0.95},
        **analyse(runs, args.split),
        "constraints": dict(CONSTRAINTS),
    }
    suite = load_suite(d / "robustness" / "robustness_suite.yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    rob = write_report(
        settings,
        d / "runs.json",
        suite,
        args.split,
        out_dir / f"robustness_{args.split}.md",
    )
    doc["robustness"] = {
        "suite": rob["suite"],
        "suite_sha256": rob["suite_sha256"],
        "label": rob["label"],
        "summary": rob["summary"],
        "missing": rob["missing"],
    }
    doc["runs"] = [{k: v for k, v in r.items() if k != "predictions"} for r in runs]
    text = render(doc, runs, rob)
    md = out_dir / ("FINAL_REPORT.md" if args.split == "test" else "FINAL_REPORT_VAL_DRY_RUN.md")
    md.write_text(text, encoding="utf-8", newline="\n")
    md.with_suffix(".json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {md} and {md.with_suffix('.json')} ({doc['label']})")
    for c in doc["comparisons"]:
        print(f"  {c['a']} - {c['b']}: {c['verdict'][PRIMARY]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
