"""Run a frozen benchmark protocol through the ``visionbench`` CLI and record ``runs.json``.

Subcommands (all take the protocol folder, e.g. ``benchmarks/eurosat-v1``):

- ``train``: for every protocol model x seed 0/1/2, in protocol order, run
  ``visionbench train --config <yaml> --seed <s>``. Before anything runs, the YAMLs and
  BENCHMARK_PROTOCOL.md are checked against ``PROTOCOL_HASHES.txt``. Runs that are already
  completed in ``runs.json`` are skipped. A failed run is retried at most once with the identical
  config, and both attempts are recorded.
- ``evaluate-val``: evaluate each completed run's val-selected checkpoint on **VAL** (never test).
- ``latency``: CPU and CUDA latency for seed 0 of each trainable model.
- ``check``: assert that no evaluation with split=test exists for any benchmark or pilot
  experiment.
- ``summary``: multi-seed VALIDATION summary (Markdown) built from the recorded val evaluations.

``runs.json`` holds experiment ids, statuses, timestamps and evaluation/latency ids only; logs go
to ``<data_root>/benchmarks/<name>/logs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS = (
    "majority_baseline",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
    "resnet18_scratch_64",
)
TRAINABLE = MODELS[1:]
SEEDS = (0, 1, 2)
LOW_INFO_CLASSES = {"SeaLake": "78 % low-information", "Forest": "34 % low-information"}
VAL_LABEL = "VALIDATION (selection-biased) — not test results"
EXP_RE = re.compile(r"experiment (exp-\d{8}-\d{6}-[0-9a-f]{6}) running")
EVAL_RE = re.compile(r"\b(ev-\d{8}-\d{6}-[0-9a-f]{6})\b")
LAT_RE = re.compile(r"\b(lat-\d{8}-\d{6}-[0-9a-f]{6})\b")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Benchmark:
    def __init__(self, protocol_dir: Path) -> None:
        from visionbench.core.config import Settings

        self.dir = protocol_dir
        self.name = protocol_dir.name
        self.settings = Settings()
        self.runs_path = protocol_dir / "runs.json"
        self.logs = self.settings.resolved_data_root / "benchmarks" / self.name / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        if self.runs_path.is_file():
            self.data = json.loads(self.runs_path.read_text(encoding="utf-8"))
        else:
            self.data = {"benchmark": self.name, "runs": []}

    # --- bookkeeping ------------------------------------------------------------------------

    def save(self) -> None:
        tmp = self.runs_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(self.runs_path)

    def verify_hashes(self) -> dict[str, str]:
        expected: dict[str, str] = {}
        for line in (self.dir / "PROTOCOL_HASHES.txt").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                digest, name = line.split(maxsplit=1)
                expected[name.lstrip("*")] = digest
        for name, digest in expected.items():
            actual = sha256(self.dir / name)
            if actual != digest:
                raise SystemExit(f"{name}: sha256 {actual} != frozen {digest}; amend the protocol")
        return expected

    def final_runs(self) -> dict[tuple[str, int], dict[str, Any]]:
        """The last attempt of each (model, seed)."""
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for r in self.data["runs"]:
            out[(r["model"], r["seed"])] = r
        return out

    def cli(self, args: list[str], log: Path) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, "-m", "visionbench.cli", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = f"> visionbench {' '.join(args)}\n{proc.stdout}{proc.stderr}"
        log.write_text(text, encoding="utf-8")
        return proc.returncode, proc.stdout + proc.stderr

    def store(self):  # noqa: ANN201
        from visionbench.tracking.store import DB_FILENAME, ExperimentStore

        return ExperimentStore(self.settings.data_dir("db") / DB_FILENAME)

    # --- subcommands ------------------------------------------------------------------------

    def train(self) -> None:
        hashes = self.verify_hashes()
        done = {k for k, r in self.final_runs().items() if r["status"] == "completed"}
        for model in MODELS:
            yaml_path = self.dir / f"{model}.yaml"
            for seed in SEEDS:
                if (model, seed) in done:
                    print(f"skip {model} seed {seed}: completed already", flush=True)
                    continue
                for attempt in (1, 2):
                    rec = self._train_one(model, yaml_path, seed, attempt, hashes)
                    if rec["status"] == "completed":
                        break
                    print(f"  {model} seed {seed} attempt {attempt}: {rec['status']}", flush=True)

    def _train_one(
        self, model: str, yaml_path: Path, seed: int, attempt: int, hashes: dict[str, str]
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "model": model,
            "config": yaml_path.name,
            "config_sha256": hashes[yaml_path.name],
            "seed": seed,
            "attempt": attempt,
            "started_at": now(),
        }
        print(f"[{rec['started_at']}] train {model} seed {seed} (attempt {attempt})", flush=True)
        log = self.logs / f"train_{model}_seed{seed}_attempt{attempt}.log"
        code, out = self.cli(["train", "--config", str(yaml_path), "--seed", str(seed)], log)
        rec["finished_at"] = now()
        m = EXP_RE.search(out)
        rec["experiment_id"] = m.group(1) if m else None
        if rec["experiment_id"]:
            with self.store() as es:
                exp = es.get(rec["experiment_id"])
            rec["status"] = exp["status"]
            rec["error"] = exp["error"]
        else:
            rec["status"], rec["error"] = "failed", f"exit code {code}; no experiment created"
        rec["exit_code"] = code
        self.data["runs"].append(rec)
        self.save()
        print(f"  -> {rec['experiment_id']} {rec['status']}", flush=True)
        return rec

    def evaluate_val(self) -> None:
        for (model, seed), r in sorted(self.final_runs().items()):
            if r["status"] != "completed" or r.get("val_eval_id"):
                continue
            device = "cpu" if model == "majority_baseline" else "cuda"
            log = self.logs / f"eval_val_{model}_seed{seed}.log"
            print(f"[{now()}] evaluate {model} seed {seed} on VAL", flush=True)
            code, out = self.cli(
                ["evaluate", r["experiment_id"], "--split", "val", "--device", device], log
            )
            m = EVAL_RE.search(out)
            if code != 0 or m is None:
                raise SystemExit(f"val evaluation failed for {r['experiment_id']}:\n{out}")
            r["val_eval_id"] = m.group(1)
            self.save()

    def latency(self) -> None:
        for (model, seed), r in sorted(self.final_runs().items()):
            if model not in TRAINABLE or seed != 0 or r["status"] != "completed":
                continue
            ids = r.setdefault("latency_ids", {})
            for device in ("cpu", "cuda"):
                if device in ids:
                    continue
                log = self.logs / f"latency_{model}_seed{seed}_{device}.log"
                print(f"[{now()}] latency {model} seed {seed} on {device}", flush=True)
                code, out = self.cli(["latency", r["experiment_id"], "--device", device], log)
                m = LAT_RE.search(out)
                if code != 0 or m is None:
                    raise SystemExit(f"latency failed for {r['experiment_id']}:\n{out}")
                ids[device] = m.group(1)
                self.save()

    def check(self) -> None:
        bench = {r["experiment_id"] for r in self.data["runs"] if r.get("experiment_id")}
        pilot_path = self.dir / "pilot.json"
        pilot = set()
        if pilot_path.is_file():
            pilot = set(json.loads(pilot_path.read_text(encoding="utf-8"))["pilot_experiment_ids"])
        with self.store() as es:
            test_evals = [
                e for e in es.evaluations(split="test") if e["experiment_id"] in bench | pilot
            ]
            all_test = es.evaluations(split="test")
            val_evals = [e for e in es.evaluations(split="val") if e["experiment_id"] in bench]
        print(f"benchmark experiments: {len(bench)}; pilot experiments: {len(pilot)}")
        print(f"val evaluations of benchmark experiments: {len(val_evals)}")
        print(f"test evaluations of benchmark or pilot experiments: {len(test_evals)}")
        print(f"test evaluations in the whole database: {len(all_test)}")
        assert not test_evals, f"TEST evaluations exist: {[e['id'] for e in test_evals]}"
        print("OK: no test evaluation exists for any benchmark or pilot experiment")

    def summary(self) -> str:
        from visionbench.core.jsonio import read_json

        root = self.settings.resolved_data_root
        rows: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
        with self.store() as es:
            for (model, seed), r in sorted(self.final_runs().items()):
                if r["status"] != "completed" or not r.get("val_eval_id"):
                    continue
                ev = next(
                    e for e in es.evaluations(r["experiment_id"]) if e["id"] == r["val_eval_id"]
                )
                assert ev["split"] == "val"
                doc = read_json(root / ev["metrics_path"])
                exp = es.get(r["experiment_id"])
                rows[model].append(
                    {
                        "seed": seed,
                        "exp": r["experiment_id"],
                        "eval": r["val_eval_id"],
                        "bal": doc["metrics"]["balanced_accuracy"],
                        "f1": doc["metrics"]["macro_f1"],
                        "acc": doc["metrics"]["accuracy"],
                        "per_class": doc["metrics"]["per_class"],
                        "best_epoch": exp["best_epoch"],
                        "epochs": len(es.epochs(r["experiment_id"])),
                        "train_seconds": exp["train_seconds"],
                    }
                )

        def ms(values: list[float]) -> str:
            if len(values) < 2:
                return f"{values[0]:.4f}" if values else "-"
            return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"

        out = [
            f"# {self.name}: multi-seed summary — {VAL_LABEL}",
            "",
            f"**{VAL_LABEL}.** Checkpoints were selected on this same validation split (best "
            "val_loss), so these numbers are optimistic. Mean ± sample sd over seeds 0/1/2 "
            "(n = 3); the split is fixed, so the sd reflects training variance only.",
            "",
            "| model | val balanced accuracy (mean ± sd) | val macro F1 (mean ± sd) | "
            "per-seed bal. acc. (s0 / s1 / s2) | per-seed macro F1 (s0 / s1 / s2) | "
            "best epoch (s0/s1/s2) | epochs run | train time s (s0/s1/s2) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model in MODELS:
            rs = rows[model]
            if not rs:
                out.append(f"| {model} | (no completed runs) |||||||")
                continue
            out.append(
                f"| {model} | {ms([r['bal'] for r in rs])} | {ms([r['f1'] for r in rs])} | "
                + " / ".join(f"{r['bal']:.4f}" for r in rs)
                + " | "
                + " / ".join(f"{r['f1']:.4f}" for r in rs)
                + " | "
                + " / ".join(str(r["best_epoch"]) for r in rs)
                + " | "
                + " / ".join(str(r["epochs"]) for r in rs)
                + " | "
                + " / ".join(f"{r['train_seconds']:.0f}" for r in rs)
                + " |"
            )
        classes = next(iter(r["per_class"] for rs in rows.values() for r in rs), {})
        out += [
            "",
            f"Per-class VAL recall, seed 0 of each model — {VAL_LABEL}. Classes marked † are "
            "largely low-information (near-duplicates among them cannot be assessed by "
            "hashing; DATASETS.md).",
            "",
            "| class | " + " | ".join(MODELS) + " |",
            "|---|" + "---|" * len(MODELS),
        ]
        for cls in classes:
            mark = f" † ({LOW_INFO_CLASSES[cls]})" if cls in LOW_INFO_CLASSES else ""
            cells = []
            for model in MODELS:
                r0 = next((r for r in rows[model] if r["seed"] == 0), None)
                rec = None if r0 is None else r0["per_class"][cls]["recall"]
                cells.append("-" if rec is None else f"{rec:.4f}")
            out.append(f"| {cls}{mark} | " + " | ".join(cells) + " |")
        out += ["", "Runs (experiment id / val evaluation id):", ""]
        for model in MODELS:
            for r in rows[model]:
                out.append(f"- {model} seed {r['seed']}: {r['exp']} / {r['eval']}")
        text = "\n".join(out) + "\n"
        dest = root / "benchmarks" / self.name / "val_summary.md"
        dest.write_text(text, encoding="utf-8", newline="\n")
        print(text)
        print(f"wrote {dest}")
        return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["train", "evaluate-val", "latency", "check", "summary"])
    ap.add_argument("protocol_dir", type=Path)
    args = ap.parse_args()
    bench = Benchmark(args.protocol_dir)
    {
        "train": bench.train,
        "evaluate-val": bench.evaluate_val,
        "latency": bench.latency,
        "check": bench.check,
        "summary": bench.summary,
    }[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
