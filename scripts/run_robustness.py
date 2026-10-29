"""Robustness driver for a frozen benchmark: evaluate every run on every condition of the suite.

- Checks ``robustness/ROBUSTNESS_HASHES.txt`` before anything runs.
- Runs ``visionbench robustness evaluate`` for each run of ``runs.json`` (protocol order),
  sequentially, each in its own process, with the run's clean VAL evaluation pinned
  (``val_eval_id``). Windows is asked not to sleep.
- Resumable: conditions already evaluated are skipped by the library; a run whose process died
  is simply started again (recorded as a new attempt with the reason).
- Writes ``robustness/robustness_runs.json`` (attempts, statuses, robustness ids, logs).
- ``--split test`` (M8, after ``run_benchmark.py evaluate-test``): each run is paired with its
  single clean TEST evaluation from ``results/test_runs.json``; the attempts go to
  ``robustness/robustness_runs_test.json``; ``VISIONBENCH_TEST_IMAGES_UNLOCKED=true`` is set in
  the evaluation processes only. The TEST conditions must be materialized first.

Usage:
    python scripts/run_robustness.py benchmarks/eurosat-v1 --split val
    python scripts/run_robustness.py benchmarks/eurosat-v1 --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionbench.core.power import keep_awake

ORDER = (
    "majority_baseline",
    "resnet18_scratch_64",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def verify_hashes(rdir: Path) -> None:
    for line in (rdir / "ROBUSTNESS_HASHES.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        actual = hashlib.sha256((rdir / name.lstrip("*")).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name}: sha256 {actual} != ROBUSTNESS_HASHES.txt; refusing to run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args()
    rdir = args.protocol_dir / "robustness"
    verify_hashes(rdir)
    keep_awake()
    suite = rdir / "robustness_suite.yaml"
    runs = json.loads((args.protocol_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    final = {(r["model"], r["seed"]): r for r in runs}
    clean_ids = {(r["model"], r["seed"]): r["val_eval_id"] for r in runs}
    env = None
    if args.split == "test":
        # M8: pinned to each run's single clean TEST evaluation (results/test_runs.json);
        # the TEST unlock is set in the evaluation processes only.
        test_runs = json.loads(
            (args.protocol_dir / "results" / "test_runs.json").read_text(encoding="utf-8")
        )["runs"]
        clean_ids = {
            (a["model"], a["seed"]): a["test_eval_id"]
            for a in test_runs
            if a["status"] == "completed"
        }
        env = os.environ | {"VISIONBENCH_TEST_IMAGES_UNLOCKED": "true"}
    record_path = rdir / (
        "robustness_runs.json" if args.split == "val" else "robustness_runs_test.json"
    )
    record: dict[str, Any] = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.is_file()
        else {"suite": str(suite.as_posix()), "split": args.split, "attempts": []}
    )
    log_dir = Path("var/benchmarks") / args.protocol_dir.name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for a in record["attempts"]:  # an attempt left "in_progress" means the driver died
        if a["status"] == "in_progress":
            a["status"] = "interrupted"
            a["note"] = (
                "driver/session ended during this attempt; rerun with the identical command "
                "(conditions already recorded are skipped)"
            )
    for model in ORDER:
        for seed in (0, 1, 2):
            r = final[(model, seed)]
            done = [
                a
                for a in record["attempts"]
                if a["experiment_id"] == r["experiment_id"] and a["status"] == "completed"
            ]
            if done:
                print(f"skip {model} seed {seed}: completed", flush=True)
                continue
            n = 1 + sum(a["experiment_id"] == r["experiment_id"] for a in record["attempts"])
            log = log_dir / f"robustness_{args.split}_{model}_seed{seed}_attempt{n}.log"
            attempt = {
                "model": model,
                "seed": seed,
                "experiment_id": r["experiment_id"],
                "clean_eval_id": clean_ids[(model, seed)],
                "attempt": n,
                "status": "in_progress",
                "started_at": now(),
                "log": log.as_posix(),
            }
            record["attempts"].append(attempt)
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"[{attempt['started_at']}] {model} seed {seed} (attempt {n})", flush=True)
            cmd = [
                "robustness", "evaluate", r["experiment_id"], "--suite", str(suite),
                "--split", args.split, "--clean-eval-id", clean_ids[(model, seed)],
                "--device", "cpu" if model == "majority_baseline" else "cuda",
            ]  # fmt: skip
            with open(log, "w", encoding="utf-8") as f:
                f.write(f"> visionbench {' '.join(cmd)}\n")
                f.flush()
                code = subprocess.run(
                    [sys.executable, "-m", "visionbench.cli", *cmd],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                ).returncode
            text = log.read_text(encoding="utf-8", errors="replace")
            attempt["finished_at"] = now()
            attempt["exit_code"] = code
            attempt["status"] = "completed" if code == 0 else "failed"
            attempt["robustness_ids"] = {
                line.split(": ")[0].strip(): line.split(": ")[1].strip()
                for line in text.splitlines()
                if line.startswith("  ") and ": rb-" in line
            }
            attempt["summary"] = next(
                (x for x in text.splitlines() if x.startswith("robustness ")), text[-300:]
            )
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"  -> {attempt['status']}: {attempt['summary']}", flush=True)
            if code != 0:
                raise SystemExit(f"{model} seed {seed} failed; see {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
