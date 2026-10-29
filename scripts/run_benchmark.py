"""Run a frozen benchmark protocol through the ``visionbench`` CLI and record ``runs.json``.

Subcommands (all take the protocol folder, e.g. ``benchmarks/eurosat-v1``):

- ``train [--only m1,m2]``: for every protocol model x seed 0/1/2, in the run order below, run
  ``visionbench train --config <yaml> --seed <s>``.
  - Before anything runs, the YAMLs and BENCHMARK_PROTOCOL.md are checked against
    ``PROTOCOL_HASHES.txt``.
  - The driver is resumable: runs already completed in ``runs.json`` are skipped.
  - Each attempt is recorded as ``in_progress`` before it starts. If the driver or session dies
    (sleep, power loss, crash, killed process), the next ``train`` marks that attempt
    ``interrupted``, with an infrastructure reason, and reruns it from scratch with the identical
    config. Interruptions do not count toward the retry limit.
  - A run that itself fails is retried at most once with the identical config, and both attempts
    are recorded.
  - A CUDA out-of-memory failure stops the driver instead, for a human to judge.
  - While training, Windows is asked not to sleep (``SetThreadExecutionState``; process-scoped).
- ``annotate <index> <text>``: add the known cause of an interruption to a ``runs.json`` entry.
- ``evaluate-val``: evaluate each completed run's val-selected checkpoint on **VAL** (never test).
- ``evaluate-test`` (M8, once): the single clean **TEST** evaluation of every run's val-selected
  checkpoint, in protocol order.
  - Refuses to start unless the protocol hashes and ``FINAL_ANALYSIS_PLAN.md`` (against
    ``FINAL_ANALYSIS_PLAN_HASHES.txt``) verify.
  - ``VISIONBENCH_TEST_IMAGES_UNLOCKED=true`` is set in the evaluation's process only.
  - Attempts are recorded in ``results/test_runs.json`` (``runs.json`` stays frozen).
  - A run that already has a TEST row in the database is never evaluated again; the row is
    recorded instead.
  - An attempt left ``in_progress`` with no row written (the driver or session died) is marked
    ``interrupted`` and rerun with the identical command. Any failed evaluation stops the driver
    for a human to judge; there is no automatic retry.
- ``latency``: CPU and CUDA latency for seed 0 of each trainable model.
- ``check``: before M8 (no ``results/test_runs.json``), assert that no TEST evaluation of any
  kind exists. After M8, assert exactly 1 clean TEST and 12 robustness TEST evaluations (paired
  with it) per benchmark run, and none for pilot/repro-check experiments or anything else.
- ``freeze-check`` (M8, before ``evaluate-test``): the automatable items of the freeze checklist.
  - Hashes: the protocol, robustness and final-analysis-plan hashes verify.
  - Zero TEST evaluations of any kind (``check``).
  - ``dataset verify --full`` passes and the dataset fingerprint equals the protocol's.
  - For each of the 15 runs, the val-selected checkpoint file, its artifact row and its VAL
    evaluation record have the same sha256.
- ``summary``: multi-seed VALIDATION summary (Markdown) built from the recorded val evaluations.

``runs.json`` holds experiment ids, statuses, timestamps and evaluation/latency ids only; logs go
to ``<data_root>/benchmarks/<name>/logs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionbench.core.power import keep_awake

# Run order fixed by the supervisor (VB-B1 budget decision); seeds 0, 1, 2 within each model.
MODELS = (
    "majority_baseline",
    "resnet18_scratch_64",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
)
TRAINABLE = MODELS[1:]
INTERRUPTED = (
    "infrastructure interruption: the driver/session ended while this run was in progress, so no "
    "outcome was recorded; rerun from scratch with the identical config (not counted toward the "
    "one-retry limit; GPU determinism was verified bitwise in the pilot)"
)
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")
SEEDS = (0, 1, 2)
LOW_INFO_CLASSES = {"SeaLake": "78 % low-information", "Forest": "34 % low-information"}
VAL_LABEL = "VALIDATION (selection-biased) — not test results"
EXP_RE = re.compile(r"experiment (exp-\d{8}-\d{6}-[0-9a-f]{6}) running")
# The CLI's final line; stopped_reason is not stored in the experiment DB.
STOPPED_RE = re.compile(r"\(stopped: (\w+); best epoch")
EVAL_RE = re.compile(r"\b(ev-\d{8}-\d{6}-[0-9a-f]{6})\b")
LAT_RE = re.compile(r"\b(lat-\d{8}-\d{6}-[0-9a-f]{6})\b")


def _relative(path: Path) -> str:
    """Path relative to the working directory (the repo root), so runs.json has no user path."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _stopped_reason(rec: dict[str, Any]) -> str:
    """Why training stopped, from the CLI's final line in the run's log."""
    log = Path(rec["log"]) if rec.get("log") else None
    text = log.read_text(encoding="utf-8", errors="replace") if log and log.is_file() else ""
    m = STOPPED_RE.search(text)
    return m.group(1) if m else "unknown"


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

    def cli(self, args: list[str], log: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
        """Run the CLI with output streamed to ``log`` (readable while it runs)."""
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"> visionbench {' '.join(args)}\n")
            f.flush()
            code = subprocess.run(
                [sys.executable, "-m", "visionbench.cli", *args],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=None if env is None else os.environ | env,
            ).returncode
        return code, log.read_text(encoding="utf-8", errors="replace")

    def store(self):  # noqa: ANN201
        from visionbench.tracking.store import DB_FILENAME, ExperimentStore

        return ExperimentStore(self.settings.data_dir("db") / DB_FILENAME)

    # --- subcommands ------------------------------------------------------------------------

    def train(self, only: list[str] | None = None) -> None:
        hashes = self.verify_hashes()
        self._mark_interrupted()
        keep_awake()
        for model in MODELS:
            if only and model not in only:
                continue
            yaml_path = self.dir / f"{model}.yaml"
            for seed in SEEDS:
                attempts = [
                    r for r in self.data["runs"] if (r["model"], r["seed"]) == (model, seed)
                ]
                if any(r["status"] == "completed" for r in attempts):
                    print(f"skip {model} seed {seed}: completed already", flush=True)
                    continue
                failures = sum(r["status"] in ("failed", "cancelled") for r in attempts)
                while failures < 2:
                    rec = self._train_one(model, yaml_path, seed, len(attempts) + 1, hashes)
                    attempts.append(rec)
                    if rec["status"] == "completed":
                        break
                    if any(m in (rec.get("error") or "") for m in OOM_MARKERS):
                        raise SystemExit(
                            f"{model} seed {seed}: CUDA out of memory ({rec['experiment_id']}); "
                            "stopping for a human decision (no batch-size change without an "
                            "approved amendment)"
                        )
                    failures += 1
                    print(f"  {model} seed {seed}: {rec['status']} ({failures} failure(s))")
                if failures >= 2:
                    print(f"  {model} seed {seed}: failed twice; recorded, moving on", flush=True)

    def _mark_interrupted(self) -> None:
        """Attempts left ``in_progress`` by a dead driver become ``interrupted``."""
        from visionbench.core.process import process_is

        for r in self.data["runs"]:
            if r["status"] != "in_progress":
                continue
            log = Path(r["log"])
            m = (
                EXP_RE.search(log.read_text(encoding="utf-8", errors="replace"))
                if log.is_file()
                else None
            )
            r["experiment_id"] = m.group(1) if m else None
            if r["experiment_id"]:
                with self.store() as es:
                    exp = es.get(r["experiment_id"])
                    if process_is(exp.get("runner_pid"), exp.get("runner_create_time")):
                        raise SystemExit(
                            f"{r['experiment_id']} is still being trained; not resuming"
                        )
                    if exp["status"] in ("pending", "running"):
                        es.transition(
                            r["experiment_id"],
                            "failed",
                            error="interrupted: benchmark session ended (infrastructure)",
                        )
                    r["experiment_status"] = es.get(r["experiment_id"])["status"]
            r["status"] = "interrupted"
            r["interruption"] = INTERRUPTED
            r["detected_at"] = now()
            print(f"marked interrupted: {r['model']} seed {r['seed']} ({r['experiment_id']})")
        self.save()

    def _train_one(
        self, model: str, yaml_path: Path, seed: int, attempt: int, hashes: dict[str, str]
    ) -> dict[str, Any]:
        log = self.logs / f"train_{model}_seed{seed}_attempt{attempt}.log"
        rec: dict[str, Any] = {
            "model": model,
            "config": yaml_path.name,
            "config_sha256": hashes[yaml_path.name],
            "seed": seed,
            "attempt": attempt,
            "status": "in_progress",
            "started_at": now(),
            "log": _relative(log),
        }
        self.data["runs"].append(rec)
        self.save()
        print(f"[{rec['started_at']}] train {model} seed {seed} (attempt {attempt})", flush=True)
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
            rec["status"] = "failed"
            rec["error"] = f"exit code {code}; no experiment created: {out[-500:]}"
        rec["exit_code"] = code
        self.save()
        print(f"  -> {rec['experiment_id']} {rec['status']}", flush=True)
        return rec

    def annotate(self, index: int, text: str) -> None:
        self.data["runs"][index]["interruption_detail"] = text
        self.save()

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

    def verify_plan(self) -> str:
        """sha256 of FINAL_ANALYSIS_PLAN.md, checked against FINAL_ANALYSIS_PLAN_HASHES.txt."""
        plan = self.dir / "FINAL_ANALYSIS_PLAN.md"
        hashes = self.dir / "FINAL_ANALYSIS_PLAN_HASHES.txt"
        if not plan.is_file() or not hashes.is_file():
            raise SystemExit(
                "FINAL_ANALYSIS_PLAN.md and FINAL_ANALYSIS_PLAN_HASHES.txt must be committed"
            )
        for line in hashes.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                digest, name = line.split(maxsplit=1)
                actual = sha256(self.dir / name.lstrip("*"))
                if actual != digest:
                    raise SystemExit(f"{name}: sha256 {actual} != {digest}; refusing to run")
        return sha256(plan)

    def test_record_path(self) -> Path:
        return self.dir / "results" / "test_runs.json"

    def evaluate_test(self) -> None:
        from visionbench.evaluation.compare import SELECTED_CHECKPOINTS

        self.verify_hashes()
        plan_sha = self.verify_plan()
        keep_awake()
        path = self.test_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"benchmark": self.name, "split": "test", "label": "TEST", "runs": []}
        )
        rec["final_analysis_plan_sha256"] = plan_sha

        def save() -> None:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
            tmp.replace(path)

        final = self.final_runs()
        for model in MODELS:
            for seed in SEEDS:
                r = final[(model, seed)]
                if r["status"] != "completed":
                    raise SystemExit(f"{model} seed {seed} is not completed; refusing to run")
                exp = r["experiment_id"]
                attempts = [a for a in rec["runs"] if a["experiment_id"] == exp]
                with self.store() as es:
                    rows = [
                        e
                        for e in es.evaluations(exp, split="test")
                        if e["checkpoint"] in SELECTED_CHECKPOINTS
                    ]
                if len(rows) > 1:
                    raise SystemExit(f"{exp} has {len(rows)} TEST evaluations; stop for review")
                if rows:
                    last = attempts[-1] if attempts else None
                    if last is None or last["status"] != "completed":
                        if last is not None:
                            last["status"] = "completed"
                            last["note"] = "row found in the database after the attempt"
                        else:
                            rec["runs"].append(
                                {
                                    "model": model,
                                    "seed": seed,
                                    "experiment_id": exp,
                                    "attempt": 1,
                                    "status": "completed",
                                    "note": "row found in the database; not evaluated again",
                                }
                            )
                            last = rec["runs"][-1]
                        last["test_eval_id"] = rows[0]["id"]
                        save()
                    print(f"skip {model} seed {seed}: TEST evaluation {rows[0]['id']}", flush=True)
                    continue
                for a in attempts:
                    if a["status"] == "in_progress":
                        a["status"] = "interrupted"
                        a["note"] = (
                            "infrastructure interruption: the driver/session ended during this "
                            "attempt and no evaluation row was written; rerun identically"
                        )
                if any(a["status"] == "failed" for a in attempts):
                    raise SystemExit(f"{model} seed {seed} failed before; a human must decide")
                device = "cpu" if model == "majority_baseline" else "cuda"
                n = len(attempts) + 1
                log = self.logs / f"eval_test_{model}_seed{seed}_attempt{n}.log"
                attempt: dict[str, Any] = {
                    "model": model,
                    "seed": seed,
                    "experiment_id": exp,
                    "attempt": n,
                    "device": device,
                    "status": "in_progress",
                    "started_at": now(),
                    "log": _relative(log),
                }
                rec["runs"].append(attempt)
                save()
                print(f"[{now()}] evaluate {model} seed {seed} on TEST (attempt {n})", flush=True)
                code, out = self.cli(
                    ["evaluate", exp, "--split", "test", "--device", device],
                    log,
                    env={"VISIONBENCH_TEST_IMAGES_UNLOCKED": "true"},
                )
                attempt["finished_at"] = now()
                attempt["exit_code"] = code
                m = EVAL_RE.search(out)
                if code != 0 or m is None:
                    attempt["status"] = "failed"
                    save()
                    raise SystemExit(f"TEST evaluation failed for {exp} (see {log}); stop")
                attempt["status"] = "completed"
                attempt["test_eval_id"] = m.group(1)
                save()
                print(f"  -> {m.group(1)}", flush=True)

    def freeze_check(self) -> None:
        from visionbench.core.jsonio import read_json

        failures: list[str] = []
        self.verify_hashes()
        print("b. protocol hashes: OK", flush=True)
        rdir = self.dir / "robustness"
        for line in (rdir / "ROBUSTNESS_HASHES.txt").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                digest, name = line.split(maxsplit=1)
                if sha256(rdir / name.lstrip("*")) != digest:
                    failures.append(f"robustness hash mismatch: {name}")
        print(f"b. final analysis plan sha256 {self.verify_plan()}: OK", flush=True)
        if self.test_record_path().is_file():
            raise SystemExit("results/test_runs.json exists: the TEST evaluation has started")
        self.check()
        print("c. zero TEST evaluations (clean and robustness): OK", flush=True)
        text = (self.dir / "BENCHMARK_PROTOCOL.md").read_text(encoding="utf-8")
        ds_id = re.search(r"\| Dataset \| `(ds-[0-9a-f]{12})`", text).group(1)
        frozen = re.search(r"\| Fingerprint \| `([0-9a-f]{64})`", text).group(1)
        log = self.logs / "freeze_check_dataset_verify_full.log"
        code, out = self.cli(["dataset", "verify", ds_id, "--full"], log)
        record = read_json(self.settings.data_dir("datasets") / ds_id / "dataset.json")
        if code != 0:
            failures.append(f"dataset verify --full failed (see {log})")
        if record["fingerprint"] != frozen:
            failures.append(f"fingerprint {record['fingerprint']} != protocol {frozen}")
        print(
            f"d. dataset verify --full exit {code}; fingerprint "
            f"{'matches' if record['fingerprint'] == frozen else 'DIFFERS'}",
            flush=True,
        )
        root = self.settings.resolved_data_root
        n_ok = 0
        with self.store() as es:
            for (model, seed), r in sorted(self.final_runs().items()):
                exp = r["experiment_id"]
                ckpt = read_json(root / es.get_evaluation(r["val_eval_id"])["metrics_path"])[
                    "checkpoint"
                ]
                rel = f"experiments/{exp}/checkpoints/{ckpt['file']}"
                rows = [a for a in es.artifacts(exp) if a["relpath"] == rel]
                on_disk = sha256(root / rel) if (root / rel).is_file() else None
                if len(rows) != 1 or not (rows[0]["sha256"] == on_disk == ckpt["sha256"]):
                    failures.append(f"{model} seed {seed}: checkpoint {rel} sha256 mismatch")
                else:
                    n_ok += 1
        print(f"e. val-selected checkpoints matching their artifact rows: {n_ok}/15", flush=True)
        if n_ok != 15:
            failures.append(f"only {n_ok}/15 checkpoints verified")
        if failures:
            raise SystemExit("FREEZE CHECK FAILED:\n- " + "\n- ".join(failures))
        print("OK: freeze checklist items b-e pass (a, f, g are checked by hand)")

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
        repro_path = self.dir / "repro_check.json"  # excluded from results, still checked
        if repro_path.is_file():
            pilot.add(
                json.loads(repro_path.read_text(encoding="utf-8"))["repro_check_experiment_id"]
            )
        with self.store() as es:
            test_evals = [
                e for e in es.evaluations(split="test") if e["experiment_id"] in bench | pilot
            ]
            all_test = es.evaluations(split="test")
            val_evals = [e for e in es.evaluations(split="val") if e["experiment_id"] in bench]
            rob_test = es.robustness_evaluations(split="test")  # M6: corrupted-input results
            rob_val = [
                r for r in es.robustness_evaluations(split="val") if r["experiment_id"] in bench
            ]
        print(f"benchmark experiments: {len(bench)}; pilot + repro-check experiments: {len(pilot)}")
        print(f"val evaluations of benchmark experiments: {len(val_evals)}")
        print(f"test evaluations of benchmark or pilot experiments: {len(test_evals)}")
        print(f"test evaluations in the whole database: {len(all_test)}")
        print(f"robustness val evaluations of benchmark experiments: {len(rob_val)}")
        print(f"robustness test evaluations in the whole database: {len(rob_test)}")
        if not self.test_record_path().is_file():  # before M8: nothing may touch TEST
            assert not test_evals, f"TEST evaluations exist: {[e['id'] for e in test_evals]}"
            assert not all_test, f"TEST evaluations exist: {[e['id'] for e in all_test]}"
            assert not rob_test, f"TEST robustness evaluations exist: {[r['id'] for r in rob_test]}"
            print(
                "OK: zero test evaluations of any kind (clean or robustness) in the whole database"
            )
            return
        # after M8: exactly one clean TEST evaluation and 12 robustness TEST evaluations per
        # benchmark run, paired with it; nothing for pilot / repro-check experiments.
        for exp in sorted(bench):
            clean = [e for e in all_test if e["experiment_id"] == exp]
            assert len(clean) == 1, f"{exp}: {len(clean)} clean TEST evaluations, expected 1"
            rob = [r for r in rob_test if r["experiment_id"] == exp]
            assert len(rob) == 12, f"{exp}: {len(rob)} robustness TEST evaluations, expected 12"
            assert {r["clean_evaluation_id"] for r in rob} == {clean[0]["id"]}, exp
        other = [e["id"] for e in all_test if e["experiment_id"] not in bench]
        other += [r["id"] for r in rob_test if r["experiment_id"] not in bench]
        assert not other, f"TEST evaluations outside the benchmark runs: {other}"
        print(
            f"OK: exactly 1 clean TEST and 12 robustness TEST evaluations for each of the "
            f"{len(bench)} benchmark runs; 0 for pilot/repro-check and anything else"
        )

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
                        "stopped": _stopped_reason(r),
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
            "best epoch (s0/s1/s2) | epochs trained (s0/s1/s2) | stopped_reason (s0/s1/s2) | "
            "train time s, indicative (s0/s1/s2) |",
            "|---|---|---|---|---|---|---|---|---|",
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
                + " / ".join(r["stopped"] for r in rs)
                + " | "
                + " / ".join(f"{r['train_seconds']:.0f}" for r in rs)
                + " |"
            )
        capped = [
            f"{model} seed {r['seed']} ({r['exp']}; best epoch {r['best_epoch']})"
            for model in MODELS
            for r in rows[model]
            if r["stopped"] == "epochs_completed"
        ]
        out += [
            "",
            "Train time is indicative only: it depends on the session and on the laptop's "
            "thermal state.",
            "",
            "Runs that reached the 25-epoch cap (stopped_reason `epochs_completed`) instead of "
            "early stopping — a finding, not acted on: "
            + ("; ".join(capped) if capped else "none")
            + ".",
        ]
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
    ap.add_argument(
        "command",
        choices=[
            "train",
            "evaluate-val",
            "freeze-check",
            "evaluate-test",
            "latency",
            "check",
            "summary",
            "annotate",
        ],
    )
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--only", help="train: comma-separated model names")
    ap.add_argument("--index", type=int, help="annotate: runs.json entry index")
    ap.add_argument("--text", help="annotate: known cause of an interruption")
    args = ap.parse_args()
    bench = Benchmark(args.protocol_dir)
    if args.command == "train":
        bench.train(args.only.split(",") if args.only else None)
        return 0
    if args.command == "annotate":
        bench.annotate(args.index, args.text)
        return 0
    {
        "evaluate-val": bench.evaluate_val,
        "evaluate-test": bench.evaluate_test,
        "freeze-check": bench.freeze_check,
        "latency": bench.latency,
        "check": bench.check,
        "summary": bench.summary,
    }[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
