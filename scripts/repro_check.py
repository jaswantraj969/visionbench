"""Full-length reproducibility check of one benchmark run (labelled "repro-check").

Reruns a protocol model + seed with its frozen YAML (hash-checked against
``PROTOCOL_HASHES.txt``) through ``visionbench train`` as a NEW experiment, then compares it with
the reference benchmark run:

- every epoch-history field except ``epoch_seconds`` (wall time), bitwise (exact float equality);
- every tensor of ``best.pt`` and ``last.pt``, bitwise; on a difference, the number of differing
  tensors and the max absolute difference are reported;
- the checkpoint epochs and the experiment's best epoch / best value / stopped_reason.

The repro-check experiment is excluded from every benchmark result. It is listed in
``<protocol_dir>/repro_check.json`` (and nowhere in ``runs.json``).

Usage:
    python scripts/repro_check.py benchmarks/eurosat-v1 --model resnet18_scratch_64 --seed 1 \
        --reference exp-20261027-202124-e89a98
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionbench.core.power import keep_awake

EXP_RE = re.compile(r"experiment (exp-\d{8}-\d{6}-[0-9a-f]{6}) running")
TIMING_FIELDS = ("epoch_seconds",)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def frozen_yaml(protocol_dir: Path, model: str) -> tuple[Path, str]:
    expected = {}
    for line in (protocol_dir / "PROTOCOL_HASHES.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("*")] = digest
    path = protocol_dir / f"{model}.yaml"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected.get(path.name) != sha:
        raise SystemExit(f"{path.name}: sha256 {sha} != PROTOCOL_HASHES.txt; refusing to run")
    return path, sha


def history(data_root: Path, exp: str) -> list[dict[str, Any]]:
    path = data_root / "experiments" / exp / "history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_history(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({k for r in a + b for k in r} - set(TIMING_FIELDS))
    diffs = []
    for i, (ra, rb) in enumerate(zip(a, b, strict=False), start=1):
        for k in fields:
            if ra.get(k) != rb.get(k):
                diffs.append({"epoch": i, "field": k, "reference": ra.get(k), "rerun": rb.get(k)})
    return {
        "fields_compared": fields,
        "fields_excluded": list(TIMING_FIELDS),
        "epochs_reference": len(a),
        "epochs_rerun": len(b),
        "bitwise_identical": len(a) == len(b) and not diffs,
        "differences": diffs[:50],
        "differences_total": len(diffs),
    }


def compare_checkpoint(data_root: Path, a: str, b: str, name: str) -> dict[str, Any]:
    import torch

    def load(exp: str) -> dict[str, Any]:
        path = data_root / "experiments" / exp / "checkpoints" / name
        return torch.load(path, map_location="cpu", weights_only=True)

    ca, cb = load(a), load(b)
    ta, tb = ca["model_state"], cb["model_state"]
    differing, max_abs = [], 0.0
    for key in sorted(set(ta) | set(tb)):
        if key not in ta or key not in tb:
            differing.append(key)
            continue
        if ta[key].dtype != tb[key].dtype or ta[key].shape != tb[key].shape:
            differing.append(key)
            continue
        if not torch.equal(ta[key], tb[key]):
            differing.append(key)
            diff = (ta[key].double() - tb[key].double()).abs().max().item()
            max_abs = max(max_abs, diff)
    return {
        "file": name,
        "epoch_reference": ca["epoch"],
        "epoch_rerun": cb["epoch"],
        "metrics_identical": ca["metrics"] == cb["metrics"],
        "tensors": len(ta),
        "tensors_differing": len(differing),
        "first_differing_tensors": differing[:10],
        "max_abs_difference": max_abs,
        "bitwise_identical": not differing and ca["epoch"] == cb["epoch"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reference", required=True, help="benchmark experiment to reproduce")
    args = ap.parse_args()
    keep_awake()  # long run: do not let Windows sleep (M6 step 0c)

    from visionbench.core.config import Settings
    from visionbench.tracking.store import DB_FILENAME, ExperimentStore

    settings = Settings()
    data_root = settings.resolved_data_root
    runs = json.loads((args.protocol_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    ref_run = [r for r in runs if r.get("experiment_id") == args.reference]
    if len(ref_run) != 1 or (ref_run[0]["model"], ref_run[0]["seed"]) != (args.model, args.seed):
        raise SystemExit(f"{args.reference} is not the {args.model} seed {args.seed} benchmark run")
    yaml_path, yaml_sha = frozen_yaml(args.protocol_dir, args.model)

    log_dir = data_root / "benchmarks" / args.protocol_dir.name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"repro_check_{args.model}_seed{args.seed}.log"
    started = now()
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        cmd = ["train", "--config", str(yaml_path), "--seed", str(args.seed)]
        f.write(f"> visionbench {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.run(
            [sys.executable, "-m", "visionbench.cli", *cmd], stdout=f, stderr=subprocess.STDOUT
        ).returncode
    wall = time.perf_counter() - t0
    m = EXP_RE.search(log.read_text(encoding="utf-8", errors="replace"))
    if code != 0 or m is None:
        raise SystemExit(f"repro-check training failed (exit {code}); see {log}")
    rerun = m.group(1)

    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as es:
        ref_row, new_row = es.get(args.reference), es.get(rerun)
    keys = ("status", "best_epoch", "best_metric", "stopped_reason", "seed", "model")
    hist = compare_history(history(data_root, args.reference), history(data_root, rerun))
    best = compare_checkpoint(data_root, args.reference, rerun, "best.pt")
    last = compare_checkpoint(data_root, args.reference, rerun, "last.pt")
    record = {
        "label": "repro-check - excluded from all benchmark results",
        "model": args.model,
        "seed": args.seed,
        "config": yaml_path.name,
        "config_sha256": yaml_sha,
        "reference_experiment_id": args.reference,
        "repro_check_experiment_id": rerun,
        "started_at": started,
        "finished_at": now(),
        "wall_seconds": round(wall, 1),
        "log": log.relative_to(Path.cwd().resolve()).as_posix()
        if log.is_relative_to(Path.cwd().resolve())
        else str(log),
        "experiment_fields": {k: {"reference": ref_row[k], "rerun": new_row[k]} for k in keys},
        "train_seconds": {"reference": ref_row["train_seconds"], "rerun": new_row["train_seconds"]},
        "history": hist,
        "best_pt": best,
        "last_pt": last,
        "bitwise_identical": hist["bitwise_identical"]
        and best["bitwise_identical"]
        and last["bitwise_identical"]
        and all(ref_row[k] == new_row[k] for k in keys),
    }
    out = args.protocol_dir / "repro_check.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in record.items() if k != "history"}, indent=2))
    print(
        f"history identical={hist['bitwise_identical']} ({hist['epochs_reference']} vs "
        f"{hist['epochs_rerun']} epochs, {hist['differences_total']} differing values)"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
