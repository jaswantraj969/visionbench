"""Pilot for a frozen benchmark protocol: timing, GPU memory and GPU determinism.

Pilot runs are NOT benchmark runs. Their experiment ids are written to ``pilot.json`` in the
protocol folder, and they are excluded from every benchmark result.

1. Timing: each trainable protocol model is trained for 1 epoch with seed 0, each in a fresh
   process. Reported: seconds per epoch (train + val), peak GPU memory
   (``torch.cuda.max_memory_allocated``), whether the batch size fits, and the wall time of the
   whole run.
2. Determinism: two identical short runs (``max_steps``, seed 0) per trainable model, each
   through the CLI in its own process. The epoch losses/metrics and every ``best.pt`` tensor
   are compared bitwise; if they differ, the max absolute difference is reported.
3. Budget: an upper-bound estimate of the full protocol (every run reaching the epoch cap).

Usage:
    python scripts/pilot_benchmark.py benchmarks/eurosat-v1 [--det-steps 50]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionbench.core.power import keep_awake

SEEDS_PER_MODEL = 3
TRAINABLE = (
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
    "resnet18_scratch_64",
)
EXP_RE = re.compile(r"experiment (exp-\d{8}-\d{6}-[0-9a-f]{6}) running")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def timing_one(yaml_path: Path) -> dict[str, Any]:
    """Runs inside a fresh process: train 1 epoch (seed 0), measure peak GPU memory."""
    import torch

    from visionbench.core.config import Settings
    from visionbench.training.config import load_train_config
    from visionbench.training.runner import run_training

    config = load_train_config(yaml_path, {"seed": 0, "epochs": 1})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out: dict[str, Any] = {"model": yaml_path.stem, "batch_size": config.batch_size}
    try:
        result = run_training(config, Settings())
    except torch.cuda.OutOfMemoryError as exc:
        return out | {"fits": False, "error": f"OutOfMemoryError: {exc}"}
    out["wall_seconds"] = round(time.perf_counter() - started, 2)
    out["experiment_id"] = result.experiment_id
    out["status"] = result.status
    out["fits"] = result.status == "completed"
    out["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    out["gpu"] = torch.cuda.get_device_name(0)
    epochs = result.history
    out["epoch_seconds"] = round(epochs[0]["epoch_seconds"], 2) if epochs else None
    out["steps_per_epoch"] = epochs[0]["steps"] if epochs else None
    return out


def run_cli_train(yaml_path: Path, steps: int) -> tuple[str, str]:
    cmd = [
        sys.executable, "-m", "visionbench.cli", "train", "--config", str(yaml_path),
        "--seed", "0", "--max-steps", str(steps),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = EXP_RE.search(proc.stdout)
    if proc.returncode != 0 or m is None:
        raise RuntimeError(f"pilot run failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return m.group(1), proc.stdout


def compare_runs(data_root: Path, a: str, b: str) -> dict[str, Any]:
    import torch

    def history(exp: str) -> list[dict[str, Any]]:
        path = data_root / "experiments" / exp / "history.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    keys = ("train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy", "steps")
    ha, hb = history(a), history(b)
    losses_equal = [[r[k] for k in keys] for r in ha] == [[r[k] for k in keys] for r in hb]
    sa = torch.load(data_root / "experiments" / a / "checkpoints" / "best.pt", weights_only=True)
    sb = torch.load(data_root / "experiments" / b / "checkpoints" / "best.pt", weights_only=True)
    ta, tb = sa["model_state"], sb["model_state"]
    differing, max_abs = [], 0.0
    for name in ta:
        if not torch.equal(ta[name], tb[name]):
            differing.append(name)
            diff = (ta[name].double() - tb[name].double()).abs().max().item()
            max_abs = max(max_abs, diff)
    return {
        "runs": [a, b],
        "epochs_compared": len(ha),
        "history_fields": list(keys),
        "history_bitwise_identical": losses_equal,
        "history_a": [{k: r[k] for k in keys} for r in ha],
        "history_b": [{k: r[k] for k in keys} for r in hb],
        "best_pt_tensors": len(ta),
        "best_pt_tensors_differing": len(differing),
        "best_pt_bitwise_identical": not differing and set(ta) == set(tb),
        "max_abs_difference": max_abs,
        "first_differing_tensors": differing[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--det-steps", type=int, default=50)
    ap.add_argument("--timing-one", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    keep_awake()  # long run: do not let Windows sleep (M6 step 0c)

    if args.timing_one:  # child process
        print("PILOT_RESULT " + json.dumps(timing_one(args.timing_one)))
        return 0

    from visionbench.core.config import Settings

    data_root = Settings().resolved_data_root
    report: dict[str, Any] = {
        "label": "pilot - excluded from benchmark results",
        "started_at": now(),
        "timing": [],
        "determinism": [],
    }
    for name in TRAINABLE:
        yaml_path = args.protocol_dir / f"{name}.yaml"
        print(f"[{now()}] timing {name} (1 epoch, seed 0)", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, str(args.protocol_dir), "--timing-one", str(yaml_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        line = next((x for x in proc.stdout.splitlines() if x.startswith("PILOT_RESULT ")), None)
        if line is None:
            result = {"model": name, "fits": False, "error": proc.stderr[-2000:]}
        else:
            result = json.loads(line.removeprefix("PILOT_RESULT "))
        result["process_wall_seconds"] = round(time.perf_counter() - t0, 2)
        report["timing"].append(result)
        print("  " + json.dumps(result), flush=True)

    for name in TRAINABLE:
        yaml_path = args.protocol_dir / f"{name}.yaml"
        print(f"[{now()}] determinism {name} (2 x {args.det_steps} steps, seed 0)", flush=True)
        a, _ = run_cli_train(yaml_path, args.det_steps)
        b, _ = run_cli_train(yaml_path, args.det_steps)
        cmp = {"model": name, "max_steps": args.det_steps} | compare_runs(data_root, a, b)
        report["determinism"].append(cmp)
        print(
            f"  {name}: history identical={cmp['history_bitwise_identical']} "
            f"best.pt identical={cmp['best_pt_bitwise_identical']} "
            f"max_abs_diff={cmp['max_abs_difference']}",
            flush=True,
        )

    # Upper bound: every run reaches the epoch cap; per-run overhead from the timing runs.
    budget = []
    for t in report["timing"]:
        if not t.get("epoch_seconds"):
            continue
        overhead = t["process_wall_seconds"] - t["epoch_seconds"]
        per_run = overhead + 25 * t["epoch_seconds"]
        budget.append(
            {
                "model": t["model"],
                "epoch_seconds": t["epoch_seconds"],
                "per_run_overhead_seconds": round(overhead, 1),
                "upper_bound_per_run_seconds": round(per_run, 1),
                "upper_bound_3_seeds_seconds": round(SEEDS_PER_MODEL * per_run, 1),
            }
        )
    total = sum(b["upper_bound_3_seeds_seconds"] for b in budget)
    report["budget"] = {
        "assumption": "every run trains the full 25 epochs (early stopping ignored); "
        "baseline runs (seconds) not included",
        "per_model": budget,
        "upper_bound_total_hours": round(total / 3600, 2),
        "limit_hours": 6,
        "within_limit": total / 3600 <= 6,
    }
    report["finished_at"] = now()
    report["pilot_experiment_ids"] = [
        t["experiment_id"] for t in report["timing"] if "experiment_id" in t
    ] + [e for d in report["determinism"] for e in d["runs"]]
    out = args.protocol_dir / "pilot.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["budget"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
