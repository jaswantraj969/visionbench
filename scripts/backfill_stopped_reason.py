"""One-off backfill of ``experiments.stopped_reason`` (schema v6) from each run's own log.

Experiments trained before schema v6 have ``stopped_reason`` NULL. Every completed run wrote
``completed: best_epoch=... stopped=<reason>`` to ``<run_dir>/train.log``; this script reads that
line and sets ``stopped_reason`` only. No other field is changed, and rows that already have a
value, or that are not completed, are left alone.

Usage:
    python scripts/backfill_stopped_reason.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys

from visionbench.core.config import Settings
from visionbench.tracking.store import DB_FILENAME, ExperimentStore

STOPPED_RE = re.compile(r" INFO completed: best_epoch=\S+ best_\w+=\S+ stopped=(\w+)$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()
    settings = Settings()
    counts = {"backfilled": 0, "already_set": 0, "not_completed": 0, "no_log_line": 0}
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        for exp in store.list():
            if exp["status"] != "completed":
                counts["not_completed"] += 1
                continue
            if exp["stopped_reason"] is not None:
                counts["already_set"] += 1
                continue
            log = settings.resolved_data_root / exp["run_dir"] / "train.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            found = STOPPED_RE.findall(text)
            if len(found) != 1:
                counts["no_log_line"] += 1
                print(f"{exp['id']}: {len(found)} completion lines in train.log; left NULL")
                continue
            if not args.dry_run:
                store.update(exp["id"], stopped_reason=found[0])
            counts["backfilled"] += 1
            print(f"{exp['id']}: stopped_reason={found[0]}")
    verb = "would backfill" if args.dry_run else "backfilled"
    print(
        f"{verb} {counts['backfilled']}; already set {counts['already_set']}; "
        f"not completed {counts['not_completed']}; no usable log line {counts['no_log_line']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
