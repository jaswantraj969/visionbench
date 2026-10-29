"""M6 pipeline check (report only): the corrupted-image path reproduces clean predictions.

The original val images are re-encoded as lossless PNG and read through the same
ManifestImageDataset + recorded val transform as robustness evaluation; the predictions must
equal the clean B1 predictions exactly. Seed 0 of every trainable model, CUDA.

Usage: python scripts/robustness_identity_check.py <empty scratch dir>
"""

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionbench.core.config import Settings
from visionbench.data.imageio import load_rgb
from visionbench.evaluation.data import load_for_evaluation, open_for_evaluation
from visionbench.robustness.evaluate import _predict
from visionbench.robustness.materialize import corrupted_relpath, open_split
from visionbench.tracking.store import DB_FILENAME, ExperimentStore

s = Settings()
runs = json.loads(Path("benchmarks/eurosat-v1/runs.json").read_text(encoding="utf-8"))["runs"]
seed0 = [r for r in runs if r["seed"] == 0 and r["model"] != "majority_baseline"]
src = open_split(s, "ds-61112928e412", "sp-55320e11726c", "val")
tmp = Path(tempfile.mkdtemp(dir=sys.argv[1]))
files = []
for e in src.entries:
    rel = corrupted_relpath(e["relpath"])
    out = tmp.joinpath(*rel.split("/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(np.asarray(load_rgb(src.raw_root / e["relpath"]), dtype=np.uint8), "RGB").save(
        buf, format="PNG"
    )
    out.write_bytes(buf.getvalue())
    files.append(rel)
with ExperimentStore(s.data_dir("db") / DB_FILENAME) as store:
    for r in seed0:
        source = open_for_evaluation(store, s, r["experiment_id"], "best")
        loaded = load_for_evaluation(source, s.resolved_data_root)
        index = {c: i for i, c in enumerate(source.class_names)}
        labels = np.array([index[e["label"]] for e in src.entries])
        proba = _predict(loaded, tmp, files, labels, torch.device("cuda"), 64)
        ev = store.get_evaluation(r["val_eval_id"])
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (
                        s.resolved_data_root / Path(ev["metrics_path"]).parent / "predictions.csv"
                    ).read_text(encoding="utf-8")
                )
            )
        )
        clean_pred = np.array([index[x["pred_label"]] for x in rows])
        clean_prob = np.array([[float(x[f"prob_{c}"]) for c in source.class_names] for x in rows])
        print(
            r["model"],
            "predictions identical to clean:",
            bool((proba.argmax(1) == clean_pred).all()),
            "| max |prob diff|:",
            float(np.abs(proba - clean_prob).max()),
        )
