"""Grad-CAM model-randomization check on real models: cascading randomization from the top.

Following Adebayo et al. (2018, "Sanity Checks for Saliency Maps"), the trained model's layers
are re-initialised one stage at a time, starting at the classifier and moving towards the input;
each stage keeps the stages above it randomised (cascading). After every stage the Grad-CAM of
the same target class (the trained model's prediction) is compared with the trained model's CAM
by Spearman rank correlation over the CAM cells, per image.

Report only (no threshold): a model whose CAM is still highly similar (mean |rho| > 0.5) after
full randomization is flagged. Images: the seeded M5 demo selection (20 VALIDATION images,
``var/demos/m5/selection.json``). Models: seed 0 of each trainable protocol model, loaded
through the verified inference path (CPU). "Re-initialise" = ``reset_parameters()`` of every
submodule that has it (conv, linear, BatchNorm: weights, biases and running statistics), with a
fixed seed per stage.

Usage:
    python scripts/cascading_randomization.py benchmarks/eurosat-v1 --out var/demos/m6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from scipy.stats import spearmanr
from torch import nn

from visionbench.core.config import Settings
from visionbench.core.power import keep_awake
from visionbench.explain.gradcam import GRADCAM_VERSION, grad_cam, target_layer_name
from visionbench.inference.service import DatasetSource, InferenceService

LABEL = "VALIDATION images - diagnostic, not results"
FLAG_ABS_RHO = 0.5
STAGES: dict[str, list[tuple[str, list[str]]]] = {
    "small_cnn": [
        ("classifier", ["classifier"]),
        ("block4", ["features.block4"]),
        ("block3", ["features.block3"]),
        ("block2", ["features.block2"]),
        ("block1", ["features.block1"]),
    ],
    "resnet18": [
        ("fc", ["fc"]),
        ("layer4", ["layer4"]),
        ("layer3", ["layer3"]),
        ("layer2", ["layer2"]),
        ("layer1", ["layer1"]),
        ("stem (conv1, bn1)", ["conv1", "bn1"]),
    ],
    "efficientnet_b0": [("classifier", ["classifier"])]
    + [(f"features.{i}", [f"features.{i}"]) for i in range(8, -1, -1)],
}
TRAINABLE = (
    "resnet18_scratch_64",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def reinit(model: nn.Module, names: list[str], seed: int) -> int:
    torch.manual_seed(seed)
    modules = dict(model.named_modules())
    count = 0
    for name in names:
        for sub in modules[name].modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=Path("var/demos/m5/selection.json"))
    args = ap.parse_args()
    keep_awake()
    runs = json.loads((args.protocol_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    seed0 = {r["model"]: r for r in runs if r["seed"] == 0 and r["status"] == "completed"}
    sel = json.loads(args.selection.read_text(encoding="utf-8"))
    svc = InferenceService(Settings(inference_device="cpu", inference_cache_size=1))
    report: dict[str, Any] = {
        "label": LABEL,
        "gradcam_version": GRADCAM_VERSION,
        "images": [i["relpath"] for i in sel["images"]],
        "models": {},
    }
    for run_name in TRAINABLE:
        exp = seed0[run_name]["experiment_id"]
        lm, _ = svc.get_model(exp)
        layer = target_layer_name(lm.model_name)
        xs = [
            svc._tensor(lm, svc.read_input(DatasetSource(sel["dataset_id"], i["relpath"])))
            for i in sel["images"]
        ]
        trained = lm.model
        with torch.inference_mode():
            targets = [int(trained(x).argmax()) for x in xs]
        base = [grad_cam(trained, x, t, layer) for x, t in zip(xs, targets, strict=True)]
        model = copy.deepcopy(trained).eval()
        stages = []
        for k, (stage, names) in enumerate(STAGES[lm.model_name]):
            n_modules = reinit(model, names, seed=1000 + k)
            model.eval()
            rhos, undefined = [], 0
            for b, x, t in zip(base, xs, targets, strict=True):
                r = grad_cam(model, x, t, layer)
                if b.all_zero or r.all_zero:
                    undefined += 1
                    continue
                rho = float(spearmanr(b.cam.flatten(), r.cam.flatten())[0])
                if math.isnan(rho):
                    undefined += 1
                    continue
                rhos.append(rho)
            mean = sum(rhos) / len(rhos) if rhos else None
            mean_abs = sum(abs(r) for r in rhos) / len(rhos) if rhos else None
            stages.append(
                {
                    "stage": stage,
                    "randomized_so_far": [s for s, _ in STAGES[lm.model_name][: k + 1]],
                    "modules_reset": n_modules,
                    "n_defined": len(rhos),
                    "n_undefined": undefined,
                    "mean_rho": mean,
                    "mean_abs_rho": mean_abs,
                }
            )
            print(
                f"{run_name} stage {k + 1} {stage}: mean rho "
                f"{'n/a' if mean is None else f'{mean:.3f}'}, mean |rho| "
                f"{'n/a' if mean_abs is None else f'{mean_abs:.3f}'} "
                f"({len(rhos)} defined, {undefined} undefined)",
                flush=True,
            )
        final = stages[-1]["mean_abs_rho"]
        report["models"][run_name] = {
            "experiment_id": exp,
            "layer": layer,
            "feature_map_size": list(base[0].feature_map_size),
            "trained_all_zero": sum(b.all_zero for b in base),
            "stages": stages,
            "flag_high_similarity_after_full_randomization": final is not None
            and final > FLAG_ABS_RHO,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cascading_randomization.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Grad-CAM cascading randomization on the seed-0 EuroSAT models - {LABEL}",
        "",
        f"{GRADCAM_VERSION}; 20 seeded VAL images (M5 demo selection); Spearman rho between the "
        "trained CAM and the CAM after each cascading stage (same target class: the trained "
        "model's prediction). Report only; flag if mean |rho| > 0.5 after full randomization.",
        "",
    ]
    for run_name, m in report["models"].items():
        lines += [
            f"## {run_name} ({m['experiment_id']}; layer {m['layer']}; CAM "
            f"{m['feature_map_size'][0]}x{m['feature_map_size'][1]})",
            "",
            "| stage (cumulative from the top) | mean rho | mean abs rho | defined / undefined |",
            "|---|---|---|---|",
        ]
        for s in m["stages"]:
            lines.append(
                f"| {s['stage']} | {_fmt(s['mean_rho'])} | {_fmt(s['mean_abs_rho'])} | "
                f"{s['n_defined']} / {s['n_undefined']} |"
            )
        flag = m["flag_high_similarity_after_full_randomization"]
        lines += ["", f"Flag (mean |rho| > 0.5 after full randomization): **{flag}**", ""]
    (args.out / "cascading_randomization.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'cascading_randomization.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
