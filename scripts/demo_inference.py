"""M5 demo: predict + Grad-CAM on seeded VALIDATION images through a live ``visionbench serve``.

Everything here is labelled "VALIDATION images - demo, not results": the images come from the
val split, which chose each checkpoint, and 20 images are far too few to say anything about a
model.

- Image selection is seeded and not hand-picked: for every class, ``--per-class`` val entries of
  the frozen split are drawn with ``numpy.random.default_rng(--seed)`` from the class's entries
  sorted by relpath. The list is written to ``selection.json`` before any request is sent.
- Models: seed 0 of each trainable protocol model (``runs.json``; the protocol's UI/demo rule).
- Requests go through the HTTP API (CPU inference by default). The first request for a model is
  "cold" (it loads the model); the rest are "warm" (cache hit).
- Overlays are saved under ``--out`` (inside ``var/``, never committed) with an attribution
  footer added for sharing (DATASETS.md).
- One synthetic, non-EuroSAT image is uploaded to show the out-of-distribution warning.

Usage (with ``visionbench serve --no-worker`` running on a fresh process, so the cache is cold):
    python scripts/demo_inference.py benchmarks/eurosat-v1 --out var/demos/m5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL = "VALIDATION images - demo, not results"
ATTRIBUTION = [
    "Contains modified Copernicus Sentinel data (acquisition year not stated by the dataset",
    "authors). EuroSAT: Helber et al. 2019 (MIT). " + LABEL + ".",
]
TRAINABLE = (
    "resnet18_scratch_64",
    "small_cnn_scratch_64",
    "resnet18_imagenet_224",
    "efficientnet_b0_imagenet_224",
)
HEADERS = {"X-VisionBench-Request": "1"}


def select_images(manifest: dict[str, Any], per_class: int, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    out = []
    for label in sorted(manifest["counts"]):
        entries = sorted(
            (e for e in manifest["entries"] if e["split"] == "val" and e["label"] == label),
            key=lambda e: e["relpath"],
        )
        for i in sorted(rng.choice(len(entries), size=per_class, replace=False).tolist()):
            out.append({"relpath": entries[i]["relpath"], "label": label})
    return out


def with_footer(png: bytes, lines: list[str]) -> Image.Image:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(img.width, *(int(measure.textlength(s, font=font)) + 12 for s in lines))
    out = Image.new("RGB", (width, img.height + 14 * len(lines) + 6), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        draw.text((6, img.height + 3 + 14 * i), line, fill=(60, 60, 60), font=font)
    return out


def synthetic_ood_image(seed: int = 0) -> bytes:
    """A generated 300x200 image with shapes and gradients: not a satellite tile."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:200, 0:300] / 200.0
    arr = np.stack([120 + 100 * np.sin(4 * xx), 80 + 120 * yy, 200 - 90 * xx], -1)
    img = Image.fromarray(np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 40, 130, 140], fill=(250, 220, 30))
    draw.rectangle([180, 60, 270, 170], fill=(30, 30, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post(client: httpx.Client, url: str, **kw: Any) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    r = client.post(url, headers=HEADERS, **kw)
    wall = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise SystemExit(f"{url}: HTTP {r.status_code}: {r.text}")
    return r.json(), wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("protocol_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--data-root", type=Path, default=Path("var"))
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = json.loads((args.protocol_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    seed0 = {r["model"]: r for r in runs if r["seed"] == 0 and r["status"] == "completed"}
    first = seed0[TRAINABLE[0]]
    exp_cfg = json.loads(
        (args.data_root / "experiments" / first["experiment_id"] / "config.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    dataset_id, split_id = exp_cfg["dataset_id"], exp_cfg["split_id"]
    manifest_path = args.data_root / "datasets" / dataset_id / "splits" / split_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = select_images(manifest, args.per_class, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(
        json.dumps(
            {
                "label": LABEL,
                "rule": f"numpy default_rng({args.seed}); {args.per_class} val entries per class "
                "drawn without replacement from the class's val entries sorted by relpath",
                "dataset_id": dataset_id,
                "split_id": split_id,
                "images": selection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows, timing = [], {}
    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        for model in TRAINABLE:
            exp = seed0[model]["experiment_id"]
            preds, cams = [], []
            for img in selection:
                ref = {"dataset_id": dataset_id, "relpath": img["relpath"]}
                p, p_wall = post(client, f"/api/experiments/{exp}/predict", json=ref)
                g, g_wall = post(client, f"/api/experiments/{exp}/gradcam", json=ref)
                preds.append((p, p_wall))
                cams.append((g, g_wall))
                name = img["relpath"].replace("/", "__").rsplit(".", 1)[0] + ".png"
                path = args.out / "overlays" / model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                png = base64.b64decode(g["overlay"]["png"].split(",", 1)[1])
                with_footer(png, ATTRIBUTION).save(path)
                rows.append(
                    {
                        "model": model,
                        "experiment_id": exp,
                        "image": img["relpath"],
                        "true_class": img["label"],
                        "predicted": p["predicted_class"],
                        "confidence": p["probabilities"][0]["probability"],
                        "correct": p["predicted_class"] == img["label"],
                        "cam_resolution": g["cam"]["resolution"],
                        "cam_all_zero": g["cam"]["all_zero"],
                        "unseen_fraction": g["geometry"]["unseen_fraction"],
                        "warnings": p["warnings"] + g["warnings"],
                        "overlay": path.as_posix(),
                    }
                )
            first_p, first_g = preds[0][0], cams[0][0]
            assert first_p["cache"] == "miss", f"{model}: model was already cached (not cold)"
            timing[model] = {
                "experiment_id": exp,
                "device": first_p["device"],
                "cold": {
                    "model_load_ms": first_p["timing_ms"]["model_load"],
                    "predict_total_ms": first_p["timing_ms"]["total"],
                    "predict_forward_ms": first_p["timing_ms"]["forward"],
                    "predict_http_ms": preds[0][1],
                    "gradcam_total_ms": first_g["timing_ms"]["total"],
                    "gradcam_http_ms": cams[0][1],
                },
                "warm_median": {
                    "predict_total_ms": statistics.median(
                        p["timing_ms"]["total"] for p, _ in preds[1:]
                    ),
                    "predict_forward_ms": statistics.median(
                        p["timing_ms"]["forward"] for p, _ in preds[1:]
                    ),
                    "predict_http_ms": statistics.median(w for _, w in preds[1:]),
                    "gradcam_total_ms": statistics.median(
                        g["timing_ms"]["total"] for g, _ in cams[1:]
                    ),
                    "gradcam_cam_ms": statistics.median(
                        g["timing_ms"]["gradcam"] for g, _ in cams[1:]
                    ),
                    "gradcam_http_ms": statistics.median(w for _, w in cams[1:]),
                    "n": len(preds) - 1,
                },
            }
            print(f"{model}: {json.dumps(timing[model])}", flush=True)

        ood_exp = seed0["resnet18_imagenet_224"]["experiment_id"]
        files = {"file": ("synthetic.png", synthetic_ood_image(), "image/png")}
        ood_p, _ = post(client, f"/api/experiments/{ood_exp}/predict", files=files)
        ood_g, _ = post(client, f"/api/experiments/{ood_exp}/gradcam", files=files)
        ood_path = args.out / "overlays" / "ood_upload_resnet18_imagenet_224.png"
        ood_png = base64.b64decode(ood_g["overlay"]["png"].split(",", 1)[1])
        with_footer(ood_png, ["Synthetic generated image (not EuroSAT) - demo, not results."]).save(
            ood_path
        )
        ood = {
            "experiment_id": ood_exp,
            "original_size": ood_p["input"]["original_size"],
            "predicted": ood_p["predicted_class"],
            "confidence": ood_p["probabilities"][0]["probability"],
            "warnings": ood_p["warnings"],
            "notes": ood_p["notes"],
            "gradcam_unseen_fraction": ood_g["geometry"]["unseen_fraction"],
            "overlay": ood_path.as_posix(),
        }
        print("OOD upload:", json.dumps(ood), flush=True)

    result = {"label": LABEL, "selection": selection, "rows": rows, "timing": timing, "ood": ood}
    (args.out / "demo_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# M5 demo - {LABEL}",
        "",
        "| model | image | true | predicted | confidence | CAM | unseen fraction |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['image']} | {r['true_class']} | {r['predicted']} | "
            f"{r['confidence']:.3f} | {r['cam_resolution']} | {r['unseen_fraction']:.4f} |"
        )
    (args.out / "demo_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    correct = sum(r["correct"] for r in rows)
    print(f"{len(rows)} predict+gradcam pairs; {correct} predictions equal the label ({LABEL})")
    print(f"wrote {args.out / 'demo_results.json'} and {args.out / 'demo_table.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
