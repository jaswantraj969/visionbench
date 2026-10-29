"""Start a real VisionBench backend for the frontend end-to-end tests (Playwright).

Creates a fresh temporary data root with nothing in it, writes a synthetic shapes dataset as a
ZIP (generated here; no download, not EuroSAT) and a small robustness suite, then runs
``visionbench serve`` (API + worker + the built frontend) on 127.0.0.1. The paths are written to
``<root>/e2e.json`` for the test. The test drives the app through the browser; only the
materialization of corrupted images (a CLI step by design) runs from the test's Node side.

Usage:
    python scripts/e2e_backend.py --root <empty dir> --port 8765
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")


def shape_png(rng: np.random.Generator, shape: str, size: int = 32) -> bytes:
    bg = tuple(int(c) for c in rng.integers(0, 256, 3))
    arr = np.asarray(Image.new("RGB", (size, size), bg), dtype=np.float64)
    arr = arr + rng.normal(0, 10, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    r = int(rng.integers(size // 6, size // 3))
    cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, 2))
    fg = tuple(int((b + rng.integers(90, 166)) % 256) for b in bg)
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_zip(path: Path, per_class: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path, "w") as zf:
        for shape in SHAPES:
            for i in range(per_class):
                zf.writestr(f"shapes/{shape}/{shape}_{i:03d}.png", shape_png(rng, shape))


SUITE = """# Small synthetic suite for the end-to-end test (not a benchmark suite).
suite: e2e-c-v1
base_seed: 0
corruptions:
  gaussian_noise:
    parameter: sigma
    severities: [0.05]
  jpeg:
    parameter: quality
    severities: [20]
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--per-class", type=int, default=16)
    args = ap.parse_args()
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} must be empty or not exist")
    (root / "suites" / "e2e" / "robustness").mkdir(parents=True)
    (root / "benchmarks").mkdir()
    data_root = root / "var"
    zip_path = root / "shapes.zip"
    write_zip(zip_path, args.per_class)
    suite_path = root / "suites" / "e2e" / "robustness" / "robustness_suite.yaml"
    suite_path.write_text(SUITE, encoding="utf-8", newline="\n")
    repo = Path(__file__).resolve().parents[1]
    info = {
        "root": str(root),
        "data_root": str(data_root),
        "zip": str(zip_path),
        "suite": str(suite_path),
        "python": sys.executable,
        "port": args.port,
    }
    (root / "e2e.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    env = os.environ | {
        "VISIONBENCH_DATA_ROOT": str(data_root),
        "VISIONBENCH_BENCHMARKS_DIR": str(root / "benchmarks"),
        "VISIONBENCH_ROBUSTNESS_SUITES_DIR": str(root / "suites"),
        "VISIONBENCH_FRONTEND_DIST": str(repo / "frontend" / "dist"),
        "VISIONBENCH_INFERENCE_DEVICE": "cpu",
        "VISIONBENCH_TEST_IMAGES_UNLOCKED": "false",
        "VISIONBENCH_CANCEL_TIMEOUT_SECONDS": "10",
    }
    print(f"e2e backend: data root {data_root}, port {args.port}", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "visionbench.cli", "serve", "--port", str(args.port)],
        env=env,
        cwd=str(repo),
    )


if __name__ == "__main__":
    sys.exit(main())
