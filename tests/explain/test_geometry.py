"""Grad-CAM geometry: the val transform's mapping and its inverse, checked against torchvision."""

import numpy as np
import pytest
import torch
from PIL import Image

from visionbench.explain.geometry import ValGeometry
from visionbench.training.config import AugmentationConfig
from visionbench.training.preprocessing import Preprocessing, build_transforms

IDENTITY = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # Normalize is a no-op: tensors stay in [0, 1]
SCRATCH_64 = Preprocessing("train_split", 64, None, "bilinear", *IDENTITY)
CROP_224 = Preprocessing("imagenet_weights", 224, 256, "bilinear", *IDENTITY)
CASES = [
    (SCRATCH_64, 64, 64),
    (SCRATCH_64, 100, 60),
    (CROP_224, 64, 64),
    (CROP_224, 100, 60),
    (CROP_224, 60, 100),
]


def _val(prep: Preprocessing):
    return build_transforms(prep, AugmentationConfig())[1]


def _marker_image(w: int, h: int, x: int, y: int, r: int = 1) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = 255
    return Image.fromarray(arr)


@pytest.mark.parametrize(("prep", "w", "h"), CASES)
def test_round_trip_original_input_original(prep: Preprocessing, w: int, h: int) -> None:
    g = ValGeometry.from_preprocessing(prep, w, h)
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0, w, 200), rng.uniform(0, h, 200)
    bx, by = g.to_original(*g.to_input(xs, ys))
    assert np.allclose(bx, xs, atol=1e-9) and np.allclose(by, ys, atol=1e-9)
    xi, yi = rng.uniform(0, g.input_size, 200), rng.uniform(0, g.input_size, 200)
    fx, fy = g.to_input(*g.to_original(xi, yi))
    assert np.allclose(fx, xi, atol=1e-9) and np.allclose(fy, yi, atol=1e-9)


@pytest.mark.parametrize(("prep", "w", "h"), CASES)
def test_mapping_matches_the_real_val_transform(prep: Preprocessing, w: int, h: int) -> None:
    """A marker in the original lands where the geometry says, in the tensor the model gets."""
    g = ValGeometry.from_preprocessing(prep, w, h)
    if prep.resize is not None:
        resized = _val(prep).transforms[0](Image.new("RGB", (w, h)))
        assert resized.size == g.resized_size
    box = g.seen_box()
    for fx, fy in ((0.3, 0.4), (0.5, 0.5), (0.7, 0.65)):
        x = int(box["left"] + fx * (box["right"] - box["left"]))
        y = int(box["top"] + fy * (box["bottom"] - box["top"]))
        t = _val(prep)(_marker_image(w, h, x, y))[0]  # one channel, s x s
        assert t.shape == (g.input_size, g.input_size)
        yy, xx = torch.meshgrid(
            torch.arange(t.shape[0]) + 0.5, torch.arange(t.shape[1]) + 0.5, indexing="ij"
        )
        mass = t.clamp_min(0).double()
        cx, cy = float((mass * xx).sum() / mass.sum()), float((mass * yy).sum() / mass.sum())
        ex, ey = g.to_input(x + 0.5, y + 0.5)  # centre of the marker's centre pixel
        assert abs(cx - float(ex)) < 1.0 and abs(cy - float(ey)) < 1.0, (cx, cy, ex, ey)


def test_border_marker_is_cropped_and_reported_unseen() -> None:
    w = h = 64
    g = ValGeometry.from_preprocessing(CROP_224, w, h)
    assert g.resized_size == (256, 256) and g.crop_offset == (16, 16)
    assert g.seen_box() == {"left": 4.0, "top": 4.0, "right": 60.0, "bottom": 60.0}
    assert g.unseen_fraction() == pytest.approx(1 - 56 * 56 / (64 * 64))  # 0.234375
    seen = g.seen_mask(w, h)
    assert not seen[1, 1] and not seen[62, 30] and seen[4, 4] and seen[32, 32]
    t = _val(CROP_224)(_marker_image(w, h, 1, 1, r=1))  # marker on pixels 0..2: all unseen
    assert float(t.max()) == 0.0  # the model never receives it
    inside = _val(CROP_224)(_marker_image(w, h, 10, 10, r=1))
    assert float(inside.max()) > 0.5
    scratch = ValGeometry.from_preprocessing(SCRATCH_64, w, h)
    assert scratch.unseen_fraction() == 0.0 and scratch.seen_mask(w, h).all()
    assert float(_val(SCRATCH_64)(_marker_image(w, h, 1, 1))[0].max()) > 0.5


@pytest.mark.parametrize(("prep", "w", "h"), CASES)
def test_cam_cell_maps_to_the_right_place_in_the_original(prep, w, h) -> None:
    g = ValGeometry.from_preprocessing(prep, w, h)
    n = 7
    cam = torch.zeros(n, n, dtype=torch.float64)
    row, col = 3, 4
    cam[row, col] = 1.0
    out = g.map_to_original(cam, w, h)
    ox, oy = g.to_original((col + 0.5) * g.input_size / n, (row + 0.5) * g.input_size / n)
    peak = np.unravel_index(np.nanargmax(out), out.shape)
    assert abs(peak[1] + 0.5 - float(ox)) <= 1.0 and abs(peak[0] + 0.5 - float(oy)) <= 1.0
    assert np.isnan(out).mean() == pytest.approx(g.unseen_fraction())
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0
