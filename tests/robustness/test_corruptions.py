"""The four corruptions on synthetic images: severity behaviour, dtype/shape/range, identity."""

from __future__ import annotations

import numpy as np
import pytest

from visionbench.robustness import corruptions as C
from visionbench.robustness.suite import image_seed


def _textured(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 216, (size, size, 3)).astype(np.uint8)
    return base


def _hf_energy(img: np.ndarray) -> float:
    x = img.astype(np.float64)
    lap = 4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]
    return float((lap**2).mean())


@pytest.mark.parametrize(
    ("name", "severity"),
    [("gaussian_noise", 0.08), ("gaussian_blur", 1.0), ("brightness", 0.2), ("jpeg", 25)],
)
def test_dtype_shape_range(name: str, severity: float) -> None:
    img = _textured()
    out = C.apply(img, name, severity, seed=1)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 255
    assert not np.array_equal(out, img)


def test_noise_variance_matches_sigma() -> None:
    img = np.full((256, 256, 3), 128, dtype=np.uint8)  # mid grey: no clipping at these sigmas
    for sigma in (0.04, 0.08, 0.12):
        out = C.gaussian_noise(img, sigma, np.random.default_rng(0))
        resid = out.astype(np.float64) / 255 - 128 / 255
        # quantization adds variance (1/255)^2/12, negligible here
        assert resid.var() == pytest.approx(sigma**2, rel=0.03)
        assert abs(resid.mean()) < 0.002


def test_blur_reduces_high_frequency_energy_monotonically() -> None:
    img = _textured()
    energies = [_hf_energy(img)] + [_hf_energy(C.gaussian_blur(img, s)) for s in (0.5, 1.0, 1.5)]
    assert all(a > b for a, b in zip(energies, energies[1:], strict=False)), energies
    k = C.gaussian_kernel(1.5)
    assert len(k) == 2 * 5 + 1 and k.sum() == pytest.approx(1.0)  # radius ceil(3 * 1.5) = 5
    flat = np.full((16, 16, 3), 77, dtype=np.uint8)
    assert np.array_equal(C.gaussian_blur(flat, 1.5), flat)  # reflect padding: no edge darkening


def test_brightness_raises_v_and_keeps_hue_order() -> None:
    img = (_textured().astype(np.float64) * 0.6).astype(np.uint8)  # V <= 0.51: no clipping
    v0 = img.max(axis=2).astype(np.float64) / 255
    means = []
    for d in (0.1, 0.2, 0.3):
        out = C.brightness(img, d)
        v = out.max(axis=2).astype(np.float64) / 255
        assert (v - v0).mean() == pytest.approx(d, abs=0.003)
        assert np.array_equal(np.argmax(out, axis=2)[v0 > 0.1], np.argmax(img, axis=2)[v0 > 0.1])
        means.append(v.mean())
    assert means[0] < means[1] < means[2]
    white = np.full((4, 4, 3), 250, dtype=np.uint8)
    assert C.brightness(white, 0.3).max() == 255  # clipped to V = 1
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    assert (C.brightness(black, 0.2) == 51).all()  # V = 0 -> grey V'


def test_jpeg_quality_changes_bytes_and_error() -> None:
    img = _textured()
    outs = {q: C.jpeg(img, q) for q in (50, 25, 10)}
    errs = [float(((outs[q].astype(float) - img) ** 2).mean()) for q in (50, 25, 10)]
    assert errs[0] < errs[1] < errs[2]
    assert not np.array_equal(outs[50], outs[25]) and not np.array_equal(outs[25], outs[10])
    assert np.array_equal(C.jpeg(img, 25), outs[25])  # deterministic


def test_identity_when_severity_is_effectively_zero() -> None:
    img = _textured()
    assert np.array_equal(C.gaussian_noise(img, 1e-9, np.random.default_rng(0)), img)
    assert np.array_equal(C.gaussian_blur(img, 1e-3), img)
    assert np.array_equal(C.brightness(img, 0.0), img)


def test_seeds_differ_by_relpath_corruption_and_severity() -> None:
    a = image_seed("eurosat-c-v1", "gaussian_noise", "0.04", 0, "A/a.jpg")
    assert a == image_seed("eurosat-c-v1", "gaussian_noise", "0.04", 0, "A/a.jpg")
    others = {
        image_seed("eurosat-c-v1", "gaussian_noise", "0.04", 0, "A/b.jpg"),
        image_seed("eurosat-c-v1", "gaussian_noise", "0.08", 0, "A/a.jpg"),
        image_seed("eurosat-c-v1", "gaussian_blur", "0.04", 0, "A/a.jpg"),
        image_seed("eurosat-c-v1", "gaussian_noise", "0.04", 1, "A/a.jpg"),
    }
    assert a not in others and len(others) == 4
    assert 0 <= a < 2**64


def test_noise_is_independent_across_relpaths_and_severities() -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    def resid(rel: str, sev: str, sigma: float) -> np.ndarray:
        seed = image_seed("eurosat-c-v1", "gaussian_noise", sev, 0, rel)
        out = C.gaussian_noise(img, sigma, np.random.default_rng(seed)).astype(float)
        return (out - 128).ravel() / sigma

    r1, r2 = resid("A/a.jpg", "0.08", 0.08), resid("A/b.jpg", "0.08", 0.08)
    r3 = resid("A/a.jpg", "0.04", 0.04)
    assert abs(np.corrcoef(r1, r2)[0, 1]) < 0.05
    assert abs(np.corrcoef(r1, r3)[0, 1]) < 0.05
