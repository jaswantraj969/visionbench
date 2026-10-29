"""Grad-CAM sanity checks on a tiny model trained on a synthetic localization task.

Task: 64x64 noisy grey images with one coloured disk (red, green or blue: the class) in a random
quadrant, plus a white square distractor in a different quadrant that is present in every class.
The class is decided ONLY by the disk, whose quadrant is known, so a working Grad-CAM must put
its peak in the disk's quadrant.

(A first design made the class the disk's QUADRANT. SmallCNN ends in global average pooling, so
it can read position only from padding effects at the borders: it reached 100 % accuracy but its
evidence is spread over the map, and the peak hit the quadrant for 24/80 images, i.e. chance. The
Grad-CAM values were identical to an independent backward-hook implementation, see below. That
design tests the architecture, not the explanation, and was replaced; the thresholds were not.)

Thresholds (fixed before running, documented in EXPERIMENT_PROTOCOL.md):
- localization: Grad-CAM peak in the disk's quadrant for >= 80 % of held-out images;
- target sensitivity: mean Pearson correlation between the CAM of the true class and the CAMs
  of the other classes < 0.5;
- model randomization (Adebayo et al., 2018): mean |Spearman rho| between the CAM of the
  trained model and of the same architecture with re-initialised weights < 0.5.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

from visionbench.explain.gradcam import ExplanationNotSupported, grad_cam, target_layer_name
from visionbench.models.small_cnn import SmallCNN

SIZE = 64
COLOURS = ((0.9, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.2, 0.9))  # class = disk colour
CLASSES = len(COLOURS)
LAYER = SmallCNN.TARGET_LAYER  # "features.block4.relu": post-ReLU, pre-pool, 8x8 at 64 px
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))  # (row half, column half)


def _sample(rng: np.random.Generator, cls: int) -> tuple[np.ndarray, int]:
    """(3 x 64 x 64 image in [0, 1], quadrant index of the disk)."""
    grey = rng.uniform(0.35, 0.55)
    img = np.repeat((grey + rng.normal(0, 0.06, (SIZE, SIZE)))[None], 3, axis=0)
    half = SIZE // 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    def centre(q: tuple[int, int], margin: int) -> tuple[int, int]:
        return (
            int(rng.integers(q[0] * half + margin, (q[0] + 1) * half - margin)),
            int(rng.integers(q[1] * half + margin, (q[1] + 1) * half - margin)),
        )

    quadrant = int(rng.integers(len(QUADRANTS)))
    cy, cx = centre(QUADRANTS[quadrant], 9)
    r = int(rng.integers(5, 9))
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    for c in range(3):
        img[c][disk] = COLOURS[cls][c]
    other = QUADRANTS[int(rng.choice([q for q in range(len(QUADRANTS)) if q != quadrant]))]
    sy, sx = centre(other, 9)
    s = int(rng.integers(4, 8))
    img[:, sy - s : sy + s, sx - s : sx + s] = 1.0
    return np.clip(img, 0, 1), quadrant


def _dataset(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rng = np.random.default_rng(seed)
    xs, ys, quads = [], [], []
    for i in range(n_per_class * CLASSES):
        cls = i % CLASSES
        img, q = _sample(rng, cls)
        xs.append(img)
        ys.append(cls)
        quads.append(q)
    return torch.tensor(np.stack(xs), dtype=torch.float32) - 0.5, torch.tensor(ys), quads


@pytest.fixture(scope="module")
def quadrant_model() -> dict:
    torch.manual_seed(0)
    x_train, y_train, _ = _dataset(120, seed=1)
    x_test, y_test, q_test = _dataset(25, seed=2)
    model = SmallCNN(CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(8):
        model.train()
        for idx in torch.randperm(len(x_train), generator=gen).split(32):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x_train[idx]), y_train[idx]).backward()
            opt.step()
    model.eval()
    for p in model.parameters():
        p.grad = None  # training leftovers; the hook test checks nothing is written later
    with torch.no_grad():
        acc = float((model(x_test).argmax(1) == y_test).float().mean())
    return {"model": model, "x": x_test, "y": y_test, "quadrant": q_test, "accuracy": acc}


def _peak_quadrant(cam: torch.Tensor) -> tuple[int, int]:
    h, w = cam.shape
    r, c = divmod(int(cam.argmax()), w)
    return int(r >= h // 2), int(c >= w // 2)


def test_localization_peak_in_the_disk_quadrant(quadrant_model, report_line) -> None:
    m = quadrant_model
    assert m["accuracy"] >= 0.9, f"precondition: the model must learn the task ({m['accuracy']})"
    hits = []
    for x, y, q in zip(m["x"], m["y"], m["quadrant"], strict=True):
        res = grad_cam(m["model"], x[None], int(y), LAYER)
        assert res.feature_map_size == (8, 8)
        hits.append(not res.all_zero and _peak_quadrant(res.cam) == QUADRANTS[q])
    rate = float(np.mean(hits))
    report_line(
        f"GRADCAM localization: peak in the disk's quadrant for {sum(hits)}/{len(hits)} held-out "
        f"images ({rate:.1%}); model accuracy {m['accuracy']:.1%}"
    )
    assert rate >= 0.8


def test_matches_an_independent_backward_hook_implementation(quadrant_model) -> None:
    """Same CAM as the textbook version: a backward hook on the layer's output tensor plus
    ``.backward()`` on a copy of the model. (A module ``register_full_backward_hook`` cannot be
    used since M6 step 0: PyTorch forbids it on the inplace ReLU that is now the target.)"""
    for i in range(5):
        x, y = quadrant_model["x"][i : i + 1], int(quadrant_model["y"][i])
        model = copy.deepcopy(quadrant_model["model"])
        layer = dict(model.named_modules())[LAYER]
        seen: dict[str, torch.Tensor] = {}

        def forward_hook(_m, _i, out, s=seen):
            s["a"] = out
            out.register_hook(lambda g, s=s: s.__setitem__("g", g))

        h1 = layer.register_forward_hook(forward_hook)
        try:
            model(x)[0, y].backward()
        finally:
            h1.remove()
        weights = seen["g"].mean(dim=(2, 3), keepdim=True)
        reference = torch.relu((weights * seen["a"]).sum(dim=1))[0].detach().double()
        ours = grad_cam(quadrant_model["model"], x, y, LAYER)
        assert torch.allclose(ours.cam, reference, rtol=1e-6, atol=1e-9)


def test_small_cnn_target_is_the_post_relu_pre_pool_map(quadrant_model) -> None:
    """The hooked tensor is ReLU(BN(conv(x))) of block4 (inplace ReLU), before the 2x2 pool."""
    model, x = quadrant_model["model"], quadrant_model["x"][:1]
    block4 = model.features.block4
    seen: list[torch.Tensor] = []
    handle = dict(model.named_modules())[LAYER].register_forward_hook(
        lambda _m, _i, out: seen.append(out.detach().clone())
    )
    try:
        with torch.no_grad():
            model(x)
            pre = model.features[:3](x)  # blocks 1-3
            expected = torch.relu(block4.bn(block4.conv(pre)))
    finally:
        handle.remove()
    assert seen[0].shape == (1, 256, 8, 8) and float(seen[0].min()) >= 0.0
    assert torch.equal(seen[0], expected)
    res = grad_cam(model, x, 0, LAYER)
    assert res.feature_map_size == (8, 8)


def test_target_sensitivity_cams_differ_between_classes(quadrant_model, report_line) -> None:
    m = quadrant_model
    corrs, skipped = [], 0
    for x, y in zip(m["x"], m["y"], strict=True):
        true = grad_cam(m["model"], x[None], int(y), LAYER)
        for other in range(CLASSES):
            if other == int(y):
                continue
            res = grad_cam(m["model"], x[None], other, LAYER)
            assert not torch.equal(res.cam, true.cam)
            if res.all_zero or true.all_zero:
                skipped += 1  # constant map: correlation undefined
                continue
            corrs.append(pearsonr(true.cam.flatten(), res.cam.flatten())[0])
    mean = float(np.mean(corrs))
    report_line(
        f"GRADCAM target sensitivity: mean Pearson r(true-class CAM, other-class CAM) = "
        f"{mean:.3f} over {len(corrs)} pairs ({skipped} pairs with an all-zero CAM skipped)"
    )
    assert mean < 0.5


def test_model_randomization_changes_the_cam(quadrant_model, report_line) -> None:
    m = quadrant_model
    torch.manual_seed(123)
    random_model = SmallCNN(CLASSES).eval()  # same architecture, re-initialised weights
    rhos, undefined = [], 0
    for x, y in list(zip(m["x"], m["y"], strict=True))[:8]:
        trained = grad_cam(m["model"], x[None], int(y), LAYER)
        rand = grad_cam(random_model, x[None], int(y), LAYER)
        if trained.all_zero or rand.all_zero:
            undefined += 1
            continue
        rhos.append(float(spearmanr(trained.cam.flatten(), rand.cam.flatten())[0]))
    report_line(
        "GRADCAM model randomization: Spearman rho(trained CAM, random-weights CAM) per image = "
        + ", ".join(f"{r:.3f}" for r in rhos)
        + f"; mean |rho| = {np.mean(np.abs(rhos)):.3f}; {undefined} image(s) with an all-zero CAM"
    )
    assert len(rhos) >= 5
    assert float(np.mean(np.abs(rhos))) < 0.5


def test_hooks_removed_after_success_and_after_errors(quadrant_model) -> None:
    model, x = quadrant_model["model"], quadrant_model["x"][:1]
    layer = dict(model.named_modules())[LAYER]
    for _ in range(100):
        grad_cam(model, x, 0, LAYER)
    for _ in range(100):
        with pytest.raises(IndexError):
            grad_cam(model, x, 99, LAYER)  # fails after the hook fired
    for module in model.modules():
        assert not module._forward_hooks and not module._backward_hooks
        assert not module._forward_pre_hooks
    assert len(layer._forward_hooks) == 0
    assert all(p.grad is None for p in model.parameters())  # nothing written into the model


def test_requires_eval_mode_and_supported_model(quadrant_model) -> None:
    model = quadrant_model["model"]
    model.train()
    try:
        with pytest.raises(ValueError, match="eval mode"):
            grad_cam(model, quadrant_model["x"][:1], 0, LAYER)
    finally:
        model.eval()
    with pytest.raises(ExplanationNotSupported):
        target_layer_name("majority_baseline")
    with pytest.raises(ExplanationNotSupported):
        grad_cam(model, quadrant_model["x"][:1], 0, "no.such.layer")


class _NegativeEvidence(nn.Module):
    """Positive activations; class 1's logit decreases with every one of them."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        self.head = nn.Linear(1, 2)
        with torch.no_grad():
            self.conv.weight.fill_(0.1)
            self.conv.bias.fill_(1.0)
            self.head.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def test_all_zero_cam_is_reported_not_rescaled() -> None:
    model = _NegativeEvidence().eval()
    x = torch.rand(1, 3, 16, 16)
    neg = grad_cam(model, x, 1, "conv")
    assert neg.all_zero and float(neg.cam.max()) == 0.0 and float(neg.normalized.max()) == 0.0
    pos = grad_cam(model, x, 0, "conv")
    assert not pos.all_zero and float(pos.normalized.max()) == pytest.approx(1.0)
