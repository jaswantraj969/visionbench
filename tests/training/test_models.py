from pathlib import Path

import pytest
import torch
from torchvision import models as tv_models

from visionbench.models import pretrained as pretrained_mod
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import WeightsUnavailableError, file_sha256
from visionbench.models.registry import ModelConfigError, build_model

CLASSES = ["a", "b", "c"]


@pytest.mark.parametrize(
    ("name", "size", "target"),
    [
        ("small_cnn", 64, "features.block4.conv"),
        ("resnet18", 224, "layer4"),
        ("efficientnet_b0", 224, "features.8"),
    ],
)
def test_registry_builds_scratch_models(name: str, size: int, target: str) -> None:
    torch.manual_seed(0)
    built = build_model(name, CLASSES)
    assert built.input_size == size and built.weights == "scratch"
    assert built.normalization_source == "train_split" and built.target_layer == target
    assert target in dict(built.model.named_modules())
    assert built.param_count == built.trainable_param_count > 0
    out = built.model.eval()(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_registry_baseline_and_errors() -> None:
    built = build_model("majority_baseline", CLASSES)
    assert isinstance(built.model, MajorityBaseline) and built.input_size is None
    with pytest.raises(ModelConfigError, match="unknown model"):
        build_model("vit", CLASSES)
    with pytest.raises(ModelConfigError, match="at least 2"):
        build_model("small_cnn", ["only"])
    with pytest.raises(ModelConfigError, match="small_cnn only supports"):
        build_model("small_cnn", CLASSES, weights="imagenet")
    with pytest.raises(ModelConfigError, match="head_only"):
        build_model("resnet18", CLASSES, finetune="head_only")
    with pytest.raises(ModelConfigError, match="hub_dir"):
        build_model("resnet18", CLASSES, weights="imagenet")


def test_majority_baseline() -> None:
    model = MajorityBaseline(CLASSES).fit({"a": 2, "b": 5, "c": 3})
    assert model.priors == [0.2, 0.5, 0.3] and model.majority_index == 1
    assert model.predict(3) == [1, 1, 1] and model.predict_proba(1) == [[0.2, 0.5, 0.3]]
    assert model.nll(1) == pytest.approx(-torch.log(torch.tensor(0.5)).item())
    again = MajorityBaseline.from_dict(model.to_dict())
    assert again.priors == model.priors and again.to_dict()["majority_class"] == "b"
    tie = MajorityBaseline(CLASSES).fit({"a": 1, "b": 1, "c": 0})
    assert tie.majority_index == 0 and tie.nll(2) > 20  # floor instead of log(0)
    with pytest.raises(ValueError):
        MajorityBaseline(CLASSES).fit({"z": 1})
    with pytest.raises(RuntimeError):
        MajorityBaseline(CLASSES).predict(1)


def _fake_cached_weights(tmp_path: Path, monkeypatch, *, good_hash: bool) -> Path:
    """Simulate a cached torchvision weight file without any network access."""
    state = tv_models.resnet18(weights=None).state_dict()
    tmp = tmp_path / "tmp.pth"
    torch.save(state, tmp)
    sha = file_sha256(tmp)
    prefix = sha[:8] if good_hash else "deadbeef"
    path = tmp_path / "hub" / "checkpoints" / f"resnet18-{prefix}.pth"
    path.parent.mkdir(parents=True)
    tmp.rename(path)
    monkeypatch.setattr(pretrained_mod, "fetch_state_dict", lambda weights, hub_dir: (state, path))
    return path


def test_pretrained_load_path_records_sha_and_preprocessing(tmp_path: Path, monkeypatch) -> None:
    path = _fake_cached_weights(tmp_path, monkeypatch, good_hash=True)
    built = build_model("resnet18", CLASSES, weights="imagenet", hub_dir=tmp_path / "hub")
    info = built.pretrained
    assert built.weights == "imagenet" and built.normalization_source == "imagenet_weights"
    assert info.sha256 == file_sha256(path) and info.enum_name == "ResNet18_Weights.IMAGENET1K_V1"
    assert (info.resize, info.crop, info.interpolation) == (256, 224, "bilinear")
    assert info.mean == (0.485, 0.456, 0.406) and built.input_size == 224
    with pytest.raises(ModelConfigError, match="uses input size 224"):
        build_model(
            "resnet18", CLASSES, weights="imagenet", hub_dir=tmp_path / "hub", input_size=64
        )
    head_only = build_model(
        "resnet18", CLASSES, weights="imagenet", finetune="head_only", hub_dir=tmp_path / "hub"
    )
    assert head_only.trainable_param_count == 512 * 3 + 3 < head_only.param_count


def test_pretrained_hash_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _fake_cached_weights(tmp_path, monkeypatch, good_hash=False)
    with pytest.raises(WeightsUnavailableError, match="does not match the hash prefix"):
        build_model("resnet18", CLASSES, weights="imagenet", hub_dir=tmp_path / "hub")


def test_weight_fallback_requires_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    def offline(weights, hub_dir):
        raise OSError("simulated: no network")

    monkeypatch.setattr(pretrained_mod, "fetch_state_dict", offline)
    with pytest.raises(WeightsUnavailableError, match="--allow-weight-fallback"):
        build_model("efficientnet_b0", CLASSES, weights="imagenet", hub_dir=tmp_path)
    built = build_model(
        "efficientnet_b0",
        CLASSES,
        weights="imagenet",
        hub_dir=tmp_path,
        allow_weight_fallback=True,
        input_size=32,
    )
    assert built.weights == "scratch" and built.pretrained is None
    assert "simulated: no network" in built.fallback_reason
    assert built.normalization_source == "train_split" and built.input_size == 32
    with pytest.raises(ModelConfigError, match="head_only"):
        build_model(
            "efficientnet_b0",
            CLASSES,
            weights="imagenet",
            hub_dir=tmp_path,
            allow_weight_fallback=True,
            finetune="head_only",
        )


def test_small_cnn_overfits_single_batch() -> None:
    torch.manual_seed(0)
    model = build_model("small_cnn", CLASSES).model
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
