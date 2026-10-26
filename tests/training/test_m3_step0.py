"""M3 step 0: configurable torch hub dir, frozen BatchNorm for head_only, norm-stats cache."""

import hashlib
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models as tv_models

from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json
from visionbench.models import pretrained as pretrained_mod
from visionbench.models.pretrained import file_sha256
from visionbench.models.registry import build_model
from visionbench.training import loop
from visionbench.training.runner import run_training

CLASSES = ["a", "b", "c"]


def _fake_weights(tmp_path: Path, monkeypatch) -> list[Path]:
    """Serve a cached resnet18 state dict without network; record the hub dirs used."""
    torch.manual_seed(0)
    state = tv_models.resnet18(weights=None).state_dict()
    tmp = tmp_path / "w.pth"
    torch.save(state, tmp)
    path = tmp_path / "fakehub" / f"resnet18-{file_sha256(tmp)[:8]}.pth"
    path.parent.mkdir(parents=True)
    tmp.rename(path)
    seen: list[Path] = []

    def fake_fetch(weights, hub_dir):
        seen.append(Path(hub_dir))
        return state, path

    monkeypatch.setattr(pretrained_mod, "fetch_state_dict", fake_fetch)
    return seen


def test_torch_home_setting(tmp_path: Path, monkeypatch) -> None:
    s = Settings(_env_file=None, data_root=tmp_path / "var")
    assert s.resolved_torch_home == (tmp_path / "var").resolve() / "cache" / "torch"
    monkeypatch.setenv("VISIONBENCH_TORCH_HOME", str(tmp_path / "hub"))
    s = Settings(_env_file=None, data_root=tmp_path / "var")
    assert s.resolved_torch_home == (tmp_path / "hub").resolve()


def test_runner_uses_and_records_torch_home(train_env, tmp_path: Path, monkeypatch) -> None:
    seen = _fake_weights(tmp_path, monkeypatch)
    settings = train_env.settings.model_copy(update={"torch_home": tmp_path / "custom_hub"})
    cfg = train_env.config(
        model="resnet18", weights="imagenet", input_size=None, epochs=1, max_steps=1
    )
    result = run_training(cfg, settings)
    resolved = read_json(result.run_dir / "config.json")["resolved"]
    assert seen == [(tmp_path / "custom_hub").resolve()]
    assert resolved["torch_hub_dir"] == str((tmp_path / "custom_hub").resolve())


def _bn_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.clone()
        for k, v in model.state_dict().items()
        if "running_" in k or "num_batches_tracked" in k
    }


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 3, 64, 64, generator=g)
    y = torch.randint(0, 3, (16,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.mark.parametrize("finetune", ["head_only", "full"])
def test_head_only_keeps_backbone_bn_frozen(tmp_path: Path, monkeypatch, finetune: str) -> None:
    _fake_weights(tmp_path, monkeypatch)
    built = build_model(
        "resnet18", CLASSES, weights="imagenet", finetune=finetune, hub_dir=tmp_path
    )
    model = built.model
    before = _bn_state(model)
    head_before = model.fc.weight.clone()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loop.train_one_epoch(model, _loader(), torch.nn.CrossEntropyLoss(), opt, torch.device("cpu"))
    after = _bn_state(model)
    unchanged = all(torch.equal(before[k], after[k]) for k in before)
    assert not torch.equal(head_before, model.fc.weight)  # the head did train
    if finetune == "head_only":
        assert unchanged and len(loop.frozen_batchnorm_layers(model)) == 20
    else:  # control: without freezing, running statistics move
        assert not unchanged and loop.frozen_batchnorm_layers(model) == []


def test_head_only_run_records_frozen_bn(train_env, tmp_path: Path, monkeypatch) -> None:
    _fake_weights(tmp_path, monkeypatch)
    cfg = train_env.config(
        model="resnet18",
        weights="imagenet",
        finetune="head_only",
        input_size=None,
        epochs=1,
        max_steps=1,
    )
    result = run_training(cfg, train_env.settings)
    resolved = read_json(result.run_dir / "config.json")["resolved"]
    assert resolved["frozen_bn"] is True and resolved["frozen_bn_layers"] == 20


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_split_directory_untouched_by_training(train_env) -> None:
    before = _tree_hashes(train_env.split_dir)
    result = run_training(train_env.config(epochs=1), train_env.settings)
    assert _tree_hashes(train_env.split_dir) == before
    assert sorted(before) == ["manifest.json", "split_info.json"]
    norm = read_json(result.run_dir / "config.json")["resolved"]["normalization"]
    expected = (
        f"cache/norm_stats/{train_env.dataset_id}/{train_env.split_id}/norm_stats_train_32px.json"
    )
    assert norm["cache_file"] == expected
    assert (train_env.settings.resolved_data_root / expected).is_file()
