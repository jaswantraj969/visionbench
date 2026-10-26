"""The test split must be unreachable from training code."""

import copy
from pathlib import Path

import pytest

from visionbench.training import data as data_mod
from visionbench.training.data import (
    SplitAccessError,
    build_dataloader,
    build_dataset,
    select_entries,
    split_class_counts,
)
from visionbench.training.preprocessing import class_weights_from_train, compute_train_norm_stats
from visionbench.training.runner import run_training


@pytest.mark.parametrize("split", ["test", "TEST", "all", ""])
def test_loader_factory_rejects_non_train_val(train_env, split: str) -> None:
    m = train_env.manifest
    with pytest.raises(SplitAccessError, match="reserved for the final evaluation"):
        build_dataloader(split, m, train_env.raw_root, None, batch_size=4, seed=0)
    with pytest.raises(SplitAccessError):
        build_dataset(split, m, train_env.raw_root, None)
    with pytest.raises(SplitAccessError):
        select_entries(m, split)
    with pytest.raises(SplitAccessError):
        split_class_counts(m, split)


@pytest.mark.parametrize("model", ["small_cnn", "majority_baseline"])
def test_training_never_opens_test_images(train_env, monkeypatch, model: str) -> None:
    opened: list[Path] = []
    real = data_mod.load_image

    def spy(path: Path):
        opened.append(Path(path))
        return real(path)

    monkeypatch.setattr(data_mod, "load_image", spy)
    result = run_training(train_env.config(model=model), train_env.settings)
    assert result.status == "completed"

    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for e in train_env.manifest["entries"]:
        by_split[e["split"]].add(e["relpath"])
    rel = {p.relative_to(train_env.raw_root).as_posix() for p in opened}
    assert by_split["test"], "fixture must have test images"
    assert rel.isdisjoint(by_split["test"])
    assert rel <= by_split["train"] | by_split["val"]
    if model == "small_cnn":
        assert by_split["train"] <= rel and by_split["val"] <= rel
    else:
        assert opened == []  # the baseline needs labels only


def _swap_relpaths(manifest: dict, splits: tuple[str, ...]) -> dict:
    """Copy of ``manifest`` where entries of ``splits`` point at other (val/test) files."""
    m = copy.deepcopy(manifest)
    pool = [e["relpath"] for e in m["entries"] if e["split"] in ("val", "test")]
    targets = [e for e in m["entries"] if e["split"] in splits]
    for e, rel in zip(targets, reversed(pool), strict=False):
        e["relpath"] = rel
    return m


def test_norm_stats_depend_on_train_only(train_env) -> None:
    m = train_env.manifest
    base = compute_train_norm_stats(m, train_env.raw_root, 32)

    changed_eval = _swap_relpaths(m, ("val", "test"))
    for e in changed_eval["entries"]:
        if e["split"] != "train":
            e["relpath"] = "does/not/exist.png"  # never opened
    same = compute_train_norm_stats(changed_eval, train_env.raw_root, 32)
    assert (same["mean"], same["std"]) == (base["mean"], base["std"])

    changed_train = _swap_relpaths(m, ("train",))
    other = compute_train_norm_stats(changed_train, train_env.raw_root, 32)
    assert (other["mean"], other["std"]) != (base["mean"], base["std"])
    assert compute_train_norm_stats(m, train_env.raw_root, 48)["mean"] != base["mean"]


def test_class_weights_from_train_counts_only(train_env) -> None:
    m = train_env.manifest
    base = class_weights_from_train(m)
    counts = split_class_counts(m, "train")
    total = sum(counts.values())
    assert base == [total / (len(counts) * n) for n in counts.values()]

    relabel_eval = copy.deepcopy(m)
    for e in relabel_eval["entries"]:
        if e["split"] in ("val", "test"):
            e["label"] = "circle"
    assert class_weights_from_train(relabel_eval) == base

    relabel_train = copy.deepcopy(m)
    next(e for e in relabel_train["entries"] if e["split"] == "train" and e["label"] != "circle")[
        "label"
    ] = "circle"
    assert class_weights_from_train(relabel_train) != base
