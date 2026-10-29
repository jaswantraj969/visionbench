"""TEST images unlock per split, only once that split's benchmark has its final report (M8)."""

from pathlib import Path

from visionbench.benchmarks import final_test_splits, split_test_images_allowed
from visionbench.core.config import Settings

SPLIT = "sp-aaaaaaaaaaaa"


def _bench(root: Path, with_report: bool) -> Settings:
    d = root / "toy-v1"
    (d / "results").mkdir(parents=True)
    (d / "BENCHMARK_PROTOCOL.md").write_text(f"| Split | `{SPLIT}` (`vb-split-v2`) |\n")
    if with_report:
        (d / "results" / "FINAL_REPORT.json").write_text("{}")
    return Settings(_env_file=None, data_root=root / "var", benchmarks_dir=root)


def test_locked_until_the_final_report_exists(tmp_path: Path) -> None:
    s = _bench(tmp_path, with_report=False)
    assert final_test_splits(s) == []
    assert not split_test_images_allowed(s, SPLIT)


def test_only_the_finalised_split_is_unlocked(tmp_path: Path) -> None:
    s = _bench(tmp_path, with_report=True)
    assert final_test_splits(s) == [SPLIT]
    assert split_test_images_allowed(s, SPLIT)
    assert not split_test_images_allowed(s, "sp-bbbbbbbbbbbb")
    assert not split_test_images_allowed(s, None)
