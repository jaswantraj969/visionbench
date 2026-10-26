from pathlib import Path

import pytest
from pydantic import ValidationError

from visionbench.core.config import DATA_SUBDIRS, Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VISIONBENCH_DATA_ROOT", "VISIONBENCH_MAX_UPLOAD_MB", "VISIONBENCH_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.data_root == Path("./var")
    assert s.max_upload_mb == 200
    assert s.log_level == "INFO"


def test_env_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("VISIONBENCH_MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("VISIONBENCH_LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)
    assert s.resolved_data_root == (tmp_path / "root").resolve()
    assert s.max_upload_mb == 5
    assert s.log_level == "DEBUG"


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_upload_mb=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")


def test_subdirs_created_on_demand_only(tmp_path: Path) -> None:
    root = tmp_path / "var"
    s = Settings(_env_file=None, data_root=root)
    assert not root.exists()  # constructing settings has no filesystem side effects

    datasets = s.data_dir("datasets")
    assert datasets == root.resolve() / "datasets"
    assert datasets.is_dir()
    assert not (root / "artifacts").exists()

    dirs = s.ensure_data_dirs()
    assert set(dirs) == set(DATA_SUBDIRS)
    assert all(p.is_dir() for p in dirs.values())


def test_unknown_subdir_rejected(tmp_path: Path) -> None:
    s = Settings(_env_file=None, data_root=tmp_path)
    with pytest.raises(ValueError, match="unknown data subdir"):
        s.data_dir("../escape")
