"""`visionbench demo-data`: deterministic synthetic shapes, folder + ZIP, importable."""

import zipfile
from pathlib import Path

import pytest

from visionbench.cli import main
from visionbench.data.demo import SHAPES, demo_images, write_demo


def test_demo_images_are_deterministic_for_a_seed() -> None:
    assert demo_images(3, seed=0) == demo_images(3, seed=0)
    assert demo_images(3, seed=0) != demo_images(3, seed=1)
    assert [rel for rel, _ in demo_images(2)] == [
        f"{s}/{s}_{i:03d}.png" for s in SHAPES for i in range(2)
    ]


def test_write_demo_writes_folder_and_zip_with_the_same_images(tmp_path: Path) -> None:
    folder, zip_path = write_demo(tmp_path, per_class=4)
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    assert len(files) == 12
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        assert names == [f"shapes/{f}" for f in files]
        assert (
            zf.read("shapes/circle/circle_000.png")
            == (folder / "circle/circle_000.png").read_bytes()
        )
    with pytest.raises(FileExistsError):
        write_demo(tmp_path, per_class=4)


def test_demo_data_command_output_imports_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(tmp_path / "var"))
    assert main(["demo-data", str(tmp_path / "demo"), "--per-class", "10"]) == 0
    assert "not EuroSAT" in capsys.readouterr().out
    assert main(["dataset", "import", str(tmp_path / "demo" / "shapes"), "--name", "demo"]) == 0
    assert "imported dataset" in capsys.readouterr().out
