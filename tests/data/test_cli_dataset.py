import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from visionbench.cli import main


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "var"
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(root))
    monkeypatch.setenv("VISIONBENCH_MAX_IMAGE_PIXELS", "10000")
    return root


def _import(messy, capsys) -> str:
    assert main(["dataset", "import", str(messy.root), "--name", "messy"]) == 0
    out = capsys.readouterr().out
    return re.search(r"imported dataset (ds-[0-9a-f]{12})", out).group(1)


def test_import_report_list_split_success(messy, data_root: Path, capsys) -> None:
    ds = _import(messy, capsys)

    assert main(["dataset", "list"]) == 0
    assert ds in capsys.readouterr().out

    assert main(["dataset", "report", ds]) == 0
    out = capsys.readouterr().out
    assert (
        "label-conflict groups: 1" in out and "perceptual check skipped (low information): 2" in out
    )

    assert main(["dataset", "report", ds, "--json"]) == 0
    assert '"dataset_id": "' + ds in capsys.readouterr().out

    assert main(["dataset", "split", ds, "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert ": new" in out
    assert "groups_in_multiple_splits=0 sha256_in_multiple_splits=0" in out
    assert main(["dataset", "split", ds, "--seed", "3"]) == 0
    assert "already existed (recomputed manifest identical)" in capsys.readouterr().out


def test_failures_exit_nonzero_without_traceback(messy, data_root: Path, capsys) -> None:
    assert main(["dataset", "import", str(messy.root.parent / "nope"), "--name", "x"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: source directory does not exist") and "Traceback" not in err

    ds = _import(messy, capsys)
    assert main(["dataset", "import", str(messy.root), "--name", "messy"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err

    assert main(["dataset", "split", "ds-000000000000", "--seed", "1"]) == 1
    assert "not found" in capsys.readouterr().err

    assert main(["dataset", "split", ds, "--seed", "1", "--conflict-policy", "error"]) == 1
    assert "label-conflict" in capsys.readouterr().err

    assert main(["dataset", "split", ds, "--seed", "1", "--fractions", "0.5", "0.5", "0.5"]) == 1
    assert "sum to 1" in capsys.readouterr().err

    assert (
        main(["--debug", "dataset", "split", ds, "--seed", "1", "--fractions", "1", "1", "1"]) == 1
    )
    assert "Traceback" in capsys.readouterr().err
    assert (
        main(["dataset", "split", ds, "--debug", "--seed", "1", "--fractions", "1", "1", "1"]) == 1
    )
    assert "Traceback" in capsys.readouterr().err


def test_allow_missing_classes_flag(tmp_path: Path, data_root: Path, make_images, capsys) -> None:
    make_images(tmp_path / "src" / "big", 20, seed=1)
    make_images(tmp_path / "src" / "tiny", 3, seed=2)
    assert main(["dataset", "import", str(tmp_path / "src"), "--name", "skewed"]) == 0
    ds = re.search(r"(ds-[0-9a-f]{12})", capsys.readouterr().out).group(1)
    assert main(["dataset", "split", ds, "--seed", "0"]) == 1
    assert "class 'tiny' has 0 items in test" in capsys.readouterr().err
    assert main(["dataset", "split", ds, "--seed", "0", "--allow-missing-classes"]) == 0
    assert "warning: class 'tiny' has 0 items in test" in capsys.readouterr().out


def test_list_empty_and_bad_config(data_root: Path, capsys) -> None:
    assert main(["dataset", "list"]) == 0
    assert capsys.readouterr().out.strip() == "no datasets"
    assert main(["dataset", "import", ".", "--name", "x", "--dhash-threshold", "99"]) == 1
    assert "dhash_threshold" in capsys.readouterr().err


def test_process_exit_codes(messy, tmp_path: Path) -> None:
    env = os.environ | {
        "VISIONBENCH_DATA_ROOT": str(tmp_path / "var"),
        "VISIONBENCH_MAX_IMAGE_PIXELS": "10000",
    }
    cmd = [sys.executable, "-m", "visionbench.cli", "dataset"]
    ok = subprocess.run(
        [*cmd, "import", str(messy.root), "--name", "m"], env=env, capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run(
        [*cmd, "split", "ds-000000000000", "--seed", "1"], env=env, capture_output=True, text=True
    )
    assert bad.returncode == 1 and "Traceback" not in bad.stderr
    usage = subprocess.run([*cmd, "split"], env=env, capture_output=True, text=True)
    assert usage.returncode == 2  # argparse usage error
