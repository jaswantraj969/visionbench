import json

import pytest

from visionbench.cli import main


def test_env_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    info = json.loads(capsys.readouterr().out)
    for key in ("python", "torch", "torchvision", "numpy", "platform", "cuda_available"):
        assert key in info


def test_subcommand_required() -> None:
    with pytest.raises(SystemExit):
        main([])
