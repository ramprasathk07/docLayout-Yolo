import sys
import types

import pytest

from indicdlp.engine.training import run_train
from train_indicdlp import main, parser_for_cli


def test_training_preview_and_call_use_doclayout_api(monkeypatch, tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: [paragraph]\n")
    monkeypatch.setattr("indicdlp.engine.training.verify_dataset", lambda _: data)
    calls = []

    class Model:
        trainer = types.SimpleNamespace(save_dir="runs/fixture")

        def train(self, **kwargs):
            calls.append(kwargs)
            return "trained"

    monkeypatch.setattr("indicdlp.engine.training.load_model", lambda w, d: Model())
    args = parser_for_cli().parse_args(["train", "--data", str(tmp_path), "--dry-run"])
    preview = run_train(args)
    assert preview["weights"].startswith("hf://")
    assert preview["train"]["fliplr"] == 0
    assert preview["train"]["mosaic"] == 0
    assert not calls
    args.dry_run = False
    assert run_train(args) == "trained"
    assert calls[0]["data"] == str(data)
    assert calls[0]["epochs"] == 100


def test_resume_restores_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    data = tmp_path / "data.yaml"
    monkeypatch.setattr("indicdlp.engine.training.verify_dataset", lambda _: data)
    calls = []
    model = types.SimpleNamespace(
        train=lambda **kw: calls.append(kw),
        trainer=types.SimpleNamespace(save_dir="run"),
    )
    monkeypatch.setattr("indicdlp.engine.training.load_model", lambda w, d: model)
    args = parser_for_cli().parse_args(["train", "--resume", str(checkpoint)])
    run_train(args)
    assert calls[0]["resume"] == str(checkpoint.resolve())
    assert "epochs" not in calls[0]  # The saved run owns its original schedule.


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--batch", "0"],
        ["train", "--imgsz", "127"],
        ["prepare", "--splits", "train"],
        ["prepare", "--max-transfer-mib", "0"],
    ],
)
def test_invalid_cli_rejected_before_network(args):
    with pytest.raises(SystemExit):
        main(args)


def test_dry_run_does_not_import_model_package(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "indicdlp.engine.training.verify_dataset", lambda _: tmp_path / "data.yaml"
    )
    monkeypatch.setitem(sys.modules, "doclayout_yolo", None)
    main(["train", "--dry-run"])
