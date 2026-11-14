import sys
import types

import pytest

from indicdlp.engine.training import attach_precision_guard, run_train, train_options
from indicdlp.config import apply_config
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


def test_standard_a6000_recipe_defaults_to_fp32_without_extra_flags():
    argv = ["train", "--config", "configs/training/a6000.yaml"]
    args = parser_for_cli().parse_args(argv)
    apply_config(args, argv)
    options = train_options(args, "data.yaml")
    assert options["amp"] is False and options["batch"] == 8
    assert options["nbs"] == 64


def test_precision_guard_forces_requested_fp32_after_checkpoint_restore(capsys):
    callbacks = {}
    model = types.SimpleNamespace(
        add_callback=lambda event, cb: callbacks.update({event: cb})
    )
    args = parser_for_cli().parse_args(["train", "--no-amp"])
    attach_precision_guard(model, args)
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(amp=True))
    callbacks["on_pretrain_routine_start"](trainer)
    assert trainer.args.amp is False
    assert "Training precision: FP32 (amp=False)" in capsys.readouterr().out


def test_precision_guard_does_not_enable_amp_from_a_fp32_resume():
    callbacks = {}
    model = types.SimpleNamespace(
        add_callback=lambda event, cb: callbacks.update({event: cb})
    )
    attach_precision_guard(model, parser_for_cli().parse_args(["train", "--amp"]))
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(amp=False))
    callbacks["on_pretrain_routine_start"](trainer)
    assert trainer.args.amp is False


def test_no_amp_guard_runs_before_upstream_amp_check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "indicdlp.engine.training.verify_dataset", lambda _: tmp_path / "data.yaml"
    )
    callbacks = {}

    class Model:
        trainer = types.SimpleNamespace(save_dir=tmp_path)

        def add_callback(self, event, callback):
            callbacks.setdefault(event, []).append(callback)

        def train(self, **options):
            # Model trainer configuration can differ from the call's kwargs,
            # notably when a checkpoint restores its saved settings.
            self.trainer.args = types.SimpleNamespace(amp=True)
            for callback in callbacks.get("on_pretrain_routine_start", []):
                callback(self.trainer)
            if self.trainer.args.amp:
                raise FileNotFoundError("yolov8n.pt")
            return "trained-without-auxiliary-check"

    monkeypatch.setattr("indicdlp.engine.training.load_model", lambda *args: Model())
    args = parser_for_cli().parse_args(["train", "--no-amp", "--batch", "8"])
    assert run_train(args) == "trained-without-auxiliary-check"
