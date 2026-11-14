import json
import sys
from types import SimpleNamespace

import pytest

from indicdlp.config import apply_config, validate_training
from indicdlp.tracking.wandb import Tracking
from train_indicdlp import main, parser_for_cli
from tests.helpers import fake_wandb_sdk as sdk


def test_a6000_recipe_and_explicit_cli_precedence():
    argv = [
        "train",
        "--config",
        "configs/training/a6000.yaml",
        "--batch",
        "8",
        "--no-amp",
        "--wandb",
        "offline",
    ]
    args = parser_for_cli().parse_args(argv)
    apply_config(args, argv)
    validate_training(args)
    assert args.imgsz == 1280 and args.batch == 8 and args.amp is False
    assert args.wandb == "offline" and args.epochs == 150 and args.warmup_epochs == 5
    assert args.nbs == 64 and args.lr0 == 0.0002 and args.warmup_bias_lr == 0


@pytest.mark.parametrize(
    "text",
    [
        "train: {typo: 1}",
        "train: {batch: 'sixteen'}",
        "train: {lr0: .nan}",
        "tracking: {mode: strange}",
    ],
)
def test_bad_recipe_rejected_before_network(tmp_path, text):
    config = tmp_path / "bad.yaml"
    config.write_text(text)
    with pytest.raises(SystemExit):
        main(["train", "--config", str(config), "--dry-run"])


def test_highres_recipe():
    args = parser_for_cli().parse_args(
        ["train", "--config", "configs/training/a6000_highres.yaml"]
    )
    apply_config(args, [])
    validate_training(args)
    assert (args.imgsz, args.batch, args.nbs) == (1600, 8, 64)


def test_tracking_callbacks_and_no_secret_logging(monkeypatch, tmp_path):
    calls, runs = sdk(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "secret-must-never-be-logged")
    args = parser_for_cli().parse_args(["train", "--wandb", "offline"])
    tracker = Tracking(args, {"train": {"lr0": 1e-4}}).start()
    callbacks = {}
    tracker.attach(
        SimpleNamespace(add_callback=lambda event, cb: callbacks.update({event: cb}))
    )
    trainer = SimpleNamespace(
        save_dir=tmp_path,
        args=SimpleNamespace(lr0=1e-4),
        epoch=0,
        metrics={"metrics/mAP50-95(B)": 0.5},
        lr={"lr/pg0": 1e-4},
        tloss=[2],
        label_loss_items=lambda loss, prefix: {"train/box_loss": 2},
    )
    callbacks["on_pretrain_routine_start"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    tracker.summary({"final/map": 0.5})
    tracker.finish()
    assert runs[0].logged[0][0]["epoch"] == 1
    assert runs[0].logged[0][0]["train/box_loss"] == 2
    assert runs[0].summary["final/map"] == 0.5
    assert runs[0].finished == [{"exit_code": 0}]
    assert "secret-must" not in json.dumps(calls)
    assert calls[0]["save_code"] is False


def test_online_resume_uses_persisted_id_and_project(monkeypatch, tmp_path):
    calls, _ = sdk(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-key")
    folder = tmp_path / "run"
    (folder / "weights").mkdir(parents=True)
    (folder / "wandb_run.json").write_text(
        json.dumps(
            {
                "id": "original",
                "project": "original-project",
                "entity": "team",
                "mode": "online",
            }
        )
    )
    args = parser_for_cli().parse_args(
        ["train", "--wandb", "online", "--resume", str(folder / "weights/last.pt")]
    )
    tracker = Tracking(args, {}).start()
    assert calls[0]["id"] == "original" and calls[0]["resume"] == "must"
    assert calls[0]["project"] == "original-project"
    tracker.finish(1)


def test_evaluation_inherits_recipe_and_explicit_flags():
    argv = [
        "test",
        "--config",
        "configs/training/a6000.yaml",
        "--batch",
        "8",
        "--wandb",
        "offline",
    ]
    args = parser_for_cli().parse_args(argv)
    apply_config(args, argv, available_only=True)
    assert args.imgsz == 1280 and args.batch == 8 and args.wandb == "offline"
    assert args.max_det == 1000


def test_disabled_tracking_never_imports_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    args = parser_for_cli().parse_args(["train"])
    assert Tracking(args, {}).start().run is None
