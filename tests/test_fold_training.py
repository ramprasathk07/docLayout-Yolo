import json
from types import SimpleNamespace

import pytest

from indicdlp.engine.training import best_epoch_from_csv, run_train
from train_indicdlp import parser_for_cli


def test_fold_training_records_context_and_evaluates_best_validation_only(
    monkeypatch, fold_dataset, tmp_path
):
    root, folds = fold_dataset
    folder = tmp_path / "trained"
    (folder / "weights").mkdir(parents=True)
    best = folder / "weights/best.pt"
    best.write_bytes(b"synthetic weights")
    (folder / "results.csv").write_text(
        "epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n1,0.8,0.4\n2,0.7,0.5\n"
    )
    callbacks, validations = {}, []

    class Model:
        trainer = SimpleNamespace(save_dir=folder, best=best)

        def add_callback(self, event, callback):
            callbacks[event] = callback

        def train(self, **options):
            self.trainer.args = SimpleNamespace(**options)
            callbacks["on_pretrain_routine_start"](self.trainer)

    evaluator = SimpleNamespace(names={0: "paragraph"})

    def validate(**kwargs):
        validations.append(kwargs)
        return SimpleNamespace(
            results_dict={"metrics/mAP50-95(B)": 0.5},
            box=SimpleNamespace(ap_class_index=[0], ap50=[0.8], ap=[0.5]),
        )

    evaluator.val = validate
    model = Model()
    monkeypatch.setattr(
        "indicdlp.engine.training.load_model",
        lambda weights, device: evaluator if weights == str(best) else model,
    )
    args = parser_for_cli().parse_args(
        ["train", "--data", str(root), "--folds", str(folds), "--fold", "1"]
    )
    args._return_summary = True
    result = run_train(args)
    assert result["best_epoch"] == 2 and result["per_class"]["0"]["name"] == "paragraph"
    assert validations[0]["split"] == "val" and "fold_01" in validations[0]["data"]
    assert json.loads((folder / "training_context.json").read_text())["fold"] == 1


def test_wrong_fold_checkpoint_is_rejected_before_model_load(fold_dataset, tmp_path):
    root, folds = fold_dataset
    checkpoint = tmp_path / "run/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / "training_context.json").write_text('{"fold": 2}')
    args = parser_for_cli().parse_args(
        [
            "train",
            "--data",
            str(root),
            "--folds",
            str(folds),
            "--fold",
            "1",
            "--resume",
            str(checkpoint),
        ]
    )
    with pytest.raises(ValueError, match="does not belong"):
        run_train(args)


def test_best_epoch_matches_upstream_detection_fitness(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.9,0.4\n2,0.8,0.5\n"
    )
    assert best_epoch_from_csv(path) == 2
