import json
from types import SimpleNamespace

import pytest

from indicdlp.evaluation.metrics import snapshot
from indicdlp.evaluation.report import write_report
from indicdlp.tracking.wandb import Tracking
from tests.helpers import fake_wandb_sdk as tracking_sdk
from train_indicdlp import main, parser_for_cli


def metric_objects(save_dir=None):
    box = SimpleNamespace(
        ap_class_index=[0],
        p=[0.8],
        r=[0.6],
        f1=[2 * 0.8 * 0.6 / 1.4],
        ap50=[0.9],
        ap=[0.5],
        all_ap=[[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]],
    )
    matrix = SimpleNamespace(
        matrix=[[2, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
    )
    metrics = SimpleNamespace(
        names={0: "paragraph", 1: "table", 2: "figure"},
        box=box,
        results_dict={
            "metrics/precision(B)": 0.4,
            "metrics/recall(B)": 0.3,
            "metrics/mAP50(B)": 0.45,
            "metrics/mAP50-95(B)": 0.25,
        },
        speed={"inference": 1.2},
        confusion_matrix=matrix,
        curves=["Precision-Recall(B)"],
        curves_results=[[[0, 0.5, 1], [[1, 0.8, 0]], "Recall", "Precision"]],
    )
    validator = SimpleNamespace(
        metrics=metrics,
        names=metrics.names,
        nt_per_class=[2, 1, 0],
        seen=3,
        save_dir=save_dir,
    )
    return metrics, validator


def test_metrics_keep_undefined_classes_and_zero_ap_distinct(tmp_path):
    metrics, validator = metric_objects(tmp_path)
    report = snapshot(metrics, validator)
    assert report["per_class"]["0"]["precision"] == 0.8
    assert report["per_class"]["0"]["map75"] == 0.4
    assert report["per_class"]["1"]["map50_95"] == 0
    assert report["per_class"]["2"]["map50_95"] is None
    assert report["per_class"]["2"]["evaluated"] is False
    assert report["aggregate"]["metrics/macro_F1(B)"] == pytest.approx(
        (2 * 0.8 * 0.6 / 1.4) / 2
    )
    assert report["images"] == 3
    assert len(report["per_class"]["0"]["ap_by_iou"]) == 10
    path = write_report(tmp_path / "report", report)
    assert {p.name for p in path.iterdir()} == {
        "metrics.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "curves.json",
        "summary.md",
    }
    json.dumps(json.loads((path / "metrics.json").read_text()), allow_nan=False)


def test_full_reports_log_scalars_tables_curves_and_numeric_artifacts(
    monkeypatch, tmp_path
):
    _, runs = tracking_sdk(monkeypatch)
    args = parser_for_cli().parse_args(["test", "--wandb", "offline"])
    tracker = Tracking(args, {}, "test").start()
    tracker.sdk.Table = lambda **kwargs: kwargs
    tracker.sdk.plot = SimpleNamespace(line_series=lambda **kwargs: kwargs)
    artifacts = []

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []

        def add_file(self, file):
            self.files.append(file)

    tracker.sdk.Artifact = Artifact
    runs[0].log_artifact = artifacts.append
    metrics, validator = metric_objects(tmp_path)
    report = snapshot(metrics, validator)
    path = write_report(tmp_path / "report", report)
    tracker.report(report, "test", path)
    values = runs[0].logged[0][0]
    assert values["test/metrics/mAP50-95(B)"] == 0.25
    assert values["test/classes/1_table/map50_95"] == 0
    assert "test/classes/2_figure/map50_95" not in values
    assert len(values["test/per_class"]["data"]) == 3
    assert (
        "test/confusion_matrix" in values
        and "test/curves/Precision-Recall(B)" in values
    )
    assert all(not file.endswith(".png") for file in artifacts[0].files)
    assert any(file.endswith("metrics.json") for file in artifacts[0].files)
    tracker.finish()


def test_every_epoch_logs_train_and_validation_losses_and_per_class_metrics(
    monkeypatch, tmp_path
):
    _, runs = tracking_sdk(monkeypatch)
    args = parser_for_cli().parse_args(["train", "--wandb", "offline"])
    tracker = Tracking(args, {}).start()
    _, validator = metric_objects(tmp_path)
    trainer = SimpleNamespace(
        epoch=2,
        save_dir=tmp_path,
        validator=validator,
        metrics={"val/box_om_loss": 1.5, "val/cls_oo_loss": 0.7},
        lr={"lr/pg0": 0.0001},
        tloss=[1],
        label_loss_items=lambda *a, **kw: {
            "train/box_om_loss": 2.0,
            "train/cls_oo_loss": 1.0,
        },
    )
    tracker.on_epoch(trainer)
    values = runs[0].logged[0][0]
    assert values["train/box_om_loss"] == 2 and values["val/box_om_loss"] == 1.5
    assert values["val/classes/0_paragraph/recall"] == 0.6
    assert (tmp_path / "reports/epochs/0003.json").exists()
    tracker.finish()


def test_test_run_checks_group_provenance_and_writes_reports(
    monkeypatch, fold_dataset, tmp_path
):
    from indicdlp.data.folds import fingerprint
    from indicdlp.data.prepare import digest

    root, folds = fold_dataset
    folder = tmp_path / "trained"
    (folder / "weights").mkdir(parents=True)
    weights = folder / "weights/best.pt"
    weights.write_bytes(b"synthetic")
    context = {
        "fold": 1,
        "dataset": fingerprint(root),
        "plan_sha256": digest((folds / "plan.json").read_bytes()),
    }
    (folder / "training_context.json").write_text(json.dumps(context))
    metrics, validator = metric_objects(tmp_path / "eval")

    class Model:
        def add_callback(self, event, callback):
            self.callback = callback

        def val(self, **kwargs):
            assert kwargs["split"] == "test" and kwargs["plots"] is True
            self.callback(validator)
            return metrics

    monkeypatch.setattr("indicdlp.engine.training.load_model", lambda *a: Model())
    report = main(
        [
            "test",
            "--data",
            str(root),
            "--folds",
            str(folds),
            "--weights",
            str(weights),
            "--report-dir",
            str(tmp_path / "test-report"),
        ]
    )
    assert report["split"] == "test"
    assert report["provenance"]["held_out_test_pages"] == 4
    assert "document groups" in report["provenance"]["isolation"]
    assert (tmp_path / "test-report/per_class.csv").exists()
    context["fold"] = 99
    (folder / "training_context.json").write_text(json.dumps(context))
    with pytest.raises(ValueError, match="provenance"):
        main(
            [
                "test",
                "--data",
                str(root),
                "--folds",
                str(folds),
                "--weights",
                str(weights),
            ]
        )


def test_packaged_class_resource_and_test_command_guard():
    from indicdlp.cli import adapter_for

    args = parser_for_cli().parse_args(["prepare"])
    assert len(adapter_for(args).names) == 42
    with pytest.raises(SystemExit):
        main(["test", "--split", "val"])
