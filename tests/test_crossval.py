import pytest

from indicdlp.engine.crossval import aggregate, run_crossval
from indicdlp.data.prepare import digest
from train_indicdlp import parser_for_cli


def arguments(root, folds, tmp_path, extra=None):
    return parser_for_cli().parse_args(
        [
            "crossval",
            "--data",
            str(root),
            "--folds",
            str(folds),
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "experiment",
            *(extra or []),
        ]
    )


def test_crossval_fresh_initialization_aggregate_and_reuse(
    monkeypatch, fold_dataset, tmp_path
):
    root, folds = fold_dataset
    weights = tmp_path / "initial.pt"
    weights.write_bytes(b"synthetic model initialization")
    monkeypatch.setattr(
        "indicdlp.engine.crossval.resolve_weights", lambda _: str(weights)
    )
    seen = []

    def train(args):
        seen.append((args.fold, args.weights, args.seed, args.resume))
        checkpoint = tmp_path / f"fold{args.fold}.pt"
        checkpoint.write_bytes(str(args.fold).encode())
        return {
            "metrics": {"metrics/mAP50-95(B)": args.fold / 10},
            "best": str(checkpoint),
            "best_sha256": digest(checkpoint.read_bytes()),
            "best_epoch": args.fold + 10,
        }

    monkeypatch.setattr("indicdlp.engine.crossval.run_train", train)
    args = arguments(root, folds, tmp_path)
    summary = run_crossval(args)
    assert summary["complete"] and summary["completed_folds"] == 3
    assert summary["metrics"]["metrics/mAP50-95(B)"]["mean"] == pytest.approx(0.2)
    assert summary["metrics"]["metrics/mAP50-95(B)"]["std"] == pytest.approx(0.1)
    assert {initial for _, initial, _, _ in seen} == {str(weights)}
    assert [fold for fold, _, _, _ in seen] == [1, 2, 3]
    assert run_crossval(args)["complete"]
    assert len(seen) == 3
    args.lr0 *= 2
    with pytest.raises(ValueError, match="changed"):
        run_crossval(args)


def test_cv_dry_run_never_loads_weights_or_starts_tracking(
    monkeypatch, fold_dataset, tmp_path
):
    root, folds = fold_dataset
    monkeypatch.setattr(
        "indicdlp.engine.crossval.resolve_weights",
        lambda _: pytest.fail("Weight download during dry run"),
    )
    args = arguments(root, folds, tmp_path, ["--dry-run"])
    preview = run_crossval(args)
    assert len(preview) == 3
    assert len({p["train"]["data"] for p in preview}) == 3
    assert not (tmp_path / "runs").exists()


def test_cv_partial_summary_is_not_complete():
    report = aggregate([{"metrics": {"map": 0.5}, "best_epoch": 12}], 5)
    assert report["complete"] is False and report["metrics"]["map"]["std"] is None


def test_cv_single_gpu_requirement(fold_dataset, tmp_path):
    root, folds = fold_dataset
    with pytest.raises(ValueError, match="one GPU"):
        run_crossval(arguments(root, folds, tmp_path, ["--device", "0,1"]))


def test_cv_resume_only_applies_checkpoint_to_interrupted_fold(
    monkeypatch, fold_dataset, tmp_path
):
    root, folds = fold_dataset
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"initial")
    last = tmp_path / "last.pt"
    last.write_bytes(b"last")
    monkeypatch.setattr(
        "indicdlp.engine.crossval.resolve_weights", lambda _: str(initial)
    )
    seen = []

    def train(args):
        seen.append((args.fold, args.resume))
        if args.fold == 2 and not args.resume:
            raise RuntimeError("simulated interruption")
        best = tmp_path / f"best{args.fold}.pt"
        best.write_bytes(str(args.fold).encode())
        return {
            "metrics": {"map": args.fold / 10},
            "best": str(best),
            "best_sha256": digest(best.read_bytes()),
            "best_epoch": 5,
        }

    monkeypatch.setattr("indicdlp.engine.crossval.run_train", train)
    with pytest.raises(RuntimeError, match="interruption"):
        run_crossval(arguments(root, folds, tmp_path))
    summary = run_crossval(
        arguments(root, folds, tmp_path, ["--resume-fold", "2", "--resume", str(last)])
    )
    assert summary["complete"]
    assert seen == [(1, None), (2, None), (2, str(last)), (3, None)]


def test_per_class_aggregation_reports_observed_fold_count():
    report = aggregate(
        [
            {
                "metrics": {"map": 0.5},
                "per_class": {
                    "21": {"name": "paragraph", "map50": 0.8, "map50_95": 0.5}
                },
            },
            {"metrics": {"map": 0.6}, "per_class": {}},
        ],
        2,
    )
    assert report["per_class"]["21"]["observed_folds"] == 1
    assert report["per_class"]["21"]["map50_95"]["std"] is None
