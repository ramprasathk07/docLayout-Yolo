"""Explicit validation/test runs with split provenance and metric report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ..data.folds import fingerprint, load_plan
from ..data.prepare import digest
from ..tracking.wandb import Tracking
from .metrics import snapshot
from .report import write_report


def evaluate_model(model, data, args, split="val", name=None):
    captured = {}
    if hasattr(model, "add_callback"):
        model.add_callback(
            "on_val_end", lambda validator: captured.update(validator=validator)
        )
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        conf=getattr(args, "conf", 0.001),
        iou=getattr(args, "iou", 0.7),
        half=args.device != "cpu" and getattr(args, "half", True),
        project=args.project,
        name=name or args.name,
        plots=True,
    )
    validator = captured.get("validator")
    if not getattr(metrics, "names", None) and hasattr(model, "names"):
        metrics.names = model.names
    report = snapshot(metrics, validator)
    report.update(
        split=split,
        checkpoint=str(getattr(args, "weights", "")),
        dataset=str(data),
        requested_confidence_threshold=getattr(args, "conf", 0.001),
        requested_postprocess_iou=getattr(args, "iou", 0.7),
        postprocessing_note="YOLOv10 uses end-to-end postprocessing and can ignore requested NMS/confidence arguments. Actual confusion-matrix thresholds are stored with that matrix; AP matching uses iou_thresholds.",
        loss_note="Standalone inference does not compute detection losses. Train-time validation losses are logged per epoch.",
    )
    destination = (
        getattr(args, "report_dir", None)
        or Path(
            getattr(validator, "save_dir", Path(args.project) / (name or args.name))
        )
        / "report"
    )
    path = write_report(destination, report)
    return report, path


def test_provenance(args):
    provenance = {
        "dataset": fingerprint(Path(args.data).resolve()),
        "isolation": "official page split",
    }
    if not args.folds:
        return provenance
    plan = load_plan(args.data, args.folds, verify=False)
    context_path = Path(args.weights).resolve().parent.parent / "training_context.json"
    if not context_path.exists():
        raise ValueError(
            "Grouped held-out testing requires a checkpoint with training_context.json"
        )
    context = json.loads(context_path.read_text())
    if (
        context.get("dataset") != provenance["dataset"]
        or context.get("plan_sha256")
        != digest((Path(args.folds) / "plan.json").read_bytes())
        or context.get("fold") not in range(1, plan["n_splits"] + 1)
    ):
        raise ValueError(
            "Checkpoint provenance does not match the held-out test/fold plan"
        )
    provenance.update(
        isolation="document groups; official test excluded from every CV fold",
        fold=context["fold"],
        held_out_test_pages=len(plan["test_ids"]),
        page_fallback_count=len(plan["page_fallback_ids"]),
    )
    if plan["page_fallback_ids"]:
        provenance["isolation"] = (
            "page-level fallback; source-document independence not established"
        )
    return provenance


def run_evaluate(args):
    from ..engine.training import load_model, training_data

    data = training_data(args)
    if args.fold and args.split != "val":
        raise ValueError(
            "Fold evaluation is validation-only; use test --folds DIR without --fold"
        )
    if not args.fold and not (Path(args.data) / f"{args.split}.txt").exists():
        raise ValueError(f"Prepared subset has no {args.split} split")
    if (
        args.report_dir
        and Path(args.report_dir).exists()
        and any(Path(args.report_dir).iterdir())
    ):
        raise ValueError("Report output is nonempty; choose a new --report-dir")
    provenance = (
        test_provenance(args)
        if args.split == "test"
        else {"dataset": fingerprint(Path(args.data).resolve())}
    )
    model = load_model(args.weights, args.device)
    tracker = Tracking(
        args,
        {"split": args.split, "weights": args.weights, "data": str(data), **provenance},
        args.split,
    ).start()
    try:
        report, path = evaluate_model(model, data, args, args.split)
        report["provenance"] = provenance
        write_report(path, report)
        tracker.report(report, args.split, path)
        print(
            json.dumps(
                {"metrics": report["aggregate"], "report_dir": str(path)}, indent=2
            )
        )
        return report
    except BaseException:
        tracker.finish(1)
        raise
    finally:
        tracker.finish()
