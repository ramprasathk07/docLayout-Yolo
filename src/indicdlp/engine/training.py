"""DocLayout-YOLO API entry points, with a network-free training preview."""

from __future__ import annotations

import json
import os
import csv
import sys
from pathlib import Path

from ..data.prepare import verify_dataset, digest, json_write
from ..config import HYPER_DEFAULTS, validate_training
from ..tracking.wandb import Tracking, disable_native_wandb


DEFAULT_WEIGHTS = (
    "hf://juliozhao/DocLayout-YOLO-DocSynth300K-pretrain/"
    "doclayout_yolo_docsynth300k_imgsz1600.pt"
)


def resolve_weights(value):
    if value.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = value[5:].split("/", 2)
        if len(parts) != 3:
            raise ValueError("HF weights must be hf://owner/repo/filename.pt")
        return hf_hub_download(repo_id="/".join(parts[:2]), filename=parts[2])
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"Weights/config file not found: {path}")
    return str(path.resolve())


def load_model(weights, device):
    import torch

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Install the server's CUDA torch build or use --device cpu."
        )
    if tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 6):
        raise RuntimeError(
            f"Unsupported torch {torch.__version__} in {sys.executable}. "
            "DocLayout-YOLO 0.0.4 expects the pre-2.6 full-object checkpoint loader. "
            "Create a separate Python 3.11 training environment with torch==2.5.1 "
            "and torchvision==0.20.1; see docs/server-a6000.md. "
            "A newer CUDA-enabled torch installation does not satisfy this fork's compatibility pin."
        )
    os.environ.setdefault("COMET_MODE", "DISABLED")
    from doclayout_yolo import YOLOv10

    disable_native_wandb()
    return YOLOv10(resolve_weights(weights))


def train_options(args, data):
    return dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        seed=args.seed,
        lr0=args.lr0,
        patience=args.patience,
        save=True,
        val=True,
        plots=True,
        exist_ok=False,
        amp=args.amp,
        **{key: getattr(args, key, default) for key, default in HYPER_DEFAULTS.items()},
    )


def training_data(args):
    data = verify_dataset(args.data)
    if getattr(args, "fold", None) is not None:
        if not args.folds:
            raise ValueError("--fold requires --folds")
        from ..data.folds import fold_data

        data = fold_data(args.data, args.folds, args.fold)
    return data


def best_epoch_from_csv(path):
    if not Path(path).exists():
        return None
    with Path(path).open(newline="") as handle:
        rows = [
            {k.strip(): v.strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    rows = [r for r in rows if "metrics/mAP50-95(B)" in r and "metrics/mAP50(B)" in r]
    if not rows:
        return None
    best = max(
        rows,
        key=lambda r: 0.1 * float(r["metrics/mAP50(B)"])
        + 0.9 * float(r["metrics/mAP50-95(B)"]),
    )
    return int(float(best["epoch"]))


def run_train(args):
    validate_training(args)
    data = training_data(args)
    options = train_options(args, data)
    if args.batch != -1 and args.batch % len(args.device.split(",")):
        raise ValueError("Batch size must be divisible by the number of devices")
    if args.batch == -1 and "," in args.device:
        raise ValueError("AutoBatch is only supported on one device")
    if "," in args.device and (args.wandb != "disabled" or getattr(args, "fold", None)):
        raise ValueError(
            "Custom W&B/fold callbacks currently require a single GPU process"
        )
    if args.resume and not Path(args.resume).is_file():
        raise ValueError("--resume must point to an existing last.pt checkpoint")
    context = None
    if getattr(args, "fold", None):
        from ..data.folds import fingerprint

        context = {
            "fold": args.fold,
            "dataset": fingerprint(Path(args.data).resolve()),
            "plan_sha256": digest((Path(args.folds) / "plan.json").read_bytes()),
        }
        if args.resume:
            saved_context = (
                Path(args.resume).resolve().parent.parent / "training_context.json"
            )
            if (
                not saved_context.exists()
                or json.loads(saved_context.read_text()) != context
            ):
                raise ValueError(
                    "Resume checkpoint does not belong to this dataset/fold plan/fold index"
                )
    preview = {
        "weights": args.resume or args.weights,
        "train": options,
        "resume": bool(args.resume),
        "tracking": {
            "mode": args.wandb,
            "project": args.wandb_project,
            "group": args.wandb_group,
        },
    }
    print(json.dumps(preview, indent=2))
    if args.dry_run:
        return preview
    # Fail on incompatible torch/CUDA/checkpoints before creating an empty W&B run.
    model = load_model(args.resume or args.weights, args.device)
    tracking = Tracking(
        args, preview, "cross-validation" if getattr(args, "fold", None) else "train"
    ).start()
    try:
        if context:
            model.add_callback(
                "on_pretrain_routine_start",
                lambda trainer: json_write(
                    Path(trainer.save_dir) / "training_context.json", context
                ),
            )
        tracking.attach(model)
        if args.resume:
            # Upstream restores saved epochs, optimizer, scheduler, and run directory.
            result = model.train(
                resume=str(Path(args.resume).resolve()),
                data=str(data),
                device=args.device,
                workers=args.workers,
            )
        else:
            result = model.train(**options)
        print(f"Training completed. Checkpoints: {model.trainer.save_dir}/weights")
        best = Path(
            getattr(
                model.trainer, "best", Path(model.trainer.save_dir) / "weights/best.pt"
            )
        )
        tracking.artifact(best)
        if getattr(args, "_return_summary", False):
            if not best.exists():
                raise RuntimeError(
                    "Best fold checkpoint missing; cannot evaluate the fold"
                )
            evaluator = load_model(str(best), args.device)
            from ..evaluation.runner import evaluate_model
            import copy

            eval_args = copy.copy(args)
            eval_args.weights = str(best)
            eval_args.project = str(model.trainer.save_dir)
            report, report_path = evaluate_model(
                evaluator, data, eval_args, "val", "cv_validation"
            )
            summary = {
                "metrics": report["aggregate"],
                "per_class": report["per_class"],
                "best": str(best.resolve()),
                "best_sha256": digest(best.read_bytes()),
                "save_dir": str(model.trainer.save_dir),
                "report_dir": str(report_path),
                "best_epoch": best_epoch_from_csv(
                    Path(model.trainer.save_dir) / "results.csv"
                ),
            }
            tracking.report(report, "cv_validation", report_path)
            result = summary
        return result
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    finally:
        tracking.finish()
