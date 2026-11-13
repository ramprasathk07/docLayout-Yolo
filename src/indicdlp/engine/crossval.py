"""Sequential fresh-weight fold training, restartable results, and mean/std reports."""

import copy
import json
import statistics
from pathlib import Path

from ..data.folds import load_plan
from ..data.prepare import digest, json_write
from ..tracking.wandb import Tracking
from .training import run_train, train_options, resolve_weights


def aggregate(results, expected):
    keys = (
        sorted(set.intersection(*(set(r["metrics"]) for r in results)))
        if results
        else []
    )
    metrics = {}
    for key in keys:
        values = [r["metrics"][key] for r in results]
        metrics[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    epochs = [r["best_epoch"] for r in results if r.get("best_epoch") is not None]
    per_class = {}
    for cid in sorted(
        {cid for r in results for cid in r.get("per_class", {})}, key=int
    ):
        observed = [
            r["per_class"][cid]
            for r in results
            if cid in r.get("per_class", {})
            and r["per_class"][cid].get("evaluated", True)
        ]
        if not observed:
            continue
        per_class[cid] = {"name": observed[0]["name"], "observed_folds": len(observed)}
        for metric in ("precision", "recall", "f1", "map50", "map75", "map50_95"):
            values = [r[metric] for r in observed if r.get(metric) is not None]
            per_class[cid][metric] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
    return {
        "complete": len(results) == expected,
        "completed_folds": len(results),
        "expected_folds": expected,
        "metrics": metrics,
        "per_class": per_class,
        "median_best_epoch": statistics.median(epochs) if epochs else None,
        "folds": results,
        "interpretation": "Unweighted mean and sample std over held-out validation folds; not a pooled mAP or confidence interval. Official test not evaluated.",
    }


def run_crossval(args):
    if bool(args.resume) != bool(args.resume_fold):
        raise ValueError(
            "CV resume requires both --resume-fold N and --resume path/to/last.pt"
        )
    if args.batch < 1:
        raise ValueError(
            "Use a fixed positive batch for comparable cross-validation runs"
        )
    plan = load_plan(args.data, args.folds)
    indices = args.fold_indices or list(range(1, plan["n_splits"] + 1))
    if len(set(indices)) != len(indices) or any(
        i < 1 or i > plan["n_splits"] for i in indices
    ):
        raise ValueError("Invalid/duplicate fold indices")
    if args.resume_fold and args.resume_fold not in indices:
        raise ValueError("--resume-fold must be one of the selected fold indices")
    # Single GPU process preserves callback tracking; each fold runs sequentially.
    if "," in args.device:
        raise ValueError(
            "crossval currently uses one GPU per process; use --device 0 on the A6000"
        )
    base_options = train_options(args, "fold-specific")
    run_settings = {
        "plan_sha256": digest((Path(args.folds) / "plan.json").read_bytes()),
        "weights": args.weights,
        "train": base_options,
    }
    initial_weights = args.weights
    if not args.dry_run:
        initial_weights = resolve_weights(args.weights)
        run_settings["initial_weights_sha256"] = digest(
            Path(initial_weights).read_bytes()
        )
    signature = digest(json.dumps(run_settings, sort_keys=True).encode())
    output = Path(args.project).resolve() / f"{args.name}_crossval"
    metadata_path = output / "experiment.json"
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        if (
            metadata_path.exists()
            and json.loads(metadata_path.read_text())["signature"] != signature
        ):
            raise ValueError(
                "CV hyperparameters/weights/plan changed; use a new --name"
            )
        json_write(metadata_path, {"signature": signature, **run_settings})
    results = []
    for index in indices:
        fold_args = copy.deepcopy(args)
        fold_args.fold = index
        fold_args.weights = initial_weights
        fold_args.resume = args.resume if index == args.resume_fold else None
        fold_args.name = f"{args.name}_fold{index:02d}"
        fold_args.wandb_group = args.wandb_group or f"{args.name}-cv-{signature[:8]}"
        fold_args._return_summary = True
        result_path = output / f"fold_{index:02d}.json"
        if result_path.exists() and not args.dry_run:
            result = json.loads(result_path.read_text())
            checkpoint = Path(result["best"])
            if (
                result["signature"] != signature
                or not checkpoint.is_file()
                or digest(checkpoint.read_bytes()) != result["best_sha256"]
            ):
                raise ValueError(
                    "Saved fold result/checkpoint changed; inspect the experiment before reusing it"
                )
            print(f"Reusing completed fold {index}")
        else:
            result = run_train(
                fold_args
            )  # Same original weights; never the previous fold's best.
            if args.dry_run:
                results.append(result)
                continue
            result.update(
                {
                    "fold": index,
                    "signature": signature,
                    "validation_pages": plan["folds"][index - 1]["val_stats"]["pages"],
                }
            )
            json_write(result_path, result)
        results.append(result)
        json_write(output / "summary.json", aggregate(results, plan["n_splits"]))
    if args.dry_run:
        return results
    # Include all already-completed folds when executing a subset of indices.
    results = [json.loads(p.read_text()) for p in sorted(output.glob("fold_*.json"))]
    if len({r["fold"] for r in results}) != len(results) or any(
        r["signature"] != signature
        or not 1 <= r["fold"] <= plan["n_splits"]
        or not Path(r["best"]).is_file()
        or digest(Path(r["best"]).read_bytes()) != r["best_sha256"]
        for r in results
    ):
        raise ValueError(
            "Completed fold results do not match this experiment/checkpoints"
        )
    summary = aggregate(results, plan["n_splits"])
    json_write(output / "summary.json", summary)
    summary_args = copy.deepcopy(args)
    summary_args.resume = None
    summary_args.name = f"{args.name}_cv_summary"
    summary_args.wandb_group = args.wandb_group or f"{args.name}-cv-{signature[:8]}"
    tracker = Tracking(summary_args, run_settings, "cross-validation-summary").start()
    try:
        tracker.summary(
            {
                "cv/" + metric + "/" + key: value
                for metric, values in summary["metrics"].items()
                for key, value in values.items()
                if value is not None
            }
        )
        tracker.summary(
            {
                "cv/complete": summary["complete"],
                "cv/completed_folds": summary["completed_folds"],
                "cv/per_class": summary["per_class"],
            }
        )
    finally:
        tracker.finish()
    print(json.dumps(summary, indent=2))
    return summary
