#!/usr/bin/env python3
"""Inspect -> bounded streaming subset -> DocLayout-YOLO fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import fields
from pathlib import Path

from indicdlp.data.prepare import prepare, verify_dataset
from indicdlp.data.ranges import Budget
from indicdlp.data.schema import Adapter
from indicdlp.data.source import HubSource
from indicdlp.engine.training import DEFAULT_WEIGHTS, run_train
from indicdlp.evaluation.runner import run_evaluate
from indicdlp.config import HYPER_DEFAULTS, apply_config, validate_training
from indicdlp.environment import load_environment


ROOT = Path(__file__).resolve().parent


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError(
            "Must be nonnegative (0 means unlimited for sample/shard limits)"
        )
    return result


def adapter_for(args):
    names = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "--classes must be a nonempty JSON array of unique class names"
        )
    options = {
        f.name: getattr(args, f.name) for f in fields(Adapter) if f.name != "names"
    }
    return Adapter(names=names, **options)


def add_source(parser, transfer):
    parser.add_argument("--repo", default="ai4bharat/indicdlp")
    parser.add_argument(
        "--revision",
        default="main",
        help="Commit SHA recommended for repeatable server runs",
    )
    parser.add_argument(
        "--max-transfer-mib",
        type=positive,
        default=transfer,
        help="Hard limit on Parquet response body bytes per invocation; metadata/API headers excluded",
    )
    parser.add_argument(
        "--max-group-mib",
        type=positive,
        default=256,
        help="Maximum uncompressed size of requested columns in one row group",
    )
    parser.add_argument(
        "--image-field", default="image", help="Top-level embedded image column"
    )
    parser.add_argument("--classes", default=str(ROOT / "resources/classes.json"))
    for name in ("type", "language", "filename", "annotations", "document-id"):
        parser.add_argument(
            f"--{name}-field",
            help="Dotted metadata field; default: conservative autodetection",
        )
    parser.add_argument(
        "--category-id-base",
        type=int,
        choices=[0, 1],
        help="Required for unnamed numeric annotation IDs; never inferred",
    )
    parser.add_argument(
        "--bbox-format",
        choices=["xywh", "xyxy", "yolo"],
        default="xywh",
        help="COCO pixel xywh by default. Native YOLO text is always normalized xywh.",
    )


def add_model(parser):
    parser.add_argument(
        "--data", default="data/indicdlp_forms", help="Prepared subset directory"
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Local checkpoint/config or hf://owner/repo/filename.pt",
    )
    parser.add_argument("--device", default="0", help="0, 0,1, or cpu")
    parser.add_argument("--imgsz", type=positive, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=nonnegative, default=4)
    parser.add_argument("--project", default="runs/indicdlp")
    parser.add_argument("--name", default="forms")
    parser.add_argument("--folds", help="Directory created by make-folds")
    parser.add_argument(
        "--fold", type=positive, help="Use this document-group fold (1-based)"
    )


def add_training(parser):
    add_model(parser)
    parser.add_argument(
        "--config", help="Training recipe YAML; explicit CLI flags override it"
    )
    parser.add_argument("--epochs", type=positive, default=100)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=positive, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        metavar="LAST_PT",
        help="Restore optimizer/scheduler from this checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify data and print options; no model/W&B initialization",
    )
    for key, value in HYPER_DEFAULTS.items():
        options = {"default": value}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value)
        parser.add_argument("--" + key.replace("_", "-"), **options)
    add_tracking(parser)


def add_tracking(parser):
    parser.add_argument(
        "--wandb", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT") or "indicdlp-doclayout",
    )
    parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None
    )
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb-log-model", action=argparse.BooleanOptionalAction, default=False
    )


def parser_for_cli():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect",
        help="Read a shard footer and metadata columns; no image-column reads",
    )
    add_source(inspect, 16)
    inspect.add_argument("--split", choices=["train", "val", "test"], default="train")
    inspect.add_argument("--rows", type=positive, default=3)
    prep = commands.add_parser(
        "prepare", help="Stream and save matching pages only; retain official splits"
    )
    add_source(prep, 2048)
    prep.add_argument("--output", default="data/indicdlp_forms")
    prep.add_argument(
        "--types",
        nargs="+",
        default=["forms"],
        help="Forms default; optionally notices and acts-and-rules. No official FIR category.",
    )
    prep.add_argument(
        "--languages", nargs="*", default=[], help="All by default; e.g. hi en mr"
    )
    prep.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    prep.add_argument("--max-train", type=nonnegative, default=2000)
    prep.add_argument("--max-val", type=nonnegative, default=250)
    prep.add_argument("--max-test", type=nonnegative, default=250)
    prep.add_argument(
        "--max-shards",
        type=nonnegative,
        default=0,
        help="Per split; 0 scans until exhausted/sample cap",
    )
    verify = commands.add_parser(
        "verify", help="Check hashes and regenerate paths after moving a subset"
    )
    verify.add_argument("--data", default="data/indicdlp_forms")
    train = commands.add_parser("train", help="Fine-tune using doclayout_yolo.YOLOv10")
    add_training(train)
    folds = commands.add_parser(
        "make-folds",
        help="Stratified document-group CV on train+val; official test held out",
    )
    folds.add_argument("--data", default="data/indicdlp_selected")
    folds.add_argument("--output", default="folds/indicdlp_selected")
    folds.add_argument("--n-splits", type=positive, default=5)
    folds.add_argument("--seed", type=int, default=42)
    grouping = folds.add_mutually_exclusive_group()
    grouping.add_argument(
        "--groups-csv", help="CSV with id,document_id or filename,document_id"
    )
    grouping.add_argument(
        "--group-regex", help="Verified filename regex capturing a source document ID"
    )
    folds.add_argument(
        "--allow-page-level",
        action="store_true",
        help="Explicitly allow missing document IDs; independence is not guaranteed",
    )
    crossval = commands.add_parser(
        "crossval", help="Train independent fold models and report validation mean/std"
    )
    add_training(crossval)
    crossval.add_argument(
        "--fold-indices",
        nargs="+",
        type=positive,
        help="Run selected folds; default all",
    )
    crossval.add_argument(
        "--resume-fold",
        type=positive,
        help="Restore this interrupted fold using --resume LAST_PT",
    )
    evaluate = commands.add_parser(
        "evaluate",
        aliases=["test"],
        help="Report validation/held-out test metrics locally and in W&B",
    )
    add_model(evaluate)
    add_tracking(evaluate)
    evaluate.add_argument(
        "--config",
        help="Use the same recipe's evaluation settings and W&B configuration",
    )
    evaluate.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--max-det", type=positive, default=1000)
    evaluate.add_argument("--conf", type=float, default=0.001)
    evaluate.add_argument("--iou", type=float, default=0.7)
    evaluate.add_argument(
        "--report-dir", help="Fresh output directory for JSON/CSV/Markdown reports"
    )
    return parser


def inspect_source(source, adapter, args):
    shards = [s for s in source.shards if s.split == args.split]
    if not shards:
        raise ValueError(f"No shards for {args.split}")
    print(
        f"Repository: {source.repo}\nPinned revision: {source.revision}\nShards: {len(source.shards)}"
    )
    with source.open(shards[0]) as handle:
        print("Schema:\n", handle.parquet.schema_arrow)
        print("Metadata projection:", handle.columns)
        print("HF ClassLabels:", json.dumps(handle.mappings, ensure_ascii=False))
        rows = handle.metadata_rows(0)[: args.rows]
        for row in rows:
            print(
                "Metadata sample:",
                json.dumps(row, ensure_ascii=False, default=str)[:6000],
            )
            try:
                print("Detected:", adapter.metadata(row, handle.mappings))
            except ValueError as exc:
                print("Adapter needs field overrides:", exc)
    print(
        f"Parquet payload read: {source.budget.used:,} bytes; image column not requested"
    )


def main(argv=None):
    parser = parser_for_cli()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.command in ("evaluate", "test"):
        try:
            apply_config(args, argv, available_only=True)
        except ValueError as exc:
            parser.error(str(exc))
        if args.wandb not in ("disabled", "offline", "online"):
            parser.error("Invalid W&B mode in evaluation config")
    if args.command in ("train", "crossval"):
        try:
            apply_config(args, argv)
            validate_training(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.command == "crossval" and (not args.folds or args.fold):
            parser.error("crossval requires --folds; select folds with --fold-indices")
        if args.command == "train" and bool(args.folds) != bool(args.fold):
            parser.error("Use --folds and --fold together")
    if args.command in ("train", "crossval", "evaluate", "test"):
        if args.batch == 0 or args.batch < -1:
            parser.error("--batch must be positive or -1")
        if args.command in ("evaluate", "test") and args.batch < 1:
            parser.error("Evaluation requires a positive --batch")
        if args.imgsz % 32:
            parser.error("--imgsz must be a multiple of 32")
        if args.command == "train" and args.lr0 <= 0:
            parser.error("--lr0 must be positive")
    if args.command == "train":
        return run_train(args)
    if args.command == "crossval":
        from indicdlp.engine.crossval import run_crossval

        return run_crossval(args)
    if args.command == "make-folds":
        from indicdlp.data.folds import make_folds

        return make_folds(
            args.data,
            args.output,
            args.n_splits,
            args.seed,
            args.groups_csv,
            args.group_regex,
            args.allow_page_level,
        )
    if args.command == "test" and args.split != "test":
        parser.error(
            "test uses the held-out test split; use evaluate --split val for validation"
        )
    if args.command in ("evaluate", "test"):
        if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
            parser.error("Invalid evaluation confidence/IoU threshold")
        return run_evaluate(args)
    if args.command == "verify":
        print(verify_dataset(args.data))
        return
    if args.command == "prepare" and not {"train", "val"}.issubset(args.splits):
        parser.error("prepare requires train and val; test is optional")
    adapter = adapter_for(args)
    source = HubSource(
        args.repo,
        args.revision,
        Budget(args.max_transfer_mib * 2**20),
        args.image_field,
        args.max_group_mib,
    )
    try:
        if args.command == "inspect":
            return inspect_source(source, adapter, args)
        limits = {s: getattr(args, f"max_{s}") for s in args.splits}
        report = prepare(
            source,
            args.output,
            adapter,
            args.types,
            args.languages,
            limits,
            args.max_shards,
        )
        print(json.dumps(report, indent=2))
        print(f"Ready: {Path(args.output).resolve() / 'data.yaml'}")
    finally:
        source.close()


def entrypoint():
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
