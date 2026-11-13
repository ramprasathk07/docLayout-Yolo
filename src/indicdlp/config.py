"""Versioned training recipes. Explicit CLI arguments override YAML values."""

import math
from pathlib import Path

import yaml


HYPER_DEFAULTS = {
    "optimizer": "AdamW",
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,
    "warmup_momentum": 0.8,
    "momentum": 0.9,
    "nbs": 64,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.1,
    "degrees": 1.0,
    "translate": 0.02,
    "scale": 0.1,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 0,
    "max_det": 1000,
    "save_period": 10,
    "cache": False,
    "cos_lr": True,
    "deterministic": True,
}
TRAIN_KEYS = set(HYPER_DEFAULTS) | {
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "device",
    "lr0",
    "patience",
    "amp",
    "seed",
}
TRACKING_KEYS = {
    "mode": "wandb",
    "project": "wandb_project",
    "entity": "wandb_entity",
    "group": "wandb_group",
    "tags": "wandb_tags",
    "log_model": "wandb_log_model",
}


def apply_config(args, argv, available_only=False):
    if not getattr(args, "config", None):
        return
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) - {
        "train",
        "tracking",
        "description",
    }:
        raise ValueError(
            "Config must contain train/tracking sections (optional description)"
        )
    explicit = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    for section, mapping in (
        ("train", {k: k for k in TRAIN_KEYS}),
        ("tracking", TRACKING_KEYS),
    ):
        values = config.get(section, {})
        if not isinstance(values, dict) or set(values) - set(mapping):
            raise ValueError(
                f"Unknown keys or invalid mapping in config section {section}"
            )
        for key, value in values.items():
            dest = mapping[key]
            if not hasattr(args, dest):
                if available_only:
                    continue
                raise ValueError(f"Config key {key} is unavailable for this command")
            flag = "--" + dest.replace("_", "-")
            if flag in explicit or "--no-" + dest.replace("_", "-") in explicit:
                continue
            default = getattr(args, dest)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif isinstance(default, list):
                valid = isinstance(value, list) and all(
                    isinstance(v, str) for v in value
                )
            else:
                valid = value is None or isinstance(value, str)
            if not valid:
                raise ValueError(f"Invalid value type for config {section}.{key}")
            setattr(args, dest, value)


def validate_training(args):
    if (
        not isinstance(args.device, str)
        or not args.device
        or (
            args.device != "cpu"
            and (
                not all(v.isdigit() for v in args.device.split(","))
                or len(set(args.device.split(","))) != len(args.device.split(","))
            )
        )
    ):
        raise ValueError("device must be cpu or unique comma-separated GPU indices")
    for key in TRAIN_KEYS:
        value = getattr(args, key)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if (
        args.epochs < 1
        or args.imgsz < 32
        or args.imgsz % 32
        or args.batch == 0
        or args.batch < -1
    ):
        raise ValueError(
            "epochs must be positive; imgsz a multiple of 32; batch positive or -1"
        )
    if args.workers < 0 or args.patience < 1 or args.nbs < 1 or args.max_det < 1:
        raise ValueError("workers must be nonnegative; patience, nbs, max_det positive")
    if (
        args.lr0 <= 0
        or not 0 < args.lrf <= 1
        or args.weight_decay < 0
        or args.warmup_epochs < 0
    ):
        raise ValueError("Invalid learning rate, weight decay, or warmup")
    if args.optimizer not in ("AdamW", "SGD", "Adam", "auto"):
        raise ValueError("Supported optimizers: AdamW, SGD, Adam, auto")
    if (
        args.close_mosaic < 0
        or min(args.box, args.cls, args.dfl, args.warmup_bias_lr) < 0
    ):
        raise ValueError("Loss gains, warmup bias LR, close_mosaic must be nonnegative")
    if not 0 <= args.momentum < 1 or not 0 <= args.warmup_momentum <= 1:
        raise ValueError("Invalid optimizer/warmup momentum")
    for key in (
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "copy_paste",
        "translate",
        "scale",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    ):
        if not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if args.wandb not in ("disabled", "offline", "online"):
        raise ValueError("W&B mode must be disabled, offline, or online")
