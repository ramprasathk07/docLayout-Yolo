"""Explicit W&B lifecycle and scalar callbacks; tokens and document images are never logged."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

from ..data.prepare import json_write
from ..evaluation.metrics import scalars, snapshot
from ..evaluation.report import write_report
from ..environment import load_environment


def disable_native_wandb():
    # The upstream integration owns init/finish and would duplicate our fold runs.
    # Modify only this process's callback registry, not persistent user settings.
    native = importlib.import_module("doclayout_yolo.utils.callbacks.wb")
    native.callbacks.clear()


def online_key():
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Online W&B requires WANDB_API_KEY in the environment or repository .env. "
            "No interactive login is attempted. Use --wandb offline for local logging."
        )
    return key


def authenticate_online(sdk):
    """Use the supplied environment key, never a prompt or legacy key-length rule."""
    key = online_key()
    try:
        version = tuple(int(part) for part in sdk.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        version = (0, 0)
    if version < (0, 31):
        raise RuntimeError(
            "W&B SDK is too old for this project's authentication flow. Run: "
            "python -m pip install --upgrade 'wandb>=0.31,<0.32'. "
            "Older SDKs may reject valid long API keys; do not truncate the key."
        )
    options = {"key": key, "prompt": False, "verify": True}
    host = os.environ.get("WANDB_BASE_URL", "").strip()
    if host:
        options["host"] = host
    try:
        authenticated = sdk.login(**options)
    except Exception as exc:
        # SDK messages can contain supplied credentials. Do not echo them.
        raise RuntimeError(
            f"W&B authentication failed ({type(exc).__name__}). "
            "Check WANDB_API_KEY, WANDB_BASE_URL, and server connectivity. "
            "No interactive login was attempted."
        ) from None
    if not authenticated:
        raise RuntimeError(
            "W&B authentication did not succeed. Check WANDB_API_KEY and WANDB_BASE_URL. "
            "No interactive login was attempted."
        )


class Tracking:
    def __init__(self, args, config, job_type="train"):
        self.args, self.run, self.sdk = args, None, None
        self.config, self.job_type = config, job_type
        self.started = time.monotonic()

    def start(self):
        if getattr(self.args, "wandb", "disabled") == "disabled":
            return self
        load_environment()
        if self.args.wandb == "online":
            online_key()
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "Install W&B: python -m pip install -r requirements-train.txt"
            ) from exc
        self.sdk = wandb
        if self.args.wandb == "online":
            authenticate_online(wandb)
        previous = None
        checkpoint = getattr(self.args, "resume", None)
        if checkpoint:
            path = Path(checkpoint).resolve().parent.parent / "wandb_run.json"
            if path.exists():
                previous = json.loads(path.read_text())
        mode = self.args.wandb
        options = dict(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.name,
            group=self.args.wandb_group,
            tags=self.args.wandb_tags,
            job_type=self.job_type,
            mode=mode,
            config=self.config,
            save_code=False,
            settings=wandb.Settings(disable_git=True),
        )
        if previous and previous["mode"] == mode == "online":
            # Reuse the actual saved account/project, not potentially changed CLI defaults.
            options.update(
                id=previous["id"],
                resume="must",
                project=previous["project"],
                entity=previous["entity"],
            )
        self.run = wandb.init(**options)
        self.run.define_metric("epoch")
        self.run.define_metric("*", step_metric="epoch")
        return self

    def attach(self, model):
        if not hasattr(model, "add_callback"):
            return
        if self.run is not None:
            model.add_callback("on_pretrain_routine_start", self.on_start)
        model.add_callback("on_fit_epoch_end", self.on_epoch)
        model.add_callback("on_train_end", self.on_end)

    def on_start(self, trainer):
        if self.run is None:
            return
        json_write(
            Path(trainer.save_dir) / "wandb_run.json",
            {
                "id": self.run.id,
                "project": self.run.project,
                "entity": self.run.entity,
                "mode": self.args.wandb,
            },
        )
        effective = {
            key: getattr(trainer.args, key)
            for key in self.config.get("train", {})
            if hasattr(trainer.args, key)
        }
        self.run.config.update({"effective_train": effective}, allow_val_change=True)

    def on_epoch(self, trainer):
        epoch = trainer.epoch + 1
        values = {
            "epoch": epoch,
            **scalars(trainer.metrics),
            **scalars(getattr(trainer, "lr", {})),
        }
        if getattr(trainer, "tloss", None) is not None:
            values.update(
                scalars(trainer.label_loss_items(trainer.tloss, prefix="train"))
            )
        values["time/elapsed_seconds"] = time.monotonic() - self.started
        validator = getattr(trainer, "validator", None)
        metrics = getattr(validator, "metrics", None)
        if metrics is not None:
            report = snapshot(metrics, validator, diagnostics=False)
            values.update({"val/" + k: v for k, v in report["aggregate"].items()})
            values.update(self.class_scalars(report, "val"))
        json_write(
            Path(trainer.save_dir) / "reports/epochs" / f"{epoch:04d}.json", values
        )
        if self.run is not None:
            self.run.log(values, step=epoch)

    @staticmethod
    def class_scalars(report, prefix):
        return {
            f"{prefix}/classes/{cid}_{row['name']}/{key}": value
            for cid, row in report["per_class"].items()
            for key, value in scalars(
                {k: v for k, v in row.items() if k != "evaluated"}
            ).items()
        }

    def on_end(self, trainer):
        validator = getattr(trainer, "validator", None)
        if getattr(validator, "metrics", None) is None:
            return
        report = snapshot(validator.metrics, validator)
        report.update(split="validation_best", epoch=trainer.epoch + 1)
        path = write_report(Path(trainer.save_dir) / "reports/validation_best", report)
        self.report(report, "validation_best", path)

    def report(self, report, prefix, path):
        if self.run is None:
            return
        values = {
            f"{prefix}/{key}": value for key, value in report["aggregate"].items()
        }
        values.update(self.class_scalars(report, prefix))
        values.update(
            {
                f"{prefix}/speed_ms/{key}": value
                for key, value in report["speed_ms_per_image"].items()
            }
        )
        values["epoch"] = report.get("epoch", 0)
        columns = [
            "class_id",
            "name",
            "targets",
            "evaluated",
            "precision",
            "recall",
            "f1",
            "map50",
            "map75",
            "map50_95",
        ]
        values[f"{prefix}/per_class"] = self.sdk.Table(
            columns=columns,
            data=[
                [cid] + [row.get(k) for k in columns[1:]]
                for cid, row in report["per_class"].items()
            ],
        )
        matrix = report.get("confusion_matrix")
        if matrix:
            values.update(
                {
                    f"{prefix}/confusion/{key}": value
                    for key, value in scalars(
                        {
                            "confidence_threshold": matrix.get("confidence_threshold"),
                            "iou_threshold": matrix.get("iou_threshold"),
                        }
                    ).items()
                }
            )
            labels = matrix["labels"]
            values[f"{prefix}/confusion_matrix"] = self.sdk.Table(
                columns=["predicted", "actual", "count"],
                data=[
                    [labels[i], labels[j], count]
                    for i, row in enumerate(matrix["counts"])
                    for j, count in enumerate(row)
                ],
            )
        for curve in report.get("curves", []):
            if not curve["x"] or not curve["y"]:
                continue
            # Keep plots below W&B's table row limit; artifacts retain every original point.
            stride = max(1, (len(curve["x"]) + 199) // 200)
            keys = [report["per_class"][str(cid)]["name"] for cid in curve["class_ids"]]
            values[f"{prefix}/curves/{curve['name']}"] = self.sdk.plot.line_series(
                xs=curve["x"][::stride],
                ys=[row[::stride] for row in curve["y"]],
                keys=keys,
                title=curve["name"],
                xname=curve["x_label"],
            )
        self.run.log(values)
        self.run.summary.update(
            {f"{prefix}/{key}": value for key, value in report["aggregate"].items()}
        )
        artifact = self.sdk.Artifact(
            f"{self.run.id}-{prefix.replace('/', '-')}-metrics", type="evaluation"
        )
        for filename in (
            "metrics.json",
            "per_class.csv",
            "confusion_matrix.csv",
            "curves.json",
            "summary.md",
        ):
            file = Path(path) / filename
            if file.exists():
                artifact.add_file(str(file))
        self.run.log_artifact(artifact)

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(values)

    def artifact(self, path):
        if (
            self.run is None
            or not self.args.wandb_log_model
            or not Path(path).is_file()
        ):
            return
        artifact = self.sdk.Artifact(f"doclayout-{self.run.id}", type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=["best"])

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
