"""One real DocLayout-YOLO optimizer step on synthetic boxes, without downloads.

Run after installing requirements-train.txt. This deliberately builds the model
from the package's architecture YAML instead of downloading pretrained weights.
It tests the detection loss and gradients; it is not a model-quality evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import doclayout_yolo
    from doclayout_yolo.models.yolov10.model import YOLOv10DetectionModel
    from doclayout_yolo.utils import DEFAULT_CFG_DICT

    torch.manual_seed(42)
    torch.set_num_threads(2)
    architecture = (
        Path(doclayout_yolo.__file__).parent / "cfg/models/v10/yolov10m-doclayout.yaml"
    )
    names = json.loads(
        (
            Path(__file__).resolve().parents[1] / "src/indicdlp/resources/classes.json"
        ).read_text()
    )
    model = YOLOv10DetectionModel(str(architecture), nc=len(names), verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[21.0], [36.0]]),
        "resized_shape": [(64, 64), (64, 64)],
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
    }
    before = next(model.parameters()).detach().clone()
    loss, components = model(batch)
    assert torch.isfinite(loss).all(), "Non-finite detection loss"
    loss.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()).detach()), (
        "Optimizer did not update weights"
    )
    result = {
        "status": "passed",
        "model": architecture.name,
        "classes": len(names),
        "torch": torch.__version__,
        "doclayout_yolo": doclayout_yolo.__version__,
        "device": "cpu",
        "batch": 2,
        "imgsz": 64,
        "steps": 1,
        "loss": loss.detach().tolist(),
        "components": components.detach().tolist(),
        "dataset_download_bytes": 0,
        "pretrained_download_bytes": 0,
    }
    output = Path(__file__).resolve().parents[1] / "artifacts/model_smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
