"""Synthetic data and SDK fixtures shared by offline tests."""

import io
import json
import re
import sys
from types import SimpleNamespace

from PIL import Image

from indicdlp.data.prepare import finalize, json_write, write_sample
from indicdlp.data.schema import Adapter


def make_prepared(root, with_groups=True, overlap=False):
    root.mkdir()
    adapter = Adapter(["paragraph", "table"])
    json_write(root / "selection.json", {"adapter": vars(adapter)})
    records = {}
    for doc in range(14):
        for page in range(2):
            sid = f"doc{doc:02d}_page{page}"
            split = "train" if doc < 9 else "val" if doc < 12 else "test"
            buffer = io.BytesIO()
            Image.new("RGB", (20, 30), (doc * 15, page * 80, 10)).save(
                buffer, format="PNG"
            )
            group = f"doc{doc:02d}" if with_groups else None
            if overlap and doc == 12:
                group = "doc00"
            metadata = {
                "filename": f"doc{doc:02d}_page_{page:02d}.png",
                "document_id": group,
                "document_type": "forms",
                "language": "hindi" if doc % 2 else "english",
            }
            records[sid] = write_sample(
                root,
                sid,
                split,
                buffer.getvalue(),
                {"annotations": [{"label": "paragraph", "bbox": [0, 0, 10, 20]}]},
                metadata,
                adapter,
                {},
            )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records.values())
    )
    finalize(root, records, adapter.names, {})
    return records


class Response:
    def __init__(self, payload, start, end, status=206):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        self.raw = io.BytesIO(payload[start : end + 1])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Session:
    def __init__(self, payload, status=206):
        self.payload, self.status, self.calls = payload, status, []

    def get(self, url, headers, **kwargs):
        start, end = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups()
        )
        self.calls.append((start, end))
        self.last_response = Response(self.payload, start, end, self.status)
        return self.last_response


class Config(dict):
    def update(self, value, **kwargs):
        super().update(value)


class Run:
    id, project, entity = "run-123", "project", "account"

    def __init__(self):
        self.summary, self.config, self.logged, self.finished = {}, Config(), [], []

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values, **kwargs):
        self.logged.append((values, kwargs))

    def finish(self, **kwargs):
        self.finished.append(kwargs)


def fake_wandb_sdk(monkeypatch):
    calls, runs = [], []

    def init(**kwargs):
        calls.append(kwargs)
        run = Run()
        runs.append(run)
        return run

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw)
    )
    return calls, runs
