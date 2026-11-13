import io
import json
from contextlib import contextmanager

import pytest
from PIL import Image

from indicdlp.data.prepare import prepare, read_manifest, verify_dataset
from indicdlp.data.ranges import Budget, BudgetExceeded, RangeReader
from indicdlp.data.schema import Adapter
from indicdlp.data.source import ParquetShard, Shard, metadata_columns, split_for


from tests.helpers import Session


def test_range_seek_and_budget():
    session, budget = Session(b"abcdefghij"), Budget(6)
    reader = RangeReader("https://fixture", 10, session, budget)
    assert reader.read(2) == b"ab"
    reader.seek(-2, 2)
    assert reader.read(2) == b"ij"
    reader.seek(0)
    with pytest.raises(BudgetExceeded):
        reader.read(3)
    assert len(session.calls) == 2  # Budget checked BEFORE any request.
    assert budget.used == 4


def test_whole_shard_response_rejected_before_body_read():
    session = Session(b"x" * 1000, status=200)
    with pytest.raises(RuntimeError, match="non-range"):
        RangeReader("https://fixture", 1000, session, Budget(10)).read(2)
    assert session.last_response.raw.tell() == 0


@pytest.mark.parametrize(
    "path,split",
    [
        ("data/train-001.parquet", "train"),
        ("data/validation/000.parquet", "val"),
        ("val/000.parquet", "val"),
        ("data/test-0.parquet", "test"),
        ("data/other.parquet", None),
    ],
)
def test_split_discovery(path, split):
    assert split_for(path) == split


def make_parquet(split):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = []
    for i, dtype in enumerate(["Newspapers", "Newspapers", "Forms", "Forms"]):
        image = Image.new(
            "RGB",
            (100, 200),
            (i * 30, {"train": 20, "val": 80, "test": 140}[split], 10),
        )
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        rows.append(
            {
                "image": {"bytes": encoded.getvalue(), "path": f"{split}_{i}.png"},
                "document_type": dtype,
                "language": "Hindi",
                "annotations": json.dumps(
                    [{"bbox": [10, 20, 30, 50], "label": "paragraph"}]
                ),
            }
        )
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer, row_group_size=2)
    return buffer.getvalue()


class Source:
    repo, revision = "fixture/indicdlp", "synthetic-revision"

    def __init__(self):
        self.payloads = {
            f"data/{s}-0.parquet": make_parquet(s) for s in ("train", "val", "test")
        }
        self.shards = [Shard(p, len(v), split_for(p)) for p, v in self.payloads.items()]
        self.budget = Budget(2**20)
        self.image_reads = []

    @contextmanager
    def open(self, shard):
        import pyarrow.parquet as pq

        session = Session(self.payloads[shard.path])
        with RangeReader("https://fixture", shard.size, session, self.budget) as reader:
            handle = ParquetShard(
                pq.ParquetFile(reader, pre_buffer=False), "image", 2**20
            )
            real_images = handle.images

            def images(group):
                self.image_reads.append((shard.path, group))
                return real_images(group)

            handle.images = images
            yield handle


def test_real_arrow_projection_excludes_image_bytes():
    source = Source()
    with source.open(source.shards[0]) as handle:
        assert "image" not in handle.columns
        assert "image.path" in handle.columns
        row = handle.metadata_rows(0)[0]
        assert row["image"] == {"path": "train_0.png"}
        assert "bytes" not in row["image"]
    assert not source.image_reads


def test_end_to_end_prepare_resume_and_tamper_detection(tmp_path):
    source = Source()
    adapter = Adapter(["paragraph", "table"])
    report = prepare(
        source,
        tmp_path / "data",
        adapter,
        ["forms"],
        ["hi"],
        {"train": 2, "val": 1, "test": 1},
    )
    assert report["selected"] == {"train": 2, "val": 1, "test": 1}
    assert len(source.image_reads) == 3
    assert all(
        group == 1 for _, group in source.image_reads
    )  # Newspaper-only groups never load images.
    root = tmp_path / "data"
    assert verify_dataset(root).exists()
    records = read_manifest(root)
    assert len(records) == 4
    assert {r["document_type"] for r in records.values()} == {"forms"}
    assert not list(root.rglob("*.parquet"))
    prepare(source, root, adapter, ["forms"], ["hi"], {"train": 2, "val": 1, "test": 1})
    assert len(source.image_reads) == 3  # No refetch on completed resume.
    first = next(iter(records.values()))
    (root / first["label"]).write_text("broken")
    with pytest.raises(ValueError, match="changed prepared file"):
        verify_dataset(root)


def test_failed_prepare_is_resumable_and_cannot_train(tmp_path):
    source = Source()
    source.budget.limit = 1
    root = tmp_path / "data"
    with pytest.raises(BudgetExceeded):
        prepare(
            source, root, Adapter(["paragraph"]), ["forms"], [], {"train": 1, "val": 1}
        )
    assert (root / "incomplete.json").exists()
    with pytest.raises(ValueError, match="Preparation incomplete"):
        verify_dataset(root)
    source.budget.limit = 2**20
    prepare(source, root, Adapter(["paragraph"]), ["forms"], [], {"train": 1, "val": 1})
    assert verify_dataset(root).exists()
    assert not (root / "incomplete.json").exists()


def test_selection_change_rejected(tmp_path):
    source = Source()
    root = tmp_path / "data"
    prepare(source, root, Adapter(["paragraph"]), ["forms"], [], {"train": 1, "val": 1})
    with pytest.raises(ValueError, match="selection/schema/revision differs"):
        prepare(
            source,
            root,
            Adapter(["paragraph"]),
            ["notices"],
            [],
            {"train": 1, "val": 1},
        )


def test_selection_file_tamper_is_detected(tmp_path):
    source = Source()
    root = tmp_path / "data"
    prepare(source, root, Adapter(["paragraph"]), ["forms"], [], {"train": 1, "val": 1})
    path = root / "selection.json"
    path.write_text(path.read_text().replace('"paragraph"', '"table"'))
    with pytest.raises(ValueError, match="Selection/schema changed"):
        verify_dataset(root)


def test_moved_dataset_gets_new_paths(tmp_path):
    source = Source()
    root = tmp_path / "original"
    prepare(source, root, Adapter(["paragraph"]), ["forms"], [], {"train": 1, "val": 1})
    moved = tmp_path / "server_path"
    root.rename(moved)
    config = verify_dataset(moved)
    assert moved.resolve().as_posix() in config.read_text()


def test_row_group_memory_guard():
    source = Source()
    with source.open(source.shards[0]) as handle:
        handle.max_group_bytes = 1
        with pytest.raises(RuntimeError, match="max-group-mib"):
            handle.images(1)


def test_binary_metadata_fields_are_excluded():
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema(
        [
            ("image", pa.binary()),
            ("attachment", pa.list_(pa.binary())),
            ("document_type", pa.string()),
        ]
    )
    assert metadata_columns(schema, "image") == ["document_type"]
