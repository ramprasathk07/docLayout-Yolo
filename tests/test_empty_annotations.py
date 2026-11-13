"""Bad/empty targets are audited without stopping or creating false negatives."""

import io
from contextlib import contextmanager

import pytest

from indicdlp.data.prepare import prepare, read_manifest, verify_dataset
from indicdlp.data.ranges import BudgetExceeded
from indicdlp.data.schema import Adapter
from indicdlp.data.source import Shard, split_for
from tests.test_streaming import NativeSource


class CurationSource(NativeSource):
    def __init__(self, no_positive=False, bad_class=False):
        super().__init__()
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        for path, data in self.payloads.items():
            table = pq.read_table(io.BytesIO(data))
            rows = table.to_pylist()
            for index, row in enumerate(rows):
                row["image"]["path"] = f"fm_hi_00020{index}_0.png"
                if index < 2 or (no_positive and index == 2):
                    row.update(bboxes=[], category_ids=[])
                elif index == 3:
                    row.update(bboxes=[[999.0, 999.0, 20.0, 20.0]], category_ids=[0])
                elif bad_class:
                    row["category_ids"] = [99]
            result = io.BytesIO()
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema),
                result,
                row_group_size=2,
            )
            self.payloads[path] = result.getvalue()
        self.shards = [
            Shard(path, len(data), split_for(path))
            for path, data in self.payloads.items()
        ]


def export(source, root):
    return prepare(
        source,
        root,
        Adapter(["paragraph"], category_id_base=0),
        ["forms"],
        [],
        {"train": 0, "val": 0, "test": 0},
    )


def test_empty_and_all_invalid_pages_are_audited_and_not_negatives(tmp_path):
    source = CurationSource()
    root = tmp_path / "data"
    report = export(source, root)
    assert report["selected"] == {"train": 1, "val": 1, "test": 1}
    assert report["skipped_pages"] == 9
    assert report["skipped_reasons"] == {
        "empty_annotations": 6,
        "all_boxes_outside_or_degenerate": 3,
    }
    assert report["dropped_boxes_in_skipped_pages"] == 3
    assert report["dropped_boxes"] == 0
    assert len(source.image_reads) == 3 and all(
        group == 1 for _, group in source.image_reads
    )
    assert len(list(root.glob("images/*/*.png"))) == 3
    assert all(p.read_text().strip() for p in root.glob("labels/*/*.txt"))
    skipped = read_manifest(root, "skipped.jsonl")
    assert len(skipped) == 9
    for record in skipped.values():
        assert record["shard"] and record["filename"] and record["reason"]
        if record["reason"] == "all_boxes_outside_or_degenerate":
            assert record["width"] == 100 and record["height"] == 200
    assert verify_dataset(root).exists()
    second = export(source, root)
    assert len(source.image_reads) == 3  # Already audited pages do not refetch images.
    assert second["skipped_pages"] == 9
    assert len(read_manifest(root, "skipped.jsonl")) == 9


def test_audited_skips_survive_interrupted_preparation(tmp_path):
    source = CurationSource()
    original_open = source.open

    @contextmanager
    def interrupted_open(shard):
        with original_open(shard) as handle:

            def interrupted_images(group):
                raise BudgetExceeded("simulated interruption before image fetch")

            handle.images = interrupted_images
            yield handle

    source.open = interrupted_open
    root = tmp_path / "data"
    with pytest.raises(BudgetExceeded):
        export(source, root)
    assert len(read_manifest(root, "skipped.jsonl")) == 2
    assert not (root / "complete.json").exists()
    source.open = original_open
    assert export(source, root)["skipped_pages"] == 9
    assert verify_dataset(root).exists()


def test_unknown_class_ids_still_fail_instead_of_being_skipped(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="Class ID out of range"):
        export(CurationSource(bad_class=True), root)
    assert not (root / "complete.json").exists()
    assert {r["reason"] for r in read_manifest(root, "skipped.jsonl").values()} == {
        "empty_annotations"
    }


def test_no_positive_pages_cannot_be_marked_training_ready(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="Need at least one"):
        export(CurationSource(no_positive=True), root)
    assert len(read_manifest(root, "skipped.jsonl")) == 12
    assert not (root / "complete.json").exists()


def test_completed_skip_manifest_cannot_be_changed_silently(tmp_path):
    root = tmp_path / "data"
    export(CurationSource(), root)
    path = root / "skipped.jsonl"
    path.write_text(path.read_text().replace('"empty_annotations"', '"changed"'))
    with pytest.raises(ValueError, match="Skipped-page manifest changed"):
        verify_dataset(root)
