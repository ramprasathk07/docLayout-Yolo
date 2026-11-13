"""Unverifiable document metadata is excluded before fetching image columns."""

import io
from contextlib import contextmanager

import pytest

from indicdlp.data.prepare import prepare, read_manifest, verify_dataset
from indicdlp.data.ranges import BudgetExceeded
from indicdlp.data.schema import Adapter
from indicdlp.data.source import Shard, split_for
from tests.test_streaming import NativeSource, Source


def rewrite(source, transform):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for path, data in source.payloads.items():
        table = pq.read_table(io.BytesIO(data))
        rows = table.to_pylist()
        for index, row in enumerate(rows):
            transform(index, row)
        result = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema), result, row_group_size=2
        )
        source.payloads[path] = result.getvalue()
    source.shards = [
        Shard(path, len(data), split_for(path))
        for path, data in source.payloads.items()
    ]
    return source


def unknown_source(all_unknown=False, bad_class=False):
    def transform(index, row):
        if index < 2 or all_unknown:
            row["image"]["path"] = "zz_hi_000200_0.png" if index % 2 == 0 else None
            # Metadata verification must happen before empty-target handling.
            row.update(bboxes=[], category_ids=[])
        elif bad_class and index == 2:
            row["category_ids"] = [99]

    return rewrite(NativeSource(), transform)


def export(source, root, languages=()):
    return prepare(
        source,
        root,
        Adapter(["paragraph"], category_id_base=0),
        ["forms"],
        list(languages),
        {"train": 0, "val": 0, "test": 0},
    )


def test_unknown_and_missing_filenames_are_audited_before_images(tmp_path):
    source, root = unknown_source(), tmp_path / "data"
    report = export(source, root)
    assert report["selected"] == {"train": 1, "val": 1, "test": 1}
    assert report["skipped_reasons"] == {"unverified_metadata": 6}
    assert report["skipped_by_split"] == {"train": 2, "val": 2, "test": 2}
    assert len(source.image_reads) == 3
    assert all(group == 1 for _, group in source.image_reads)
    skipped = read_manifest(root, "skipped.jsonl")
    for record in skipped.values():
        assert "Cannot identify document type safely" in record["error"]
        assert record["metadata_fields"] == ["bboxes", "category_ids", "image"]
        assert record["shard"] and record["row_group"] == 0
        assert record["document_type"] is None
        assert record["filename"] in ("zz_hi_000200_0.png", None)
    assert len(list(root.glob("images/*/*.png"))) == 3
    assert verify_dataset(root).exists()
    assert export(source, root)["skipped_pages"] == 6
    assert len(source.image_reads) == 3
    assert read_manifest(root, "skipped.jsonl") == skipped


def test_resume_keeps_saved_pages_and_durable_metadata_skips(tmp_path):
    source, root = unknown_source(), tmp_path / "data"
    original_open = source.open

    @contextmanager
    def interrupted_open(shard):
        if shard.split == "val":
            raise BudgetExceeded("simulated interruption after train was saved")
        with original_open(shard) as handle:
            yield handle

    source.open = interrupted_open
    with pytest.raises(BudgetExceeded):
        export(source, root)
    saved = read_manifest(root)
    assert len(saved) == 1
    assert len(read_manifest(root, "skipped.jsonl")) == 2
    assert not (root / "complete.json").exists()
    source.open = original_open
    report = export(source, root)
    assert report["skipped_pages"] == 6
    assert all(read_manifest(root)[sid] == record for sid, record in saved.items())
    assert len(source.image_reads) == 3  # Train image group was not fetched twice.
    assert verify_dataset(root).exists()


def test_all_unknown_pages_cannot_be_marked_ready(tmp_path):
    source, root = unknown_source(all_unknown=True), tmp_path / "data"
    with pytest.raises(ValueError, match="Need at least one"):
        export(source, root)
    assert len(read_manifest(root, "skipped.jsonl")) == 12
    assert not source.image_reads
    assert not (root / "complete.json").exists()


def test_unknown_region_ids_remain_fatal_after_metadata_skips(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="Class ID out of range"):
        export(unknown_source(bad_class=True), root)
    assert len(read_manifest(root, "skipped.jsonl")) == 2
    assert not (root / "complete.json").exists()


@pytest.mark.parametrize(
    "languages,expected_skips,expected_pages", [(["hi"], 6, 2), ([], 0, 4)]
)
def test_missing_language_is_skipped_only_when_filter_requires_it(
    tmp_path, languages, expected_skips, expected_pages
):
    def transform(index, row):
        row["document_type"] = "Forms"
        if index < 2:
            row["language"] = None

    source, root = rewrite(Source(), transform), tmp_path / "data"
    report = export(source, root, languages)
    assert report["selected"] == dict.fromkeys(("train", "val", "test"), expected_pages)
    assert report["skipped_pages"] == expected_skips
    assert report["skipped_reasons"] == (
        {"unverified_language": 6} if languages else {}
    )
    assert len(source.image_reads) == (3 if languages else 6)
