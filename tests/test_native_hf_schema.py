"""Authenticated server metadata reproduced without downloading dataset images."""

from pathlib import Path

import pytest

from indicdlp.data.schema import Adapter, indicdlp_filename_metadata


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("ar", "acts-and-rules"),
        ("br", "brochures"),
        ("fm", "forms"),
        ("mg", "magazines"),
        ("mn", "manuals"),
        ("np", "newspapers"),
        ("nt", "notices"),
        ("nv", "novels"),
        ("qp", "question-papers"),
        ("rp", "research-papers"),
        ("sy", "syllabi"),
        ("tb", "textbooks"),
    ],
)
def test_official_filename_prefixes(prefix, expected):
    adapter = Adapter(["paragraph"])
    result = adapter.metadata({"image": {"path": f"{prefix}_as_000209_0.png"}}, {})
    assert result["document_type"] == expected
    assert result["language"] == "assamese"
    assert (
        result["document_id"] is None
    )  # Numeric filename components are not proven source-document IDs.


@pytest.mark.parametrize(
    "filename", ["xx_hi_000209_0.png", "fm_zz_000209_0.png", "fm_random.png"]
)
def test_unknown_encoded_filename_is_not_guessed(filename):
    assert indicdlp_filename_metadata(filename) == {}


def test_exact_server_metadata_example():
    import json

    names = json.loads(
        (
            Path(__file__).resolve().parents[1] / "src/indicdlp/resources/classes.json"
        ).read_text()
    )
    adapter = Adapter(names, category_id_base=0)
    row = {
        "image": {"path": "rp_as_000209_0.png"},
        "bboxes": [
            [
                185.64244079589844,
                192.20416259765625,
                1280.049560546875,
                1339.861083984375,
            ],
            [186.296142578125, 2083.227783203125, 567.893798828125, 55.859474182128906],
            [
                183.82766723632812,
                1702.7939453125,
                1285.0443115234375,
                270.7046813964844,
            ],
            [
                789.0162353515625,
                2157.350341796875,
                80.74272155761719,
                50.65068435668945,
            ],
            [
                186.10113525390625,
                1586.5067138671875,
                1137.51708984375,
                60.617515563964844,
            ],
            [
                165.281005859375,
                2057.540283203125,
                725.4952392578125,
                170.67591857910156,
            ],
        ],
        "category_ids": [21, 24, 21, 20, 26, 11],
    }
    assert adapter.metadata(row, {})["document_type"] == "research-papers"
    # Synthetic dimensions cover all supplied coordinates. Actual export reads dimensions from the image.
    lines, dropped = adapter.annotations(row, 1600, 2400, {})
    assert [int(line.split()[0]) for line in lines] == row["category_ids"]
    assert dropped == 0
    assert len(lines) == 6
    expected = adapter.convert_box(row["bboxes"][0], 1600, 2400)
    assert list(map(float, lines[0].split()[1:])) == pytest.approx(expected, abs=1e-8)


def test_native_top_level_classlabel_path_is_preserved():
    adapter = Adapter(["paragraph", "table"])
    row = {"bboxes": [[0, 0, 20, 30]], "category_ids": [0]}
    lines, _ = adapter.annotations(
        row, 20, 30, {"category_ids": ["table", "paragraph"]}
    )
    assert lines == ["1 0.50000000 0.50000000 1.00000000 1.00000000"]


def test_native_unnamed_numeric_ids_still_require_explicit_base():
    with pytest.raises(ValueError, match="category-id-base"):
        Adapter(["paragraph"]).annotations(
            {"bboxes": [[0, 0, 20, 30]], "category_ids": [0]}, 20, 30, {}
        )


def test_native_arrays_must_have_matching_lengths():
    with pytest.raises(ValueError, match="different lengths"):
        Adapter(["paragraph"], category_id_base=0).annotations(
            {"bboxes": [[0, 0, 20, 30]], "category_ids": []}, 20, 30, {}
        )
