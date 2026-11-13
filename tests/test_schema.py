import pytest

from indicdlp.data.schema import Adapter, classlabels


@pytest.fixture
def adapter():
    return Adapter(["paragraph", "table", "placeholder-text"])


@pytest.mark.parametrize("value", ["Forms", "form", "FORMS"])
def test_domain_filter_normalization(adapter, value):
    assert adapter.metadata({"document_type": value, "language": "hi"}, {}) == {
        "document_type": "forms",
        "language": "hindi",
        "filename": None,
        "document_id": None,
    }


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Hindi_Forms_doc_12.png", "forms"),
        ("en/notices/page.png", "notices"),
        ("Hindi_Acts_and_Rules_12.png", "acts-and-rules"),
        ("Tamil_Novels_12.png", "novels"),
        ("English_Research_Papers_12.png", "research-papers"),
    ],
)
def test_filename_fallback(adapter, filename, expected):
    assert adapter.metadata({"file_name": filename}, {})["document_type"] == expected


def test_no_substring_match_or_numeric_guess(adapter):
    with pytest.raises(ValueError, match="Cannot identify"):
        adapter.metadata({"filename": "transforms_123.png"}, {})
    with pytest.raises(ValueError, match="Cannot identify"):
        adapter.metadata({"doc_type": 3}, {})


def test_explicit_type_must_exist(adapter):
    adapter.type_field = "meta.kind"
    with pytest.raises(ValueError, match="Explicit"):
        adapter.metadata({"filename": "forms.png"}, {})


def test_source_document_id_is_preserved_for_grouped_validation(adapter):
    adapter.document_id_field = "metadata.document"
    row = {"document_type": "Forms", "metadata": {"document": "source-12"}}
    assert adapter.metadata(row, {})["document_id"] == "source-12"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Acts & Rules", "acts-and-rules"),
        ("Brochures", "brochures"),
        ("Forms", "forms"),
        ("Newspapers", "newspapers"),
        ("Manuals", "manuals"),
        ("Notices", "notices"),
        ("Novels", "novels"),
        ("Syllabi", "syllabi"),
    ],
)
def test_requested_eight_document_types(adapter, value, expected):
    assert adapter.metadata({"document_type": value}, {})["document_type"] == expected


def test_classlabel_metadata_and_annotation_mappings(adapter):
    features = {
        "doc_type": {"_type": "ClassLabel", "names": ["Forms"]},
        "objects": {
            "_type": "Sequence",
            "feature": {
                "category_id": {"_type": "ClassLabel", "names": ["table", "paragraph"]}
            },
        },
    }
    mappings = classlabels(features)
    assert adapter.metadata({"doc_type": 0}, mappings)["document_type"] == "forms"
    lines, _ = adapter.annotations(
        {"objects": {"bbox": [[10, 20, 20, 40]], "category_id": [0]}},
        100,
        200,
        mappings,
    )
    assert lines == ["1 0.20000000 0.20000000 0.20000000 0.20000000"]


def test_numeric_id_base_never_guessed(adapter):
    row = {"annotations": [{"category_id": 1, "bbox": [10, 20, 20, 40]}]}
    with pytest.raises(ValueError, match="category-id-base"):
        adapter.annotations(row, 100, 200, {})
    adapter.category_id_base = 1
    assert adapter.annotations(row, 100, 200, {})[0][0].startswith("0 ")


def test_coco_category_names_remap(adapter):
    row = {
        "annotations": {
            "categories": [{"id": 42, "name": "table"}],
            "annotations": [{"category_id": 42, "bbox": [0, 0, 100, 200]}],
        }
    }
    assert adapter.annotations(row, 100, 200, {})[0] == [
        "1 0.50000000 0.50000000 1.00000000 1.00000000"
    ]


def test_boxes_clip_and_drop_invalid_area(adapter):
    row = {
        "annotations": [
            {"label": "table", "bbox": [-10, -20, 50, 60]},
            {"label": "paragraph", "bbox": [1000, 20, 1, 2]},
        ]
    }
    lines, dropped = adapter.annotations(row, 100, 200, {})
    assert lines == ["1 0.20000000 0.10000000 0.40000000 0.20000000"]
    assert dropped == 1


@pytest.mark.parametrize("value", ["0 0.5 0.5 1 1\n", ["0 0.5 0.5 1 1"]])
def test_native_yolo_text_is_already_zero_based(adapter, value):
    adapter.category_id_base = 1
    assert adapter.annotations({"labels": value}, 100, 200, {})[0][0].startswith("0 ")


@pytest.mark.parametrize("box", [[float("nan"), 0, 1, 1], [1, 2, 3], [1, 1, 0, 1]])
def test_bad_yolo_boxes(adapter, box):
    adapter.bbox_format = "yolo"
    with pytest.raises(ValueError):
        adapter.convert_box(box, 100, 200)


def test_xyxy_conversion(adapter):
    adapter.bbox_format = "xyxy"
    assert adapter.convert_box([10, 20, 30, 60], 100, 200) == [0.2, 0.2, 0.2, 0.2]


def test_mismatched_object_arrays(adapter):
    with pytest.raises(ValueError, match="different lengths"):
        adapter.annotations(
            {"objects": {"bbox": [[0, 0, 1, 1]], "category_id": []}}, 10, 10, {}
        )


def test_unknown_layout_class_and_crowd_fail(adapter):
    with pytest.raises(ValueError, match="Unknown layout class"):
        adapter.annotations(
            {"annotations": [{"label": "invented", "bbox": [1, 1, 2, 2]}]}, 10, 10, {}
        )
    with pytest.raises(ValueError, match="Crowd"):
        adapter.annotations(
            {"annotations": [{"label": "table", "bbox": [1, 1, 2, 2], "iscrowd": 1}]},
            10,
            10,
            {},
        )
