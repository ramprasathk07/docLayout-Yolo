import pytest

from indicdlp.data.folds import load_plan, make_folds
from indicdlp.data.prepare import read_manifest


from tests.helpers import make_prepared as prepared


def test_grouped_folds_are_disjoint_reproducible_and_keep_test_out(tmp_path):
    root = tmp_path / "data"
    records = prepared(root)
    plan = make_folds(root, tmp_path / "folds", 3, 42)
    again = make_folds(root, tmp_path / "folds2", 3, 42)
    assert plan == again
    tested = []
    for fold in plan["folds"]:
        train, val = set(fold["train_ids"]), set(fold["val_ids"])
        assert not train & val
        assert not (train | val) & set(plan["test_ids"])
        assert not {records[s]["document_id"] for s in train} & {
            records[s]["document_id"] for s in val
        }
        tested.extend(val)
        assert fold["val_stats"]["regions"]["0"] > 0
    assert len(tested) == len(set(tested)) == 24
    assert load_plan(root, tmp_path / "folds") == plan
    yaml = (tmp_path / "folds/fold_01/data.yaml").read_text()
    assert "test:" not in yaml
    assert len((tmp_path / "folds/fold_01/val.txt").read_text().splitlines()) > 0


def test_test_overlapping_source_documents_are_purged_from_development(tmp_path):
    root = tmp_path / "data"
    prepared(root, overlap=True)
    plan = make_folds(root, tmp_path / "folds", 3)
    assert set(plan["excluded_test_overlapping_ids"]) == {"doc00_page0", "doc00_page1"}
    for fold in plan["folds"]:
        assert not set(plan["excluded_test_overlapping_ids"]) & set(
            fold["train_ids"] + fold["val_ids"]
        )


def test_missing_document_ids_require_explicit_fallback(tmp_path):
    root = tmp_path / "data"
    prepared(root, with_groups=False)
    with pytest.raises(ValueError, match="Missing source document"):
        make_folds(root, tmp_path / "folds", 3)
    plan = make_folds(root, tmp_path / "folds", 3, allow_page_level=True)
    assert len(plan["page_fallback_ids"]) == 28


def test_filename_regex_and_csv_grouping(tmp_path):
    root = tmp_path / "data"
    records = prepared(root, with_groups=False)
    regex_plan = make_folds(
        root, tmp_path / "regex", 3, group_regex=r"^(doc\d+)_page_\d+\.png$"
    )
    csv = tmp_path / "groups.csv"
    csv.write_text(
        "id,document_id\n" + "".join(f"{sid},{sid.split('_')[0]}\n" for sid in records)
    )
    csv_plan = make_folds(root, tmp_path / "csv", 3, groups_csv=csv)
    assert regex_plan["groups"] == csv_plan["groups"]
    assert regex_plan["folds"] == csv_plan["folds"]


def test_changed_plan_is_rejected(tmp_path):
    root = tmp_path / "data"
    prepared(root)
    make_folds(root, tmp_path / "folds", 3)
    path = tmp_path / "folds/plan.json"
    path.write_text(path.read_text().replace('"seed": 42', '"seed": 43'))
    with pytest.raises(ValueError, match="changed after creation"):
        load_plan(root, path.parent)


def test_fold_lists_regenerate_after_dataset_move(tmp_path):
    root = tmp_path / "data"
    prepared(root)
    make_folds(root, tmp_path / "folds", 3)
    moved = tmp_path / "server_data"
    root.rename(moved)
    load_plan(moved, tmp_path / "folds")
    assert (
        str(moved.resolve().as_posix())
        in (tmp_path / "folds/fold_01/train.txt").read_text()
    )
    assert len(read_manifest(moved)) == 28


def test_fold_directory_and_group_counts_are_validated(tmp_path):
    root = tmp_path / "data"
    prepared(root)
    with pytest.raises(ValueError, match="outside"):
        make_folds(root, root / "folds", 3)
    with pytest.raises(ValueError, match="Not enough"):
        make_folds(root, tmp_path / "folds", 20)
