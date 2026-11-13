"""Reproducible, stratified document-group CV with an untouched official test set."""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import yaml

from .prepare import atomic, digest, json_write, read_manifest, verify_dataset


def fingerprint(root):
    return {
        name: digest((root / name).read_bytes())
        for name in ("manifest.jsonl", "selection.json")
    }


def grouping(records, groups_csv=None, group_regex=None, allow_page_level=False):
    mapping, key = {}, None
    if groups_csv:
        with Path(groups_csv).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            key = "id" if "id" in columns else "filename"
            if not {key, "document_id"}.issubset(columns):
                raise ValueError(
                    "Groups CSV needs id,document_id or filename,document_id columns"
                )
            for row in reader:
                if not row[key] or not row["document_id"] or row[key] in mapping:
                    raise ValueError("Groups CSV has missing values or duplicate keys")
                mapping[row[key]] = row["document_id"]
    pattern = re.compile(group_regex) if group_regex else None
    if pattern and not pattern.groups:
        raise ValueError("--group-regex must capture the source document identifier")
    groups, fallback = {}, []
    for sid, record in sorted(records.items()):
        if groups_csv:
            group = mapping.get(record.get(key))
        elif pattern:
            match = pattern.search(record.get("filename") or "")
            group = (
                (match.groupdict().get("document_id") or match.group(1))
                if match
                else None
            )
        else:
            group = record.get("document_id")
        if group is None or str(group).strip() == "":
            if not allow_page_level:
                raise ValueError(
                    f"Missing source document group for {sid} ({record.get('filename')}). "
                    "Preserve --document-id-field during prepare, provide --groups-csv, "
                    "or supply a verified --group-regex. --allow-page-level explicitly permits "
                    "page-level fallback but cannot prevent source-document leakage."
                )
            group = f"page:{sid}"
            fallback.append(sid)
        groups[sid] = str(group)
    return groups, fallback


def subset_stats(ids, records, groups, label_counts):
    regions = Counter()
    for sid in ids:
        regions.update(label_counts[sid])
    return {
        "pages": len(ids),
        "documents": len({groups[s] for s in ids}),
        "languages": dict(
            Counter(records[s].get("language") or "unknown" for s in ids)
        ),
        "domains": dict(Counter(records[s]["document_type"] for s in ids)),
        "regions": dict(sorted(regions.items())),
    }


def write_fold_files(root, output, plan):
    records = read_manifest(root)
    names = json.loads((root / "selection.json").read_text(encoding="utf-8"))[
        "adapter"
    ]["names"]
    for fold in plan["folds"]:
        folder = output / f"fold_{fold['index']:02d}"
        config = {
            "path": root.as_posix(),
            "names": dict(enumerate(names)),
            "nc": len(names),
        }
        for split in ("train", "val"):
            listing = folder / f"{split}.txt"
            paths = [
                (root / records[sid]["image"]).as_posix()
                for sid in fold[f"{split}_ids"]
            ]
            atomic(listing, "\n".join(paths) + "\n")
            config[split] = listing.as_posix()
        # No test key: CV evaluates validation only. Test is evaluated after final training.
        atomic(
            folder / "data.yaml",
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        )


def make_folds(
    root,
    output,
    n_splits=5,
    seed=42,
    groups_csv=None,
    group_regex=None,
    allow_page_level=False,
):
    from sklearn.model_selection import StratifiedGroupKFold

    root, output = Path(root).resolve(), Path(output).resolve()
    verify_dataset(root)
    if n_splits < 2:
        raise ValueError("Cross-validation needs at least two folds")
    if output == root or output.is_relative_to(root):
        raise ValueError("Keep fold plans outside the prepared dataset directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Fold output already exists; reuse it with crossval or choose a new directory"
        )
    records = read_manifest(root)
    groups, fallback = grouping(records, groups_csv, group_regex, allow_page_level)
    test_ids = sorted(s for s, r in records.items() if r["split"] == "test")
    test_groups = {groups[s] for s in test_ids}
    pool = sorted(
        s
        for s, r in records.items()
        if r["split"] in ("train", "val") and groups[s] not in test_groups
    )
    excluded = sorted(set(records) - set(pool) - set(test_ids))
    if len({groups[s] for s in pool}) < n_splits:
        raise ValueError(
            "Not enough independent source documents after excluding test-overlapping groups"
        )
    original_strata = {
        s: f"{records[s].get('language') or 'unknown'}|{records[s]['document_type']}"
        for s in pool
    }
    group_counts = {
        label: len({groups[s] for s in pool if original_strata[s] == label})
        for label in set(original_strata.values())
    }
    sparse = sorted(label for label, count in group_counts.items() if count < n_splits)
    strata = [
        "__rare__" if original_strata[s] in sparse else original_strata[s] for s in pool
    ]
    label_counts = {
        s: Counter(
            line.split()[0]
            for line in (root / r["label"]).read_text().splitlines()
            if line.strip()
        )
        for s, r in records.items()
    }
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always", UserWarning)
        for i, (train_idx, val_idx) in enumerate(
            splitter.split(pool, strata, [groups[s] for s in pool]), 1
        ):
            train_ids, val_ids = (
                [pool[j] for j in train_idx],
                [pool[j] for j in val_idx],
            )
            if not train_ids or not val_ids:
                raise ValueError(
                    "A grouped fold is empty; reduce fold count or improve group balance"
                )
            folds.append(
                {
                    "index": i,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "train_stats": subset_stats(
                        train_ids, records, groups, label_counts
                    ),
                    "val_stats": subset_stats(val_ids, records, groups, label_counts),
                }
            )
    plan = {
        "version": 1,
        "fingerprint": fingerprint(root),
        "seed": seed,
        "n_splits": n_splits,
        "pool": "official train + val; official test and its source groups excluded",
        "groups": groups,
        "page_fallback_ids": fallback,
        "sparse_strata": sparse,
        "stratification": "language + domain; rare strata combined",
        "splitter_notes": [str(note.message) for note in notes],
        "test_ids": test_ids,
        "excluded_test_overlapping_ids": excluded,
        "folds": folds,
    }
    validate_plan(plan, records)
    output.mkdir(parents=True, exist_ok=True)
    write_fold_files(root, output, plan)
    json_write(output / "plan.json", plan)
    atomic(output / "plan.sha256", digest((output / "plan.json").read_bytes()) + "\n")
    print(
        f"Created {n_splits} folds: {len(pool)} development pages, {len(test_ids)} held-out test pages, "
        f"{len(excluded)} test-overlapping development pages excluded."
    )
    if fallback:
        print(
            f"Page-level fallback used for {len(fallback)} pages; document independence is not established."
        )
    return plan


def validate_plan(plan, records):
    groups = plan["groups"]
    test_ids = {s for s, r in records.items() if r["split"] == "test"}
    if set(plan["test_ids"]) != test_ids or set(groups) != set(records):
        raise ValueError("Fold plan does not match source IDs/test assignment")
    test_groups = {groups[s] for s in test_ids}
    pool = {
        s
        for s, r in records.items()
        if r["split"] in ("train", "val") and groups[s] not in test_groups
    }
    validation_ids = []
    if len(plan["folds"]) != plan["n_splits"]:
        raise ValueError("Wrong number of folds")
    for i, fold in enumerate(plan["folds"], 1):
        train, val = set(fold["train_ids"]), set(fold["val_ids"])
        if (
            fold["index"] != i
            or not train
            or not val
            or train & val
            or train | val != pool
        ):
            raise ValueError("Invalid fold membership")
        if len(train) != len(fold["train_ids"]) or len(val) != len(fold["val_ids"]):
            raise ValueError("Duplicate page IDs in fold")
        if {groups[s] for s in train} & {groups[s] for s in val}:
            raise ValueError("Source document leakage between train and validation")
        validation_ids.extend(val)
    if Counter(validation_ids) != Counter({s: 1 for s in pool}):
        raise ValueError("Every development page must be validated exactly once")


def load_plan(root, output, verify=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if verify:
        verify_dataset(root)
    data = (output / "plan.json").read_bytes()
    if digest(data) != (output / "plan.sha256").read_text().strip():
        raise ValueError("Fold plan changed after creation")
    plan = json.loads(data)
    if plan["fingerprint"] != fingerprint(root):
        raise ValueError("Prepared dataset changed; create new folds")
    validate_plan(plan, read_manifest(root))
    write_fold_files(root, output, plan)
    return plan


def fold_data(root, output, index):
    plan = load_plan(root, output, verify=False)
    if not 1 <= index <= plan["n_splits"]:
        raise ValueError("Fold index out of range (folds are numbered from 1)")
    return Path(output).resolve() / f"fold_{index:02d}" / "data.yaml"
