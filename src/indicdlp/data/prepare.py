"""Stream selected domains into a restartable, audited YOLO subset."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from .schema import domain, language


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    temporary.replace(path)


def json_write(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(root):
    path = root / "manifest.jsonl"
    if not path.exists():
        return {}
    # Recover only a torn last append. Any corruption in completed records is fatal.
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result, valid = {}, bytearray()
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise
            break
        if record["id"] in result:
            raise ValueError("Duplicate source ID in manifest")
        result[record["id"]] = record
        valid.extend(line.rstrip(b"\r\n") + b"\n")
    if bytes(valid) != data:
        atomic(path, bytes(valid))
    return result


def verify_record(root, record):
    for name in ("image", "label"):
        path = (root / record[name]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes the dataset directory")
        if not path.exists() or digest(path.read_bytes()) != record[f"{name}_sha256"]:
            raise ValueError(
                f"Missing or changed prepared file: {path}; restore it or use a new output directory"
            )


def write_sample(root, sid, split, raw_image, row, metadata, adapter, mappings):
    with Image.open(io.BytesIO(raw_image)) as original:
        width, height = original.size
        # Do not EXIF-transpose: annotations refer to the source pixel coordinates.
        image = original.convert("RGB")
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
    lines, dropped = adapter.annotations(row, width, height, mappings)
    # Reject accidental negatives created by malformed/out-of-frame annotations.
    if not lines:
        raise ValueError(f"Selected image {sid} has no usable region boxes")
    image_bytes = encoded.getvalue()
    labels = ("\n".join(lines) + "\n").encode("utf-8")
    image_path, label_path = f"images/{split}/{sid}.png", f"labels/{split}/{sid}.txt"
    atomic(root / image_path, image_bytes)
    atomic(root / label_path, labels)
    return {
        "id": sid,
        "split": split,
        **metadata,
        "image": image_path,
        "label": label_path,
        "width": width,
        "height": height,
        "boxes": len(lines),
        "dropped_boxes": dropped,
        "image_sha256": digest(image_bytes),
        "label_sha256": digest(labels),
    }


def finalize(root, records, names, report):
    counts = Counter(r["split"] for r in records.values())
    if not counts["train"] or not counts["val"]:
        raise ValueError(
            "Need at least one selected train AND val image; increase scan/sample limits or check filters"
        )
    config = {
        "path": root.resolve().as_posix(),
        "names": dict(enumerate(names)),
        "nc": len(names),
    }
    for split in ("train", "val", "test"):
        paths = sorted(
            "./" + r["image"] for r in records.values() if r["split"] == split
        )
        if paths:
            atomic(root / f"{split}.txt", "\n".join(paths) + "\n")
            config[split] = f"{split}.txt"
    atomic(
        root / "data.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    report.update(
        {
            "selected": dict(counts),
            "domains": dict(Counter(r["document_type"] for r in records.values())),
            "languages": dict(
                Counter(r["language"] or "unknown" for r in records.values())
            ),
            "dropped_boxes": sum(r["dropped_boxes"] for r in records.values()),
        }
    )
    json_write(root / "report.json", report)
    json_write(
        root / "complete.json",
        {
            "manifest_sha256": digest((root / "manifest.jsonl").read_bytes()),
            "selection_sha256": digest((root / "selection.json").read_bytes()),
            "selected": dict(counts),
        },
    )


def prepare(source, root, adapter, types, languages, limits, max_shards=0):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    types = sorted({domain(t) for t in types})
    languages = sorted({language(t) for t in languages})
    settings = {
        "version": 1,
        "repo": source.repo,
        "revision": source.revision,
        "adapter": vars(adapter),
        "types": types,
        "languages": languages,
    }
    settings_path = root / "selection.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError(
                "Output selection/schema/revision differs; use a new --output directory"
            )
    elif any(root.iterdir()):
        raise ValueError(
            "Output directory is nonempty without selection.json; use a fresh directory"
        )
    else:
        json_write(settings_path, settings)
    # A failed extension must never leave a previous completion marker behind.
    (root / "complete.json").unlink(missing_ok=True)
    records = read_manifest(root)
    for record in records.values():
        verify_record(root, record)
    counts = Counter(r["split"] for r in records.values())
    if any(limits[s] and counts[s] > limits[s] for s in limits):
        raise ValueError(
            "Existing subset exceeds requested limits; use a new output directory"
        )
    seen_hashes = {r["image_sha256"]: r["split"] for r in records.values()}
    report = {
        "revision": source.revision,
        "scanned_rows": 0,
        "scanned_shards": 0,
        "transfer_bytes_this_run": 0,
        "duplicate_pages_skipped": 0,
        "limited": False,
    }
    try:
        with (root / "manifest.jsonl").open("ab") as manifest:
            for split in ("train", "val", "test"):
                if split not in limits:
                    continue
                shards = [s for s in source.shards if s.split == split]
                if not shards:
                    raise ValueError(f"Requested split {split!r} has no source shards")
                visited = 0
                for shard in shards:
                    if limits[split] and counts[split] >= limits[split]:
                        report["limited"] = True
                        break
                    if max_shards and visited >= max_shards:
                        report["limited"] = True
                        break
                    visited += 1
                    report["scanned_shards"] += 1
                    print(
                        f"[{split}] {shard.path} | saved={counts[split]} | "
                        f"transfer={source.budget.used / 2**20:.1f} MiB",
                        flush=True,
                    )
                    with source.open(shard) as handle:
                        for group in handle.groups():
                            if limits[split] and counts[split] >= limits[split]:
                                break
                            rows = handle.metadata_rows(group)
                            selected = []
                            for index, row in enumerate(rows):
                                report["scanned_rows"] += 1
                                sid = digest(
                                    f"{source.repo}@{source.revision}/{shard.path}:{group}:{index}".encode()
                                )[:24]
                                if sid in records:
                                    continue
                                metadata = adapter.metadata(row, handle.mappings)
                                if metadata["document_type"] not in types:
                                    continue
                                if languages and metadata["language"] is None:
                                    raise ValueError(
                                        "Language filter requested but language is missing; set --language-field"
                                    )
                                if languages and metadata["language"] not in languages:
                                    continue
                                selected.append((index, sid, metadata))
                            if not selected:
                                continue
                            table = handle.images(group)
                            for index, sid, metadata in selected:
                                if limits[split] and counts[split] >= limits[split]:
                                    report["limited"] = True
                                    break
                                raw = handle.image_at(table, index)
                                record = write_sample(
                                    root,
                                    sid,
                                    split,
                                    raw,
                                    rows[index],
                                    metadata,
                                    adapter,
                                    handle.mappings,
                                )
                                sha = record["image_sha256"]
                                if sha in seen_hashes:
                                    # Exact duplicate pixels do not leak across official splits.
                                    (root / record["image"]).unlink()
                                    (root / record["label"]).unlink()
                                    report["duplicate_pages_skipped"] += 1
                                    continue
                                record.update(
                                    {
                                        "shard": shard.path,
                                        "row_group": group,
                                        "row": index,
                                    }
                                )
                                manifest.write(
                                    (
                                        json.dumps(record, ensure_ascii=False) + "\n"
                                    ).encode("utf-8")
                                )
                                manifest.flush()
                                os.fsync(manifest.fileno())
                                records[sid] = record
                                seen_hashes[sha] = split
                                counts[split] += 1
                            del table
        if any(limit and counts[split] >= limit for split, limit in limits.items()):
            report["limited"] = True
        report["transfer_bytes_this_run"] = source.budget.used
        finalize(root, records, adapter.names, report)
    except Exception as exc:
        report.update(
            {
                "transfer_bytes_this_run": source.budget.used,
                "error": str(exc),
                "selected": dict(counts),
            }
        )
        json_write(root / "incomplete.json", report)
        raise
    (root / "incomplete.json").unlink(missing_ok=True)
    return report


def verify_dataset(root):
    root = Path(root).resolve()
    marker = root / "complete.json"
    if not marker.exists():
        raise ValueError(
            "Preparation incomplete: complete.json missing. Rerun prepare before training."
        )
    complete = json.loads(marker.read_text())
    if complete["manifest_sha256"] != digest((root / "manifest.jsonl").read_bytes()):
        raise ValueError("Manifest changed after completion")
    if complete["selection_sha256"] != digest((root / "selection.json").read_bytes()):
        raise ValueError("Selection/schema changed after completion")
    records = read_manifest(root)
    for record in records.values():
        verify_record(root, record)
    names = json.loads((root / "selection.json").read_text(encoding="utf-8"))[
        "adapter"
    ]["names"]
    # Regenerate absolute root and lists if the prepared subset was moved to a server.
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    finalize(root, records, names, report)
    return root / "data.yaml"
