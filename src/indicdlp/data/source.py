"""Remote Parquet access. No snapshot_download, file cache, or dataset materialization."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass

from .ranges import RangeReader, http_session
from .schema import classlabels


@dataclass(frozen=True)
class Shard:
    path: str
    size: int
    split: str


def split_for(path):
    match = re.search(r"(?:^|/)(train|validation|val|test)(?:[-/.]|$)", path)
    if not match:
        return None
    return "val" if match[1] == "validation" else match[1]


def arrow_imports():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Install data dependencies: python -m pip install -r requirements-data.txt"
        ) from exc
    return pa, pq


def metadata_columns(schema, image_field):
    """Exclude binary data recursively; project only image.path from image structs."""
    pa, _ = arrow_imports()

    def has_binary(dtype):
        if (
            pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_fixed_size_binary(dtype)
        ):
            return True
        if pa.types.is_struct(dtype):
            return any(has_binary(f.type) for f in dtype)
        if (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
        ):
            return has_binary(dtype.value_type)
        if pa.types.is_map(dtype):
            return has_binary(dtype.item_type) or has_binary(dtype.key_type)
        return False

    columns = []
    for f in schema:
        if f.name == image_field:
            if pa.types.is_struct(f.type) and "path" in [
                child.name for child in f.type
            ]:
                columns.append(f"{image_field}.path")
            continue
        if not has_binary(f.type):
            columns.append(f.name)
    if not columns:
        raise ValueError(
            "No metadata columns available; cannot filter before fetching images"
        )
    return columns


class ParquetShard:
    def __init__(self, parquet, image_field, max_group_bytes):
        self.parquet = parquet
        self.image_field = image_field
        self.max_group_bytes = max_group_bytes
        self.columns = metadata_columns(parquet.schema_arrow, image_field)
        metadata = parquet.schema_arrow.metadata or {}
        hf = json.loads(metadata.get(b"huggingface", b"{}"))
        self.mappings = classlabels(hf.get("info", {}).get("features", {}))

    def check_size(self, group, columns):
        rg = self.parquet.metadata.row_group(group)
        size = sum(
            rg.column(i).total_uncompressed_size
            for i in range(rg.num_columns)
            if any(
                rg.column(i).path_in_schema == c
                or rg.column(i).path_in_schema.startswith(c + ".")
                for c in columns
            )
        )
        if size > self.max_group_bytes:
            raise RuntimeError(
                f"Row group requests {size / 2**20:.1f} MiB uncompressed; exceeds "
                "--max-group-mib. Increase it only on a server with sufficient RAM."
            )

    def groups(self):
        return range(self.parquet.num_row_groups)

    def metadata_rows(self, group):
        self.check_size(group, self.columns)
        return self.parquet.read_row_group(
            group, columns=self.columns, use_threads=False
        ).to_pylist()

    def images(self, group):
        self.check_size(group, [self.image_field])
        return self.parquet.read_row_group(
            group, columns=[self.image_field], use_threads=False
        )

    def image_at(self, table, index):
        value = table.slice(index, 1).to_pylist()[0].get(self.image_field)
        if isinstance(value, dict):
            value = value.get("bytes")
        if not isinstance(value, bytes):
            raise ValueError(
                f"Expected embedded image bytes in {self.image_field!r}. "
                "Path-only images are not fetched automatically. Inspect the schema."
            )
        return value


class HubSource:
    def __init__(self, repo, revision, budget, image_field="image", max_group_mib=256):
        from huggingface_hub import HfApi, get_token

        self.repo, self.budget = repo, budget
        self.image_field, self.max_group_bytes = image_field, int(max_group_mib * 2**20)
        token = get_token()
        api = HfApi(token=token)
        try:
            info = api.dataset_info(repo, revision=revision)
            self.revision = info.sha
            files = api.list_repo_tree(
                repo, repo_type="dataset", revision=self.revision, recursive=True
            )
            self.shards = sorted(
                [
                    Shard(f.path, f.size, split_for(f.path))
                    for f in files
                    if f.path.endswith(".parquet") and split_for(f.path)
                ],
                key=lambda f: f.path,
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot list HF dataset. Accept the ai4bharat/indicdlp access conditions, "
                "set HF_TOKEN (or run hf auth login), and check repository/revision/network."
            ) from exc
        if not self.shards:
            raise ValueError("No train/val/test Parquet shards found")
        self.session = http_session(token)

    @contextmanager
    def open(self, shard):
        from huggingface_hub import hf_hub_url

        _, pq = arrow_imports()
        url = hf_hub_url(
            self.repo, shard.path, repo_type="dataset", revision=self.revision
        )
        with RangeReader(url, shard.size, self.session, self.budget) as reader:
            # Disable Arrow prefetch/read-ahead so column projection controls requests.
            parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=0)
            yield ParquetShard(parquet, self.image_field, self.max_group_bytes)

    def close(self):
        self.session.close()
