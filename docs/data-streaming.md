# IndicDLP streaming and annotation adapters

Use the [A6000 server guide](server-a6000.md) for the full training/testing workflow.

## Native HF schema confirmed from server inspection

At revision `7f23f6d8569dd6336534dd9700259b9843ecab5e`, the authenticated
inspection reported `image: {bytes, path}`, `bboxes: list[list[float]]`, and
`category_ids: list[int32]`. There is no separate document-type column or HF
ClassLabel mapping in that schema. The adapter now reads the parallel box/ID
arrays directly. The selected-domain preset explicitly uses the official
zero-based YOLO class order and COCO pixel `xywh` boxes.

Document domains come from filename prefixes documented by the
[official labeled examples](https://github.com/AI4Bharat/IndicDLP#-samples-from-the-indicdlp-dataset):

| Prefix | Document domain |
| --- | --- |
| `ar` | Acts & Rules |
| `br` | Brochures |
| `fm` | Forms |
| `mg` | Magazines |
| `mn` | Manuals |
| `np` | Newspapers |
| `nt` | Notices |
| `nv` | Novels |
| `qp` | Question Papers |
| `rp` | Research Papers |
| `sy` | Syllabi |
| `tb` | Textbooks |

For example, `rp_as_000209_0.png` is Research Papers / Assamese and
`fm_bn_000220_0.png` is Forms / Bengali. The adapter deliberately does not
assume the numeric filename components identify source documents/pages;
grouped CV still requires verified source-document grouping.

After updating code from the earlier schema adapter, use a fresh output path
because failed preparation attempts recorded the old adapter settings:

```bash
python scripts/download_selected_indicdlp.py --output data/indicdlp_selected_v2
```

No `--type-field` or `--annotations-field` override is needed for this native
schema. The preset pins the inspected revision and sets `category_id_base: 0`
and `bbox_format: xywh` in `configs/data/selected.yaml`.

## Inspect first, with no image-column reads

```bash
python train_indicdlp.py inspect --rows 3 --max-transfer-mib 16
```

This prints the immutable HF revision, Arrow schema, named ClassLabel mappings, a small metadata sample, and detected document type/language. Only the first shard's first row group's **nonbinary metadata columns** are read; embedded `image.bytes` is excluded. Parquet footer requests can incidentally overlap other bytes near the file's end. No document images are decoded or saved by `inspect`.

**The gated dataset rows could not be accessed in this local session.** Column names, document-type metadata, and category-ID conventions therefore need confirmation through this bounded command on your authenticated server. The adapter supports common HF/COCO/YOLO structures and fails with an explicit error on ambiguous formats. It never silently trains the whole dataset when it cannot find the domain.

If automatic field detection does not match the schema, supply exact fields on both `inspect` and `prepare`, for example:

```bash
python train_indicdlp.py inspect \
  --type-field metadata.document_type \
  --language-field metadata.language \
  --filename-field file_name \
  --annotations-field annotations \
  --image-field image
```

These field names are **examples**, not a claim about inaccessible gated rows. Dotted paths can traverse JSON metadata. Without a document-type field, the adapter can infer domains from complete filename tokens such as `Hindi_Forms_...`; it rejects opaque names. Numeric document domains need named HF ClassLabel metadata. The embedded image column must be top-level; path-only image references are rejected rather than fetched from unverified locations.

Annotation handling:

- COCO records containing pixel `[x, y, width, height]` boxes; JSON strings and lists/dictionaries of objects are supported.
- Named region labels or named HF ClassLabels are remapped to `src/indicdlp/resources/classes.json`.
- Unnamed numeric annotation IDs require **explicit** `--category-id-base 0` or `--category-id-base 1`, after you confirm their convention and ordering. Numeric IDs must correspond to the official class order; otherwise provide your verified ordered class-name array with `--classes`.
- Native YOLO text labels already use zero-based IDs and normalized center/width/height. They do not need `--category-id-base`.
- For other numeric bbox representations set `--bbox-format xyxy` (pixel corners) or `--bbox-format yolo` (normalized center/size). Default is COCO `xywh`; normalization is never guessed from coordinate magnitudes.

## Prepare a small server smoke subset

After inspection, this selects at most 12 training pages, four validation pages, and four test pages:

```bash
python train_indicdlp.py prepare \
  --output data/indicdlp_smoke \
  --types forms \
  --max-train 12 --max-val 4 --max-test 4 \
  --max-transfer-mib 2048 --max-group-mib 1024
```

Add the verified schema overrides and `--category-id-base` if required by your source. Few selected pages can still require a large row-group read; the transfer limit may stop this command before all sample caps are reached. It reports an error, leaves a resumable manifest, and does not mark the dataset ready. Increase the budget on the server and repeat the same command to resume.

```bash
python train_indicdlp.py train --data data/indicdlp_smoke --dry-run
python train_indicdlp.py train --data data/indicdlp_smoke \
  --epochs 1 --imgsz 640 --batch 2 --workers 2 --device 0 --name smoke
```

## Prepare all matching Forms pages on the server

```bash
python train_indicdlp.py prepare \
  --output data/indicdlp_forms \
  --types forms \
  --max-train 0 --max-val 0 --max-test 0 \
  --max-transfer-mib 32768 --max-group-mib 1024
```

Here `0` disables the **selected-page** caps; it does not disable the domain filter or transfer budget. The 32 GiB budget is a ceiling, not an estimate of subset size. Default `prepare` limits, when omitted, are 2,000 train / 250 val / 250 test pages and 2 GiB of Parquet payload per invocation. Capped selection follows source order; it is not a random or stratified sample.

To include broader FIR-adjacent domains, use a **new** directory:

```bash
python train_indicdlp.py prepare \
  --output data/indicdlp_administrative \
  --types forms notices acts-and-rules \
  --max-train 0 --max-val 0 --max-test 0 \
  --max-transfer-mib 32768 --max-group-mib 1024
```

Optionally add `--languages hi en mr` or full language names. All languages are retained by default. Unknown language metadata fails when a language filter is requested. `--max-shards N` limits scanning per split for exploratory runs; it can miss matching pages in later shards. `--revision COMMIT_SHA` pins the revision printed by `inspect`; preparation always records and reads from an immutable SHA internally.

## Streaming, storage, and recovery details

1. The Hub API lists shard paths/sizes and resolves one immutable revision. It does not download Parquet files.
2. A seekable HTTP reader issues exact bounded `Range` requests. HTTP 200 responses are rejected **before** reading their bodies, preventing an accidental whole-shard fallback. Response sizes and `Content-Range` must match.
3. PyArrow reads metadata columns first, with read-ahead disabled. Image columns are requested only for row groups with selected pages, then only selected pages are decoded/written.
4. Parquet compression operates by column chunk/row group. **A mixed row group can transfer image bytes for unrelated pages too.** This is not an HF server-side row-selection API and cannot promise that every transferred byte belongs to a selected form. Row groups with no matches avoid image-column reads entirely.
5. Only selected PNGs, YOLO labels, and audit files remain on disk. Upstream training needs random access, so selected pages are staged once on the server instead of re-streamed on every epoch. No raw shard cache or full `load_dataset`/`snapshot_download` is used.

`--max-transfer-mib` counts actual Parquet body bytes returned to the reader in that invocation; API listing, HTTP headers, transport overhead, and separate model-weight downloads are outside that count. `--max-group-mib` limits requested uncompressed column chunks, not total process RAM; Arrow/Python/Pillow add overhead. Repeated preparation invocations each get a new transfer budget.

Outputs:

```text
data/indicdlp_forms/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt
  data.yaml                  # official class names + selected split lists
  train.txt, val.txt, test.txt
  selection.json             # filters, schema adapter, immutable revision
  manifest.jsonl             # source row, domain/language, image + label hashes
  report.json                # counts, limits, transfer bytes, dropped boxes
  complete.json              # written only after successful preparation
```

Official train/val/test assignments are preserved. Exact decoded RGB image duplicates are skipped across all splits to prevent exact-page leakage. Existing near-duplicates or different pages from the same source document cannot be detected without reliable document IDs. Out-of-frame boxes are clipped; zero-area boxes are dropped and counted. Unknown class IDs, malformed annotations, and crowd/ignore boxes still fail explicitly.

Selected pages with empty annotations are skipped rather than treated as verified
background images. Pages whose boxes are all outside the image or degenerate are
also excluded, with a distinct audit reason. `skipped.jsonl` records each source
row ID, filename, split, shard, row group, row index, reason, and image dimensions
when available. Native empty box/ID arrays are detected from metadata before
requesting image columns. Reports include cumulative skipped-page counts,
reason/split breakdowns, and dropped-box counts for skipped pages separately from
successfully exported pages. Review `all_boxes_outside_or_degenerate` entries to
distinguish isolated source issues from a coordinate-format/dimension mismatch.

Resume uses both `manifest.jsonl` and `skipped.jsonl`, preserving completed pages
and avoiding repeated image fetches for already audited exclusions. A torn final
audit append can be recovered. Completed exports checksum the skip ledger too.
This update can resume existing `indicdlp_selected_v2` exports; a fresh output
directory is not required. No ready marker is written unless training and
validation both contain at least one exported page with usable targets.

Repeat `prepare` with identical filters/schema/revision to resume. Existing files are checked by SHA-256 and do not re-fetch their images. Larger sample limits are allowed; changed filters or revisions require a new directory. A torn final manifest append is recovered. Do not run concurrent preparers against the same output directory. Incomplete preparation is blocked from training.

After copying a prepared subset to another machine, run:

```bash
python train_indicdlp.py verify --data /server/path/indicdlp_forms
```

This checks hashes and regenerates paths. The training/evaluation commands also do this verification automatically.

