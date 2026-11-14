# A6000 server: data, five-fold training, and held-out testing

Run these commands from the repository root. The presets target **one RTX A6000 with 48 GB VRAM**. They are reproducible starting recipes, not measured optimums; benchmark the first epoch's VRAM/throughput on your server.

## 1. Install and authenticate

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[train,dev]'
python -m pip check

export HF_TOKEN='your_hugging_face_token'
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT='indicdlp-doclayout'
# Optional: export WANDB_ENTITY='your_team'
```

Accept the [dataset access gate](https://huggingface.co/datasets/ai4bharat/indicdlp) with the HF account associated with the token. Commands load `.env` from the current directory and repository root; exported variables take precedence. Use the exact names `HF_TOKEN` and `WANDB_API_KEY`. Online tracking requires a nonempty API key and fails with a clear error instead of prompting; offline/disabled tracking needs no key. HF/W&B keys are never placed in CLI arguments, configs, or our metric reports. `WANDB_BASE_URL` optionally selects your API host; leave it unset to keep the SDK's configured host. Keys must belong to that host. The CUDA 12.1 wheels require a compatible NVIDIA driver. The older upstream checkpoint finalizer is why training is pinned below torch 2.6.

W&B uses `wandb>=0.31,<0.32`. The old 0.19 dependency rejected newer long API
keys; [W&B added support for keys longer than 40 characters in 0.22.3](https://github.com/wandb/wandb/releases/tag/v0.22.3).
Keep the full key unchanged. Online tracking explicitly passes the environment
key to `wandb.login(prompt=False, verify=True)` before creating a run. If your
account uses Forge, set `WANDB_BASE_URL=https://api.forge.coreweave.com` in `.env`.
No key or host is inferred from key length.

After applying an update, refresh the editable package metadata and check auth:

```bash
python -m pip install --upgrade -e '.[train]'
python scripts/check_wandb.py
```

This check authenticates without creating a run or downloading data/model weights.
It never prints the key. Missing credentials, obsolete SDKs, and authentication
failures exit with an error and never fall back to a prompt.

If your existing environment reports torch 2.15.0+cu132, use the separate
`.venv-train` environment above. CUDA support does not imply checkpoint-loader
compatibility: starting in torch 2.6, `torch.load` defaults to `weights_only=True`,
whereas this fork uses full serialized model objects. This code deliberately
keeps the supported torch 2.5.1 / torchvision 0.20.1 combination. Check the actual
interpreter before launching; model/runtime errors now fail before W&B creates a run:

```bash
.venv-train/bin/python -c "import sys,torch,torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"

CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python train_indicdlp.py train \
  --data data/indicdlp_selected_v2 --config configs/training/a6000.yaml \
  --device 0 --name selected_a6000_gpu1 --wandb online
```

`CUDA_VISIBLE_DEVICES=1` selects physical GPU 1, presented to the process as device 0.
Prepared data can be reused directly; changing the environment requires no data download.

The installed command is `indicdlp`. Every command also works as `python train_indicdlp.py ...` from the repository root.

## 2. Verify the live schema, then stream only the eight requested types

```bash
indicdlp inspect --rows 3 --max-transfer-mib 16

python scripts/download_selected_indicdlp.py \
  --output data/indicdlp_selected \
  --max-transfer-mib 32768
```

The downloader reads [configs/data/selected.yaml](../configs/data/selected.yaml): Acts & Rules, Brochures, Forms, Newspapers, Manuals, Notices, Novels, and Syllabi. It retains the source train/val/test assignments, with test images/labels physically in separate `images/test` and `labels/test` directories. It scans metadata and saves only matching pages; mixed Parquet row groups can still transfer some unrelated bytes. It does not retain raw Parquet shards.

Pages with unverifiable document metadata are skipped before image loading and
recorded in `skipped.jsonl` as `unverified_metadata`, including the parser error
and source row location. Unknown filename prefixes are not guessed.
Pages without usable region targets are also excluded and recorded in `skipped.jsonl`,
with empty annotations distinguished from boxes entirely outside the image or
degenerate. They are not silently converted into background training examples.
After an interrupted download, rerun the same command against the same output
directory; completed files and audited exclusions are remembered.

The default transfer ceiling is **32 GiB per invocation**, not a prediction of subset size. Since eight domains cover a substantial portion of IndicDLP, the selected data can still be large. Repeating the same command resumes saved pages. To cap a server smoke subset instead:

```bash
python scripts/download_selected_indicdlp.py --output data/smoke \
  --max-train 40 --max-val 12 --max-test 12 --max-transfer-mib 2048
```

Authenticated server inspection has now confirmed the native `image`, `bboxes`,
and `category_ids` schema at the revision pinned by the selected-domain preset.
Document type and language are decoded from documented filename prefixes, so
no type/annotation field override is needed for that revision. The preset
explicitly selects zero-based IDs and pixel `xywh` boxes. If an older attempt
created `selection.json`, use a fresh output such as `data/indicdlp_selected_v2`
and substitute that directory throughout the commands below. Full details:
[data-streaming.md](data-streaming.md).

## 3. Make five independent document-group folds

```bash
indicdlp make-folds --data data/indicdlp_selected \
  --output folds/indicdlp_selected --n-splits 5 --seed 42
```

This pools **official train + validation only**, stratifies approximately by language/domain, and keeps pages from one source document together. Official test pages never enter any CV fold. If a document group spans development and official test, its development pages are excluded from all folds. Exact image duplicates were already removed during preparation.

Grouping must be trustworthy. The default uses `document_id` preserved from the source metadata. If unavailable, supply either:

```bash
# CSV header: id,document_id OR filename,document_id; cover every selected page, including test.
indicdlp make-folds --data data/indicdlp_selected --output folds/indicdlp_selected \
  --n-splits 5 --groups-csv document_groups.csv

# Example ONLY for verified filenames such as source12_page_003.png:
indicdlp make-folds --data data/indicdlp_selected --output folds/indicdlp_selected \
  --n-splits 5 --group-regex '^(.*?)_page_[0-9]+\.[^.]+$'
```

Opaque filenames are never guessed into documents. `--allow-page-level` explicitly permits missing document IDs, but that mode **cannot establish source-document independence** and is recorded in the reports. Review `plan.json`: it contains fold page IDs, document groups, test exclusions, languages/domains, and per-class annotation counts. Sparse language/domain strata are combined for splitting and recorded; equal class frequencies across folds are not guaranteed.

## 4. Check the A6000 recipe and run cross-validation

```bash
indicdlp crossval --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --config configs/training/a6000.yaml --name selected_a6000 --dry-run

indicdlp crossval --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --config configs/training/a6000.yaml --name selected_a6000 \
  --wandb-group selected-a6000
```

Each fold starts from the **same original pretrained checkpoint**, verified by SHA-256. A previous fold's best model is never used as the next fold's initialization. Folds run sequentially on GPU 0 with one W&B run each. A separate W&B summary run logs the CV aggregates. Five folds cost approximately five training runs; no actual test metrics are used for early stopping or hyperparameter selection.

| Setting | Standard A6000 | High resolution A6000 |
| --- | ---: | ---: |
| Config | `configs/training/a6000.yaml` | `configs/training/a6000_highres.yaml` |
| Image size | 1280 | 1600 |
| Physical batch | 8 | 8 |
| Nominal batch (`nbs`) | 64 | 64 |
| Steady-state gradient accumulation | about 8 steps | about 8 steps |
| Optimizer | AdamW | AdamW |
| Initial LR → cosine final LR | 2e-4 → 2e-6 | 2e-4 → 2e-6 |
| Warmup | 5 epochs, bias LR starts at 0 | same |
| Weight decay | 5e-4 | 5e-4 |
| Epoch limit / patience | 150 / 30 | 150 / 30 |
| AMP / workers | disabled (FP32) / 8 | enabled / 8 |

Accumulation and decay scaling are handled by the upstream trainer; warmup accumulation ramps toward the nominal batch. Flips, mosaic, mixup, perspective, and shear are disabled. Rotation is limited to ±1 degree, translation to 2%, scale variation to 10%, and brightness variation to 10%. All 42 region class IDs are retained. `max_det=1000` accommodates dense documents such as newspapers. No per-class weighting is invented without validation evidence.

Explicit CLI flags override YAML, including `--batch 8`, `--lr0 0.0001`, `--no-amp`, `--epochs 1`, and `--wandb offline`. Start with a short run on the smoke subset to measure memory before committing to five full runs. Reduce batch first on OOM; A6000 memory capacity alone does not guarantee a particular batch will fit. The custom tracking/CV lifecycle currently runs on **one GPU per process**.

If startup fails with `FileNotFoundError: yolov8n.pt` during the AMP check,
the DocLayout pretrained weights may already be loaded correctly. The upstream
checker loads an additional YOLOv8n model from a fork asset URL; a missing file
is not caught by that checker. Run in FP32 to avoid this download dependency:

```bash
CUDA_VISIBLE_DEVICES=1 python train_indicdlp.py train \
  --data data/indicdlp_selected_v2 --config configs/training/a6000.yaml \
  --no-amp --batch 8 --device 0 --name selected_a6000_gpu1_fp32 --wandb online
```

This works with the existing CLI and needs no package update. Alternatively,
`configs/training/a6000_fp32.yaml` records the same FP32 recipe with batch 8.
Full precision can run slower and use more memory; reduce to `--batch 4` on OOM.
Nominal batch stays at 64, and upstream accumulation adapts to the physical batch.
Use a fresh run name when changing precision; this failure happens before the
first training epoch, so start again without `--resume`. Training and validation
W&B tracking remains enabled. No additional dataset download is required.

The standard `a6000.yaml` now defaults to `amp: false`, batch 8. The trainer
adapter also enforces a requested FP32 setting in `on_pretrain_routine_start`,
after checkpoint configuration restoration, before the upstream AMP check.
It prints `Training precision: FP32 (amp=False)` at that point. High-resolution
AMP training remains an explicit alternative requiring a working upstream check.
When troubleshooting an older server checkout, first confirm that its launch
preview contains `"amp": false`; changing a local PC file does not update the
server checkout. Use `--dry-run` with the same arguments to inspect the preview.

Results live in:

```text
runs/indicdlp/selected_a6000_fold01/weights/{best,last}.pt
runs/indicdlp/selected_a6000_fold01/reports/epochs/*.json
runs/indicdlp/selected_a6000_fold01/reports/validation_best/
runs/indicdlp/selected_a6000_crossval/{experiment,summary}.json
runs/indicdlp/selected_a6000_crossval/fold_01.json
```

Use the checkpoint paths recorded in the fold JSON; the upstream trainer may increment a run folder name if it already exists. `summary.json` reports fold mean, **sample** standard deviation, range, observed-fold counts per class, and median best epoch. These are fold summaries, not a pooled detector mAP or a confidence interval. A partial fold run is explicitly marked incomplete.

Completed fold JSON/checkpoints are verified and reused on rerun. To resume an interrupted fold while reusing completed folds:

```bash
indicdlp crossval --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --config configs/training/a6000.yaml --name selected_a6000 \
  --wandb-group selected-a6000 \
  --resume-fold 3 --resume runs/indicdlp/selected_a6000_fold03/weights/last.pt
```

The checkpoint's dataset, fold plan, and fold index must match. Online W&B resumes the persisted run ID/project; offline W&B creates another local run because its SDK does not provide equivalent online resume semantics. Changing the CV plan, initialization, or training hyperparameters requires a new experiment name.

## 5. Evaluate held-out test pages once the recipe is frozen

Predeclare a model to test, such as fold 1, **before inspecting test results**. Read its `best` path from `runs/indicdlp/selected_a6000_crossval/fold_01.json` and use that path below:

```bash
indicdlp test --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --weights runs/indicdlp/selected_a6000_fold01/weights/best.pt \
  --config configs/training/a6000.yaml --name selected_a6000_test \
  --wandb-group selected-a6000 --report-dir reports/selected_a6000_test
```

This tests that predeclared fold model; it does not claim to refit a final model on every development page. A grouped test run checks the checkpoint's saved training context against the fold plan before loading the model. The test set is never appended to training or validation. `test` always means test; use `evaluate --split val --folds DIR --fold 1` for an individual fold's validation set.

Metric artifacts are generated **after the server evaluation**; no real-data accuracy scores have been fabricated locally. W&B receives aggregate and per-class precision, recall, F1, AP50, AP75, AP50–95, annotation support, timing, confusion-matrix data, and PR/F1/P/R curves. During training it also receives **every trainer-provided training and validation loss component**, optimizer-group learning rates, and epoch metrics. Missing-target classes have undefined AP (`null`), while target-present classes with no detections get zero AP.

The explicit report directory contains `metrics.json`, `per_class.csv`, `confusion_matrix.csv`, `curves.json`, and `summary.md`. Full curve points are kept in the metric artifact; dashboard plots are downsampled to respect W&B table limits. Reports contain numeric results and labels, not document images. `--wandb-log-model` optionally uploads the best checkpoint; it is off by default. `--wandb offline` stores SDK logs locally for later `wandb sync`.

Standalone inference does not compute detection losses; no fake test loss is logged. Train-time validation losses are logged per epoch. The SDK can also collect its normal system/GPU telemetry. For FIR-specific quality, additionally evaluate a separately annotated FIR corpus: the eight selected domains are not a guarantee of actual FIR documents, and IndicDLP has no standalone FIR category.

Sources: [NVIDIA A6000 memory](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/), [DocLayout-YOLO trainer](https://github.com/opendatalab/DocLayout-YOLO/blob/main/train.py), [grouped stratification](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
