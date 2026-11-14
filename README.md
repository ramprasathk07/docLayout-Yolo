# IndicDLP ? DocLayout-YOLO

An installable Python codebase for bounded Hugging Face data streaming, document-group cross-validation, A6000 training, W&B tracking, and separate held-out testing.

The requested download preset selects **Acts & Rules, Brochures, Forms, Newspapers, Manuals, Notices, Novels, and Syllabi**. Only selected pages and labels are retained; the entire dataset is not downloaded/cached. Parquet row groups can include some unrelated image bytes in transit. IndicDLP has no separate FIR domain; these are document-layout proxies, not verified FIR records.

## Codebase

```text
pyproject.toml                    # installable package, extras, CLI, test configuration
src/indicdlp/
  cli.py                         # inspect / prepare / make-folds / train / crossval / test
  config.py                      # strict recipe validation and CLI precedence
  data/                          # HTTP ranges, HF Parquet, schema adapters, export, folds
  engine/                        # DocLayout-YOLO training and independent CV orchestration
  evaluation/                    # detection metrics, held-out runner, JSON/CSV/MD reports
  tracking/wandb.py               # W&B lifecycle, epoch metrics, tables, curves, artifacts
  resources/classes.json         # official 42-class order, included in package wheels
configs/
  data/selected.yaml              # the eight requested domains and streaming limits
  training/a6000.yaml             # 1280 px, batch 8, FP32 on one 48 GB A6000
  training/a6000_highres.yaml     # 1600 px, batch 8
  training/a6000_fp32.yaml        # 1280 px, batch 8; no AMP-check asset download
scripts/
  download_selected_indicdlp.py  # bounded eight-domain downloader
  offline_model_smoke.py         # actual model/loss/backward/optimizer test, no weights download
tests/                           # offline synthetic unit/integration tests and shared fixtures
docs/
  server-a6000.md                 # setup, credentials, recipes, CV, resume, test commands
  data-streaming.md              # live schema inspection, adapters, transfer/storage behavior
  validation.md                  # local checks and remaining server-only verification
train_indicdlp.py                # thin repository entry point, compatible with earlier commands
```

## Run on the server

Follow [docs/server-a6000.md](docs/server-a6000.md) for installation and environment variables (`HF_TOKEN`, `WANDB_API_KEY`, optional project/entity). The workflow is:

Use a separate Python 3.11 training environment with **torch 2.5.1 / torchvision
0.20.1**. Torch 2.6+ changes checkpoint-loading defaults and is rejected by this
upstream fork. Commands load repository/current-directory `.env` files without
overriding exported variables. Online W&B requires `WANDB_API_KEY`; it never
prompts interactively. `WANDB_BASE_URL` can select a custom API host.

The training extra requires **W&B 0.31.x**, which supports modern long API keys.
After updating, run `python -m pip install --upgrade -e '.[train]'` to refresh
installed dependencies. Use `python scripts/check_wandb.py` to verify credentials
without starting training or creating a run. Do not truncate API keys to 40 characters.

```bash
# After installing the package and accepting the HF access gate:
indicdlp inspect --max-transfer-mib 16
python scripts/download_selected_indicdlp.py

# Requires trustworthy source-document IDs; the guide covers CSV/regex overrides.
indicdlp make-folds --data data/indicdlp_selected \
  --output folds/indicdlp_selected --n-splits 5 --seed 42

indicdlp crossval --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --config configs/training/a6000.yaml --name selected_a6000 \
  --wandb-group selected-a6000

# Predeclare the model to test; use the exact checkpoint path saved in fold_01.json.
indicdlp test --data data/indicdlp_selected --folds folds/indicdlp_selected \
  --weights runs/indicdlp/selected_a6000_fold01/weights/best.pt \
  --config configs/training/a6000.yaml --name selected_a6000_test \
  --wandb-group selected-a6000 --report-dir reports/selected_a6000_test
```

Official test images remain separate from the CV development pool. Folds share no document groups; development documents overlapping test groups are purged. Every fold starts from the same initial weights. W&B receives training/validation losses, learning rates, aggregate and per-class detection metrics; final validation/test reports add confusion matrices, curves, and numeric artifacts. Test metrics are computed only by the explicit test step, after model/recipe selection.

Use `--dry-run` for train/CV previews without weights downloads or W&B initialization. Use `--wandb offline` to keep tracking local. Explicit flags override recipe values. The preset transfer ceiling is 32 GiB per invocation; repeat preparation with the same selection to resume. A lower image cap does not guarantee a small network transfer because of Parquet row-group compression.

## Development and validation

```bash
python -m pip install -e '.[train,dev]'
python -m pytest -q -p no:cacheprovider
ruff check src tests scripts train_indicdlp.py
ruff format --check src tests scripts train_indicdlp.py
python scripts/offline_model_smoke.py
```

Tests use generated images/Parquet files and prohibit external requests. No IndicDLP images or pretrained checkpoints were downloaded on this PC. The live gated row schema, online W&B service, CUDA training, and real-data accuracy still require the authenticated server checks. See [validation evidence](docs/validation.md).

Sources: [IndicDLP dataset](https://huggingface.co/datasets/ai4bharat/indicdlp), [official class order](https://github.com/AI4Bharat/IndicDLP/blob/main/data/indic.yaml), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [default pretrained weights](https://huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain).
