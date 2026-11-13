# Local validation — 2026-11-13

Working directory: `F:\gitRepos\DocLayout`.

- **89 automated tests passed**, including real PyArrow 24.0.0 Parquet reads through the seekable range reader, complete synthetic prepare/verify/resume flows, grouped cross-validation, W&B payload/lifecycle tests, held-out checkpoint provenance, and JSON/CSV report generation. Tests block requests to external services.
- The synthetic source includes newspaper-only and Forms row groups in train/val/test. Tests assert that only the Forms-containing row groups request image columns and only Forms pages are exported.
- HTTP 200 whole-file fallback is refused before consuming its response body. Hard byte budget, row-group memory guard, changed-selection rejection, and file-hash tamper detection pass.
- CLI help and Ruff checks passed.
- The project was reorganized into an installable `src/` package with separate data, engine, evaluation, and tracking modules. The wheel built offline, matched every current source module, and passed an isolated local installation/CLI check with all 42 packaged class names.
- **Actual DocLayout-YOLO 0.0.4 CPU optimizer smoke passed**: `yolov10m-doclayout.yaml`, 42 classes, batch 2, image size 64, one forward/loss/backward/AdamW step. Loss: approximately **32.5649**. All losses/gradients were finite and model weights changed. Result: `artifacts/model_smoke.json`.
- Runtime for that isolated model step: Python 3.12, torch 2.13.0+cpu, torchvision 0.28.0+cpu. This verifies model computation, not the older upstream trainer's complete checkpoint lifecycle. Server training dependencies deliberately use torch 2.5.1 / torchvision 0.20.1 to support that lifecycle.
- An additional integration smoke exercised the actual upstream `DetMetrics` on synthetic detection statistics: per-class precision/recall/F1/AP, all four curves, and undefined AP for classes without targets were exported successfully. The PC has NumPy 2.4.6, so this isolated test temporarily aliased removed `np.trapz` to `np.trapezoid`; production dependencies pin NumPy below 2 and require no such alias. These synthetic statistics are not real model accuracy measurements.
- Model, OpenCV, torchvision, and Arrow imports for local validation came from existing pip-cache wheels temporarily extracted into the ignored `artifacts/test-deps` directory, then removed after validation. The system Python installation was not modified.
- **IndicDLP document images downloaded: 0. Pretrained checkpoints downloaded: 0.** No full dataset, shard, or model checkpoint download was attempted.

Not locally verified:

- The gated HF row schema and actual annotation ID convention. Public dataset metadata and upstream code were inspected, but this environment's shell cannot connect to HF, and gated rows were not available through public browsing. Run the bounded `inspect` command on the authenticated server; use explicit adapter overrides if needed.
- Live authenticated HF redirects/range responses. Real Parquet parsing over a simulated HTTP transport passed; external server behavior remains a server preflight check.
- CUDA execution, distributed training, long-run convergence, full-epoch checkpoint/resume, and model accuracy on real FIRs. Run the small server smoke subset before the full selected subset.
- Live W&B authentication, SDK uploads, and online/offline server lifecycle. Local tests use a fake SDK to check every metric/table/curve/artifact payload and resume behavior without contacting W&B.
