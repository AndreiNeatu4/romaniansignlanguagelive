# Romanian Sign Language — Live Alphabet Recognition

Real-time recognition of the **Romanian manual alphabet** (30 letters: A–Z plus
Â, Ă, Ș, Ț) from a webcam. Landmarks come from MediaPipe (hands + face + pose),
features feed a **CNN-BiLSTM-Attention** model, and the whole thing ships as a
**zero-backend static site** that runs entirely in the browser and is hosted on
**Cloudflare Pages**.

## The model at a glance

| Property | Value |
|----------|-------|
| Architecture | `cnn_bilstm_attn` (1D-CNN → bidirectional LSTM → attention pooling) |
| Input window | **45 frames** @ 30 fps effective (~1.5 s) |
| Features / frame | **216** = 126 hand (2×21×3) + 60 face anchors (20×3) + 6 wrist-global + 6 fingertip-global + 6 wrist-velocity + 6 wrist-accel + 6 mask/reserved |
| Classes | 30 |
| Test accuracy | ~98.9% |

All of these are defined in [config.py](config.py) — edit settings there, not in
the scripts. ⚠️ `SEQUENCE_LENGTH` (45) and the 216-feature layout are baked into
the exported ONNX model and `static-site/model.js` / `model_meta.json`; changing
them requires re-exporting and updating the site (see below).

## Repository layout

```
config.py              All pipeline/model settings (single source of truth)
run_pipeline.py        Runs the 4 training stages in order
alphabet/<L>/<L>.mp4   Source videos — one reference clip per letter

data_preparation/
  feature_extractor.py        Shared 216-dim per-frame extractor (training + JS port parity)
  extract_augmented_fast.py   Stage 1 — videos → augmented temporal sequences (.npy)
  extract_static_images.py    Stage 2 — static letter images → sequences (.npy)
  prepare_augmented_dataset.py Stage 3 — combine .npy → dataset.pkl + class_labels.json
training/
  train_model.py              Stage 4 — train CNN-BiLSTM-Attn → models/alphabet/

models/alphabet/       Trained weights (best_model.pth) + class_labels.json + reports
setup/requirements.txt Python/training dependencies (PyTorch cu121, MediaPipe, …)

static-site/           The deployed Cloudflare app (pure HTML/JS/ONNX, no server)
  tools/export_onnx.py   best_model.pth → assets/model/sign_model.onnx (+ meta/labels)
  tools/build_assets.py  copy style.css + per-letter videos, write videos.json
  tools/parity_dump.py + parity_check.mjs  verify feature_extractor.js == Python
wrangler.jsonc         Cloudflare Pages config (serves ./static-site)

deprecated/            Retired scripts/docs from earlier workflows (see its README)
```

## End-to-end workflow

### 1. Set up the environment (training only)

```bash
pip install -r setup/requirements.txt
```

### 2. Train the model

```bash
python run_pipeline.py            # runs all 4 stages
# or skip stages you've already done:
python run_pipeline.py --skip-extract --skip-images
```

Data flow:

```
alphabet/  ──(stage 1: extract_augmented_fast)──┐
           ──(stage 2: extract_static_images)───┤→ data/alphabet_augmented/ (.npy)
                                                 │
   data/alphabet_augmented/ ─(stage 3: prepare)─→ data/alphabet_processed/dataset.pkl
                                                 │
   dataset.pkl ─────────────(stage 4: train)────→ models/alphabet/best_model.pth
```

### 3. Export the model + build the site assets

```bash
python static-site/tools/export_onnx.py     # best_model.pth → static-site/assets/model/sign_model.onnx
python static-site/tools/build_assets.py     # style.css + reference videos + videos.json
```

Optional — after editing `feature_extractor.js`, re-verify parity:

```bash
python static-site/tools/parity_dump.py      # dump Python extractor output
node   static-site/tools/parity_check.mjs    # assert JS matches within 1e-4
```

### 4. Deploy to Cloudflare

The site is static, so there's no build step. Cloudflare Pages serves the
`static-site/` directory (see [wrangler.jsonc](wrangler.jsonc)). Connect the repo
in Pages with **build command: none** and **output directory: `static-site`**, or
drag-and-drop the contents of `static-site/`. Full instructions and local-testing
notes are in [static-site/README.md](static-site/README.md).

## Deprecated

Earlier approaches (the FastAPI/Docker server, the desktop PyTorch recognizer,
one-off data-organisation and scraping scripts, and stale docs) live under
[deprecated/](deprecated/) and are **not** part of the current train → ONNX →
Cloudflare pipeline. See [deprecated/README.md](deprecated/README.md).
