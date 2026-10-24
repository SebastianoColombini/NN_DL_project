# GUI: architecture and standalone roadmap

## Architecture

```
browser (index.html, styles.css, app.js + js/*)        offline, no build step
   │  fetch JSON / multipart
   ▼
app/server.py  (FastAPI, 127.0.0.1)                    HTTP API, input checks, PNG encoding
   │
   ▼
saliency/      (plain Python package, no GUI code)
   registry.py   ModelSpec per version: label, input size, weights file, params,
                 MIT1003 reference metrics, training curves
   models/       notebook classes copied unchanged + SaliencyNet wrapper
   weights.py    safetensors -> strict load_state_dict -> fp32 -> eval()
   inference.py  training preprocessing, Predictor (lazy, thread-safe model cache),
                 device choice cuda > mps > cpu, timings, metrics
   metrics.py    notebook losses (KLD, CC, SIM, total loss)
   paths.py      resource root (repo or frozen bundle), SALIENCY_WEIGHTS_DIR
```

- **One process, one entrypoint:** `python -m app`. Models are loaded on first use and kept in
  memory; one prediction runs at a time (a lock), which keeps memory use predictable.
- **Rendering in the browser:** the server returns each map as an 8-bit grayscale PNG at the
  original image size. The browser turns it into a colour map with a lookup table on a
  `<canvas>`, so colormap, opacity and view changes are instant and never call the server.
  Results are cached per image and model.
- **Fidelity:** preprocessing is exactly the training one (`Resize` + `ToTensor`, no
  normalisation). Metrics resize the fixation map with the same transform, as in the
  notebooks. `tools/eval_mit1003.py` reproduces the notebook MIT1003 numbers to 4 decimals.

### HTTP API

| route | |
|---|---|
| `GET /api/health` | device (type, GPU name, VRAM), torch/timm versions, which weight files are present |
| `GET /api/models` | model specs for the UI (input size, params, reference metrics, training curves) |
| `GET /api/samples`, `GET /samples/<file>` | bundled MIT1003 samples and their fixation maps |
| `POST /api/predict` | multipart: `image` or `sample_id`, `models=v1.1,v2.0`, optional `gt`; returns maps (PNG data URLs at original size), ms, metrics |

Errors are JSON `{"detail": "..."}`: 400 bad input, 413 too large, 503 missing weights (the
message says which file is expected where).

### Restyling

All colours, fonts, radii, spacing and shadows are CSS variables in `:root` at the top of
`app/static/styles.css`, with a dark-theme override under
`@media (prefers-color-scheme: dark)`. The components only use these tokens.

## Standalone application (roadmap, not implemented yet)

Constraints already respected by the code:
- every file is located through `saliency/paths.py`, which uses `sys._MEIPASS` in a frozen
  build;
- no network access at runtime (timm is created with `pretrained=False`, the page loads no
  external resource);
- a single entrypoint (`app/__main__.py`) and weights inside the app (`weights/`).

**S1 — PyInstaller, one folder.** Bundle the CPU build of torch with `weights/`,
`app/static/` and `samples/` as data files:

```bash
pyinstaller -n SaliencyGUI --onedir \
  --add-data "weights:weights" --add-data "app/static:app/static" --add-data "samples:samples" \
  --collect-submodules timm app/__main__.py
```

Expected size 400–700 MB (mostly torch). PyInstaller does not cross-compile: build on each OS,
later with a GitHub Actions matrix (windows-latest, macos-14, ubuntu-latest) that uploads the
zipped folders as release assets.

**S2 — native window (optional).** Replace the browser with `pywebview`: start uvicorn in a
thread and open a window on `http://127.0.0.1:<port>`. Same frontend, no code changes.

**S3 — light build without torch.** Export each model to ONNX (opset ≥ 17), check it
numerically against PyTorch (Swin attention included, e.g. max map difference < 1e-3 on the
samples), and run it with `onnxruntime` (numpy-only preprocessing). The app drops to about
100–150 MB and can use DirectML / CUDA / CoreML execution providers for GPUs.

Once the final model is chosen, the standalone build can ship only that one: remove the other
entry from `REGISTRY` and its weights file.
