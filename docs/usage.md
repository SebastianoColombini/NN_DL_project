# Local GUI: installation, usage, weights, evaluation

A small local web app that runs the trained saliency models of this project on your own
images or on MIT1003 samples, and shows the predicted saliency maps next to the
ground-truth fixation maps.

- Models: **v1.1** (CNN encoder trained from scratch, input 240×320) and **v2.0**
  (Swin-Tiny encoder, input 224×224). Both use the 3-branch deep-supervision decoder of the
  notebooks; the code is copied unchanged from `notebooks/v1.1.ipynb` and `notebooks/v2.0.ipynb`.
- Views: overlay, map, side by side; compare both models; jet / inferno / viridis / gray
  colormaps; opacity slider; CC / KLD / SIM against the fixation map when one is available;
  PNG download of maps and overlays.
- Everything runs on your machine: no data leaves it and the app needs no internet access
  once installed. The trained weights are in the repo (`weights/`).

## Requirements

Measured with PyTorch 2.14 (CPU) on a 4-core machine:

| | params | weights file | model input | CPU time / image | RAM (process) |
|---|---|---|---|---|---|
| v1.1 | 15.09 M | 29 MB | 240×320 | ~1.1 s | ~0.5 GB |
| v2.0 | 27.90 M | 54 MB | 224×224 | ~0.6 s | ~0.5 GB |

- **Minimum:** any 64-bit CPU, ~2 GB of free RAM, Python 3.10 or newer (tested with 3.12). A GPU is not needed:
  one image takes about a second on CPU.
- **GPU (optional):** an NVIDIA card with CUDA, or Apple Silicon (MPS). Inference needs about
  0.5–0.7 GB of VRAM per model (mostly the CUDA context), so a card with 8 GB is more than
  enough.
- Disk: ~1 GB for the CPU build of PyTorch, ~3 GB for the CUDA build.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

Install PyTorch for your hardware first, then the rest:

```bash
# CPU only (Linux / Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
#   e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# macOS (Apple Silicon or Intel): the default wheels, MPS is used automatically
pip install torch torchvision

pip install -r requirements.txt
```

`timm` is pinned to `1.0.30`: the checkpoints are loaded in strict mode and a different timm
version could rename the Swin layers.

## Running

```bash
python -m app                      # opens http://localhost:7860 in the browser
python -m app --no-browser --port 8000
python -m app --device cpu         # auto (default) = cuda > mps > cpu
```

The server listens on `127.0.0.1` only. Drop images on the page (or pick a sample), choose a
model or "Compare", and switch views and colormaps freely: colormap and opacity are applied
in the browser, so they do not re-run the model.

Notes on the input:
- JPEG, PNG, WebP, BMP and GIF (first frame), up to 20 MB.
- The EXIF orientation of phone photos is applied before prediction, so the map matches the
  photo as you see it.
- The preprocessing is the one used in training (resize to the model input, no
  normalisation); the map is then resized back to the original image size.
- Metrics need a fixation map: samples have one, and for your own image you can add one
  ("add fixation map"). As in the notebooks, the map is resized to the model resolution
  before computing CC, KLD and SIM.
- The displayed maps are stretched to their own maximum and the overlay is transparent where
  saliency is low; downloads use the same rendering.

## Weights

`weights/saliency_v1.1.safetensors` and `weights/saliency_v2.0.safetensors` hold the best
checkpoints of the two runs (`*_best.pth` from training), one file per model, with keys
`encoder.*`, `decoder3.*`, `decoder4.*`, `decoder5.*`, `final.*`.

- Float tensors with 2 or more dimensions are stored in fp16, everything else (biases,
  BatchNorm/LayerNorm statistics, integer buffers) as in the original; at load time all
  floats go back to fp32. This keeps both files well under GitHub's 100 MB limit (the original
  v2.0 encoder alone is 110 MB).
- On the whole MIT1003 set the converted weights give the same metrics as the originals within
  1e-4 (largest per-pixel difference between the maps: 0.002 for v1.1, 0.007 for v2.0).
- `weights/MANIFEST.json` lists sizes, sha256 and the source checkpoints.
- Another folder can be used with `--weights-dir` or the `SALIENCY_WEIGHTS_DIR` variable.

To regenerate them from the original checkpoints (`<dir>/1.1/*_best.pth`, `<dir>/2.0/...`):

```bash
python tools/convert_weights.py --src /path/to/checkpoints --out weights
```

## Evaluation and tests

```bash
# MIT1003: notebook numbers vs original checkpoints vs converted weights
python tools/eval_mit1003.py --model all --mit-dir /path/to/mit --pth-dir /path/to/checkpoints \
    --weights-dir weights --out-dir eval_out
pytest
```

`--mit-dir` contains the extracted `ALLSTIMULI/` and `ALLFIXATIONMAPS/`. Tests that need the
original checkpoints look for them in `checkpoints/<version>/` (or `SALIENCY_PTH_DIR`) and are
skipped otherwise.

| MIT1003 | Total loss | KLD ↓ | CC ↑ | SIM ↑ |
|---|---|---|---|---|
| v1.1 (notebook) | 7.6305 | 0.9229 | 0.6044 | 0.4809 |
| v1.1 (this repo, converted weights) | 7.6305 | 0.9229 | 0.6043 | 0.4809 |
| v2.0 (notebook) | 7.5088 | 0.9436 | 0.5929 | 0.4609 |
| v2.0 (this repo, converted weights) | 7.5087 | 0.9436 | 0.5930 | 0.4609 |

## Adding a model version

1. Put the model classes in `saliency/models/vX_Y.py` (copied unchanged from the notebook) with
   a `build_vX_Y()` that returns a `SaliencyNet` (submodules `encoder`, `decoder3`,
   `decoder4`, `decoder5`, `final`).
2. Add a `ModelSpec` to `REGISTRY` in `saliency/registry.py` (id, label, input size, weights
   file, params, MIT1003 reference metrics, training info). The GUI picks it up automatically.
3. Convert the checkpoints (`<dir>/X.Y/*_best.pth`):
   `python tools/convert_weights.py --src <dir> --out weights --models vX.Y`.

## Restyling

The look of the GUI is defined by the design tokens at the top of `app/static/styles.css`
(colours, fonts, radius, spacing, shadows, with a dark-theme override).
