"""FastAPI server: JSON API for the saliency models plus the static frontend."""

from __future__ import annotations

import base64
import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from saliency import paths
from saliency.registry import DEFAULT_MODEL, REGISTRY, get_spec
from saliency.weights import WeightsNotFoundError, weights_path

APP_VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def _png_data_url(arr: np.ndarray) -> str:
    """Encode a float map in [0, 1] as an 8-bit grayscale PNG data URL."""
    img = Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _open_image(data: bytes, what: str) -> Image.Image:
    if len(data) == 0:
        raise HTTPException(400, f"The {what} file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The {what} file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise HTTPException(413, f"The {what} has too many pixels (limit {MAX_PIXELS:,}).")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(400, f"The {what} is not a supported image (use JPEG, PNG, WebP, BMP or GIF).")
    # Phone photos store their orientation in EXIF: apply it so the map matches what the user sees.
    return ImageOps.exif_transpose(img)


@lru_cache(maxsize=1)
def _samples() -> dict[str, dict]:
    root = paths.samples_dir()
    index = root / "samples.json"
    if not index.is_file():
        return {}
    out: dict[str, dict] = {}
    for entry in json.loads(index.read_text(encoding="utf-8")):
        sid = entry["id"]
        image = root / f"{sid}.jpeg"
        if not image.is_file():
            continue
        gt = root / f"{sid}_fixMap.jpg"
        with Image.open(image) as im:
            width, height = ImageOps.exif_transpose(im).size
        out[sid] = {
            "id": sid,
            "title": entry.get("title", sid),
            "image_url": f"/samples/{image.name}",
            "gt_url": f"/samples/{gt.name}" if gt.is_file() else None,
            "has_gt": gt.is_file(),
            "width": width,
            "height": height,
            "_image": image,
            "_gt": gt if gt.is_file() else None,
        }
    return out


def create_app(device: str = "auto", weights_dir: str | Path | None = None) -> FastAPI:
    # torch is imported lazily so that `--help` and import errors stay fast and clear.
    from saliency.inference import Predictor, device_info

    predictor = Predictor(device=device, weights_dir=weights_dir)
    lock = threading.Lock()  # one inference at a time keeps memory use predictable
    app = FastAPI(title="Visual Saliency Prediction", version=APP_VERSION, docs_url=None, redoc_url=None)

    def _available(model_id: str) -> bool:
        return predictor.available(model_id)

    @app.get("/api/health")
    def health():
        import timm
        import torch

        return {
            "status": "ok",
            "device": device_info(predictor.device),
            "versions": {"torch": torch.__version__, "timm": timm.__version__, "app": APP_VERSION},
            "models": [
                {
                    "id": spec.id,
                    "available": _available(spec.id),
                    "weights_path": str(weights_path(spec, weights_dir)),
                }
                for spec in REGISTRY.values()
            ],
        }

    @app.get("/api/models")
    def models():
        return {
            "default": DEFAULT_MODEL,
            "models": [{**spec.to_json(), "available": _available(spec.id)} for spec in REGISTRY.values()],
        }

    @app.get("/api/samples")
    def samples():
        return {"samples": [{k: v for k, v in s.items() if not k.startswith("_")} for s in _samples().values()]}

    @app.get("/samples/{name}")
    def sample_file(name: str):
        for s in _samples().values():
            if name == s["_image"].name:
                return FileResponse(s["_image"])
            if s["_gt"] is not None and name == s["_gt"].name:
                return FileResponse(s["_gt"])
        raise HTTPException(404, "Sample not found.")

    @app.post("/api/predict")
    async def predict(
        image: UploadFile | None = File(None),
        sample_id: str | None = Form(None),
        models: str | None = Form(None),
        gt: UploadFile | None = File(None),
        use_gt: str | None = Form(None),
    ):
        if image is not None and not image.filename:
            image = None
        if (image is None) == (not sample_id):
            raise HTTPException(400, "Send either an image file or a sample_id (exactly one).")

        model_ids = [m.strip() for m in (models or DEFAULT_MODEL).split(",") if m.strip()]
        if not model_ids:
            raise HTTPException(400, "No model selected.")
        for mid in model_ids:
            try:
                get_spec(mid)
            except KeyError:
                raise HTTPException(400, f"Unknown model '{mid}'. Available: {', '.join(REGISTRY)}.")

        gt_img = None
        if image is not None:
            name = image.filename or "image"
            img = _open_image(await image.read(MAX_UPLOAD_BYTES + 1), "image")
        else:
            sample = _samples().get(sample_id)
            if sample is None:
                raise HTTPException(400, f"Unknown sample '{sample_id}'.")
            name = sample["_image"].name
            img = ImageOps.exif_transpose(Image.open(sample["_image"]))
            wants_gt = (use_gt or "1").strip().lower() not in ("0", "false", "no", "off")
            if wants_gt and sample["_gt"] is not None:
                gt_img = Image.open(sample["_gt"])
        if gt is not None and gt.filename:
            gt_img = _open_image(await gt.read(MAX_UPLOAD_BYTES + 1), "fixation map")

        img = img.convert("RGB")
        width, height = img.size
        if gt_img is not None:
            # Metrics use the map as is (resized to the model input like in the notebooks);
            # only the copy returned for display is matched to the image size.
            gt_img = gt_img.convert("L")

        results = []
        for mid in model_ids:
            try:
                with lock:
                    pred = predictor.predict(img, mid)
                    metrics = predictor.metrics(pred, gt_img) if gt_img is not None else None
            except WeightsNotFoundError as exc:
                raise HTTPException(503, str(exc))
            results.append(
                {
                    "model": mid,
                    "input_size": list(pred.input_size),
                    "inference_ms": round(pred.inference_ms, 1),
                    "map_png": _png_data_url(pred.map_full),
                    "metrics": {k: round(float(v), 4) for k, v in metrics.items()} if metrics else None,
                }
            )

        gt_json = None
        if gt_img is not None:
            shown = gt_img if gt_img.size == img.size else gt_img.resize(img.size, Image.BILINEAR)
            gt_json = {"png": _png_data_url(np.asarray(shown, dtype=np.float32) / 255.0)}
        return {"image": {"name": name, "width": width, "height": height}, "gt": gt_json, "results": results}

    static = paths.static_dir()

    @app.middleware("http")
    async def revalidate_frontend(request, call_next):
        response = await call_next(request)
        # Revalidate the page and its JS/CSS on every load (a cheap 304 via ETag), so browsers
        # never keep running a stale ES module after the frontend is updated.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
