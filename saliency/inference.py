"""Single-image inference with the same preprocessing used during training."""
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .metrics import compute_metrics
from .registry import get_spec
from .weights import load_model, weights_path


def pick_device(pref: str = "auto") -> torch.device:
    """'auto' picks cuda > mps > cpu; an explicit choice must be available."""
    pref = (pref or "auto").lower()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if pref == "mps" and not has_mps:
        raise ValueError("MPS requested but not available")
    return torch.device(pref)


def device_info(device) -> dict:
    device = torch.device(device)
    info = {"type": device.type, "name": None, "vram_total_mb": None, "vram_free_mb": None}
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info["vram_total_mb"] = int(total // 2**20)
        info["vram_free_mb"] = int(free // 2**20)
    elif device.type == "mps":
        info["name"] = "Apple MPS"
    return info


def _transform(size):
    # Same pipeline as the training notebooks (no normalisation).
    return transforms.Compose([transforms.Resize(tuple(size)), transforms.ToTensor()])


def preprocess_image(img: Image.Image, size) -> torch.Tensor:
    return _transform(size)(img.convert("RGB")).unsqueeze(0)


def preprocess_map(img: Image.Image, size) -> torch.Tensor:
    return _transform(size)(img.convert("L")).unsqueeze(0)


@dataclass
class Prediction:
    model_id: str
    map_model: np.ndarray
    map_full: np.ndarray
    inference_ms: float
    input_size: tuple
    tensor: torch.Tensor


class Predictor:
    """Keeps loaded models in memory; safe to share between threads."""

    def __init__(self, device="auto", weights_dir=None):
        if "OMP_NUM_THREADS" not in os.environ:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.device = pick_device(device) if isinstance(device, str) else torch.device(device)
        self.weights_dir = weights_dir
        self._models = {}
        self._lock = threading.Lock()

    def available(self, model_id) -> bool:
        return weights_path(get_spec(model_id), self.weights_dir).is_file()

    def get_model(self, model_id):
        model = self._models.get(model_id)
        if model is None:
            with self._lock:
                model = self._models.get(model_id)
                if model is None:
                    model = load_model(get_spec(model_id), self.device, self.weights_dir)
                    self._models[model_id] = model
        return model

    def predict(self, img: Image.Image, model_id) -> Prediction:
        spec = get_spec(model_id)
        model = self.get_model(model_id)
        x = preprocess_image(img, spec.input_size).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            out = model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            out = out.float().clamp_(0.0, 1.0)
            full = F.interpolate(out, size=(img.height, img.width), mode="bilinear", align_corners=False)
            full = full.clamp_(0.0, 1.0)
        # clone outside inference mode so callers get ordinary tensors
        out = out.cpu().clone()
        return Prediction(
            model_id=spec.id,
            map_model=out[0, 0].numpy().copy(),
            map_full=full[0, 0].cpu().numpy().copy(),
            inference_ms=elapsed,
            input_size=tuple(spec.input_size),
            tensor=out,
        )

    def metrics(self, pred: Prediction, gt: Image.Image) -> dict:
        gt_t = preprocess_map(gt, pred.input_size)
        return compute_metrics(pred.tensor, gt_t)
