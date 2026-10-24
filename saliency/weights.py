"""Loading model weights (converted safetensors files or the original .pth checkpoints)."""
from pathlib import Path

import torch
from safetensors.torch import load_file

from . import paths
from .models import SUBMODULES, SaliencyNet


class WeightsNotFoundError(FileNotFoundError):
    pass


def weights_path(spec, weights_dir=None) -> Path:
    root = Path(weights_dir) if weights_dir is not None else paths.weights_dir()
    return root / spec.weights_file


def load_state_dict(path) -> dict:
    """Read a converted file; every floating tensor comes back as fp32."""
    sd = load_file(str(path), device="cpu")
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}


def load_model(spec, device="cpu", weights_dir=None) -> SaliencyNet:
    path = weights_path(spec, weights_dir)
    if not path.is_file():
        raise WeightsNotFoundError(
            f"Weights for model {spec.id} not found: expected file '{spec.weights_file}' "
            f"at {path}. Put it in that folder or point the SALIENCY_WEIGHTS_DIR environment "
            f"variable to the folder that contains it."
        )
    model = spec.builder()
    model.load_state_dict(load_state_dict(path), strict=True)
    return model.float().to(device).eval()


def load_original_pth(spec, pth_dir) -> SaliencyNet:
    """Build the model from the five ``<part>_best.pth`` files of a training run.

    ``pth_dir`` is either the run folder itself or its parent (``<pth_dir>/<version>/``).
    """
    pth_dir = Path(pth_dir)
    version = spec.id.lstrip("v")
    if not (pth_dir / "encoder_best.pth").is_file() and (pth_dir / version).is_dir():
        pth_dir = pth_dir / version
    model = spec.builder()
    for name in SUBMODULES:
        f = pth_dir / f"{name}_best.pth"
        if not f.is_file():
            raise WeightsNotFoundError(f"Checkpoint for model {spec.id} not found: expected {f}")
        sd = torch.load(f, map_location="cpu", weights_only=True)
        getattr(model, name).load_state_dict(sd, strict=True)
    return model.float().eval()
