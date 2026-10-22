"""Static description of the available models."""
from dataclasses import dataclass, field
from typing import Callable

from .model_info import TRAINING
from .models import SaliencyNet, build_v1_1, build_v2_0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    input_size: tuple  # (H, W)
    weights_file: str
    params: int
    reference: dict
    training: dict
    builder: Callable[[], SaliencyNet] = field(repr=False, compare=False)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "input_size": list(self.input_size),
            "weights_file": self.weights_file,
            "params": self.params,
            "reference": dict(self.reference),
            "training": {
                k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.training.items()
            },
        }


REGISTRY: dict = {
    "v1.1": ModelSpec(
        id="v1.1",
        label="v1.1 · CNN encoder",
        description="VGG-style CNN encoder trained from scratch on SALICON, "
                    "3-branch deep-supervision decoder.",
        input_size=(240, 320),
        weights_file="saliency_v1.1.safetensors",
        params=15_085_165,
        # Best checkpoint evaluated on the whole MIT1003 set (notebook results).
        reference={"dataset": "MIT1003", "total_loss": 7.6305, "kld": 0.9229, "cc": 0.6044, "sim": 0.4809},
        training=TRAINING["1.1"],
        builder=build_v1_1,
    ),
    "v2.0": ModelSpec(
        id="v2.0",
        label="v2.0 · Swin-T encoder",
        description="Swin-Tiny encoder (ImageNet-1k init, last stage fine-tuned) on SALICON, "
                    "same 3-branch decoder.",
        input_size=(224, 224),
        weights_file="saliency_v2.0.safetensors",
        params=27_898_711,
        reference={"dataset": "MIT1003", "total_loss": 7.5088, "kld": 0.9436, "cc": 0.5929, "sim": 0.4609},
        training=TRAINING["2.0"],
        builder=build_v2_0,
    ),
}

DEFAULT_MODEL = "v1.1"


def get_spec(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except KeyError:
        raise KeyError(f"unknown model id {model_id!r} (available: {', '.join(REGISTRY)})") from None
