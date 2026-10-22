import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from saliency.registry import REGISTRY  # noqa: E402
from saliency.weights import load_model, load_original_pth, weights_path  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))

# Original training checkpoints (`<dir>/<version>/*_best.pth`); the tests that need them are skipped otherwise.
PTH_DIR = Path(os.environ.get("SALIENCY_PTH_DIR", REPO / "checkpoints"))
WEIGHTS_DIR = REPO / "weights"
SAMPLES_DIR = REPO / "samples"
MODEL_IDS = list(REGISTRY)


def has_pth(model_id):
    return (PTH_DIR / model_id.lstrip("v") / "encoder_best.pth").is_file()


def has_converted(model_id):
    return weights_path(REGISTRY[model_id], WEIGHTS_DIR).is_file()


def need_pth(model_id):
    if not has_pth(model_id):
        pytest.skip(f"original checkpoints not found in {PTH_DIR} (set SALIENCY_PTH_DIR)")


def need_converted(model_id):
    if not has_converted(model_id):
        pytest.skip(f"converted weights not found in {WEIGHTS_DIR}")


_cache = {}


@pytest.fixture(scope="session")
def original_model():
    def get(model_id):
        need_pth(model_id)
        key = ("pth", model_id)
        if key not in _cache:
            _cache[key] = load_original_pth(REGISTRY[model_id], PTH_DIR)
        return _cache[key]
    return get


@pytest.fixture(scope="session")
def converted_model():
    def get(model_id):
        need_converted(model_id)
        key = ("st", model_id)
        if key not in _cache:
            _cache[key] = load_model(REGISTRY[model_id], "cpu", WEIGHTS_DIR)
        return _cache[key]
    return get
