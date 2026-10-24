import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from saliency.inference import preprocess_image, preprocess_map

from conftest import SAMPLES_DIR


@pytest.mark.parametrize("size", [(240, 320), (224, 224)])
def test_matches_training_transform(size):
    img = Image.open(SAMPLES_DIR / "i2252150223.jpeg")
    gt = Image.open(SAMPLES_DIR / "i2252150223_fixMap.jpg")
    ref = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    x = preprocess_image(img, size)
    assert x.shape == (1, 3, *size)
    assert torch.equal(x[0], ref(img.convert("RGB")))
    g = preprocess_map(gt, size)
    assert g.shape == (1, 1, *size)
    assert torch.equal(g[0], ref(gt.convert("L")))


def test_handles_other_modes():
    rgba = Image.fromarray(np.zeros((50, 60, 4), dtype=np.uint8), "RGBA")
    assert preprocess_image(rgba, (224, 224)).shape == (1, 3, 224, 224)
    assert preprocess_map(rgba, (224, 224)).shape == (1, 1, 224, 224)
