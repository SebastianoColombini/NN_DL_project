import threading

import numpy as np
import pytest
import torch
from PIL import Image

from saliency.inference import Prediction, Predictor, device_info, pick_device

from conftest import MODEL_IDS, SAMPLES_DIR, WEIGHTS_DIR, need_converted

# Notebook preprocessing, single image: (cc, kld, sim)
REFERENCE = {
    "v1.1": {
        "i05june05_static_street_boston_p1010764": (0.733, 0.640, 0.577),
        "i05june05_static_street_boston_p1010785": (0.651, 0.722, 0.558),
    },
    "v2.0": {
        "i05june05_static_street_boston_p1010764": (0.646, 0.679, 0.565),
        "i05june05_static_street_boston_p1010785": (0.703, 0.664, 0.565),
    },
}


@pytest.fixture(scope="module")
def predictor():
    return Predictor("cpu", weights_dir=WEIGHTS_DIR)


def test_device_helpers():
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device("auto").type in {"cpu", "cuda", "mps"}
    assert device_info("cpu") == {"type": "cpu", "name": None, "vram_total_mb": None, "vram_free_mb": None}


def test_available(tmp_path):
    p = Predictor("cpu", weights_dir=tmp_path)
    assert p.available("v1.1") is False
    with pytest.raises(KeyError):
        p.available("nope")


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_reference_values(predictor, model_id):
    need_converted(model_id)
    for stem, (cc, kld, sim) in REFERENCE[model_id].items():
        img = Image.open(SAMPLES_DIR / f"{stem}.jpeg")
        pred = predictor.predict(img, model_id)
        assert isinstance(pred, Prediction)
        assert pred.map_model.shape == pred.input_size and pred.map_model.dtype == np.float32
        assert pred.map_full.shape == (img.height, img.width) and pred.map_full.dtype == np.float32
        assert 0 <= pred.map_full.min() and pred.map_full.max() <= 1
        assert pred.tensor.shape == (1, 1, *pred.input_size)
        assert pred.inference_ms > 0
        m = predictor.metrics(pred, Image.open(SAMPLES_DIR / f"{stem}_fixMap.jpg"))
        assert m["cc"] == pytest.approx(cc, abs=0.005)
        assert m["kld"] == pytest.approx(kld, abs=0.005)
        assert m["sim"] == pytest.approx(sim, abs=0.005)


def test_model_cache_thread_safe(predictor):
    need_converted("v1.1")
    got = []
    threads = [threading.Thread(target=lambda: got.append(predictor.get_model("v1.1"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 4 and all(m is got[0] for m in got)
