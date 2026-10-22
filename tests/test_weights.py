import pytest
import torch
from PIL import Image
from safetensors import safe_open

from saliency.inference import preprocess_image
from saliency.registry import REGISTRY
from saliency.weights import WeightsNotFoundError, load_model, weights_path

from conftest import MODEL_IDS, SAMPLES_DIR, WEIGHTS_DIR, need_converted

STREETS = ["i05june05_static_street_boston_p1010764", "i05june05_static_street_boston_p1010785"]


def test_missing_weights_message(tmp_path):
    spec = REGISTRY["v1.1"]
    with pytest.raises(WeightsNotFoundError) as e:
        load_model(spec, weights_dir=tmp_path / "nowhere")
    msg = str(e.value)
    assert "saliency_v1.1.safetensors" in msg and str(tmp_path / "nowhere") in msg
    assert "SALIENCY_WEIGHTS_DIR" in msg
    assert isinstance(e.value, FileNotFoundError)


def test_weights_path_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SALIENCY_WEIGHTS_DIR", str(tmp_path))
    assert weights_path(REGISTRY["v2.0"]) == tmp_path.resolve() / "saliency_v2.0.safetensors"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_original_strict_load(original_model, model_id):
    model = original_model(model_id)
    assert not model.training
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_converted_file_layout(model_id):
    need_converted(model_id)
    path = WEIGHTS_DIR / REGISTRY[model_id].weights_file
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        assert meta["version"] == model_id
        assert any(k.startswith("sha256:") for k in meta)
        for k in f.keys():
            assert k.split(".")[0] in {"encoder", "decoder3", "decoder4", "decoder5", "final"}
            t = f.get_tensor(k)
            if t.is_floating_point():
                expected = torch.float16 if t.ndim >= 2 else torch.float32
                assert t.dtype == expected, k


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_converted_state_dict_close(original_model, converted_model, model_id):
    a = original_model(model_id).state_dict()
    b = converted_model(model_id).state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        if a[k].is_floating_point():
            # fp16 round-trip: relative error <= 2^-11, plus subnormal range
            torch.testing.assert_close(b[k], a[k], rtol=1e-3, atol=1e-6, msg=k)
        else:
            assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_converted_outputs_close(original_model, converted_model, model_id):
    size = REGISTRY[model_id].input_size
    x = torch.cat([preprocess_image(Image.open(SAMPLES_DIR / f"{s}.jpeg"), size) for s in STREETS])
    with torch.inference_mode():
        ya = original_model(model_id)(x)
        yb = converted_model(model_id)(x)
    assert float((ya - yb).abs().max()) < 0.01
