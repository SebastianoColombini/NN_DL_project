import json

import pytest

from saliency.registry import DEFAULT_MODEL, REGISTRY, ModelSpec, get_spec


def test_ids_and_default():
    assert list(REGISTRY) == ["v1.1", "v2.0"]
    assert DEFAULT_MODEL == "v1.1"
    assert get_spec("v2.0") is REGISTRY["v2.0"]
    with pytest.raises(KeyError):
        get_spec("v9")


def test_static_fields():
    a, b = REGISTRY["v1.1"], REGISTRY["v2.0"]
    assert a.input_size == (240, 320) and b.input_size == (224, 224)
    assert a.weights_file == "saliency_v1.1.safetensors"
    assert b.weights_file == "saliency_v2.0.safetensors"
    assert a.label == "v1.1 · CNN encoder" and b.label == "v2.0 · Swin-T encoder"
    assert a.reference == {"dataset": "MIT1003", "total_loss": 7.6305, "kld": 0.9229, "cc": 0.6044, "sim": 0.4809}
    assert b.reference["total_loss"] == 7.5088


def test_training_info():
    t = REGISTRY["v1.1"].training
    assert (t["best_epoch"], t["epochs_run"], t["best_val_loss"]) == (14, 29, 3.0735)
    assert len(t["train_loss"]) == len(t["val_loss"]) == 29
    assert min(t["val_loss"]) == t["val_loss"][t["best_epoch"] - 1]
    t2 = REGISTRY["v2.0"].training
    assert (t2["best_epoch"], t2["epochs_run"]) == (127, 130)
    assert len(t2["train_loss"]) == 130
    assert t2["hparams"]["backbone"] == "swin_tiny_patch4_window7_224"


@pytest.mark.parametrize("model_id", ["v1.1", "v2.0"])
def test_params_match_builder(model_id):
    spec = REGISTRY[model_id]
    model = spec.builder()
    assert sum(p.numel() for p in model.parameters()) == spec.params


def test_to_json():
    d = REGISTRY["v1.1"].to_json()
    assert "builder" not in d
    assert d["input_size"] == [240, 320]
    assert set(d) == {f for f in ModelSpec.__dataclass_fields__ if f != "builder"}
    json.dumps([s.to_json() for s in REGISTRY.values()])
