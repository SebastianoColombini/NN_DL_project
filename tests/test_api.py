import io
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import SAMPLES_DIR, WEIGHTS_DIR, need_converted

from app.server import create_app

SAMPLE = "i05june05_static_street_boston_p1010764"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(device="cpu", weights_dir=WEIGHTS_DIR))


def decode_png(data_url):
    assert data_url.startswith("data:image/png;base64,")
    import base64

    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    assert img.mode == "L"
    return np.asarray(img)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["device"]["type"] == "cpu"
    assert re.match(r"\d+\.\d+", body["versions"]["torch"])
    assert [m["id"] for m in body["models"]] == ["v1.1", "v2.0"]


def test_models(client):
    body = client.get("/api/models").json()
    assert body["default"] == "v1.1"
    m = body["models"][0]
    assert m["input_size"] == [240, 320]
    assert m["reference"]["cc"] == pytest.approx(0.6044)
    assert "builder" not in m


def test_samples_and_files(client):
    samples = client.get("/api/samples").json()["samples"]
    assert len(samples) >= 6
    s = next(s for s in samples if s["id"] == SAMPLE)
    assert s["has_gt"] and s["width"] == 1024 and s["height"] == 768
    assert client.get(s["image_url"]).status_code == 200
    assert client.get(s["gt_url"]).status_code == 200
    assert client.get("/samples/samples.json").status_code == 404
    assert client.get("/samples/..%2FREADME.md").status_code == 404


def test_index_and_static(client):
    assert "<html" in client.get("/").text.lower()
    assert client.get("/static/styles.css").status_code == 200


def test_predict_sample_both_models_with_gt(client):
    need_converted("v1.1")
    need_converted("v2.0")
    r = client.post("/api/predict", data={"sample_id": SAMPLE, "models": "v1.1,v2.0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"name": f"{SAMPLE}.jpeg", "width": 1024, "height": 768}
    assert decode_png(body["gt"]["png"]).shape == (768, 1024)
    expected = {"v1.1": (0.733, 0.640, 0.577), "v2.0": (0.646, 0.679, 0.565)}
    for res in body["results"]:
        assert decode_png(res["map_png"]).shape == (768, 1024)
        cc, kld, sim = expected[res["model"]]
        assert res["metrics"]["cc"] == pytest.approx(cc, abs=0.005)
        assert res["metrics"]["kld"] == pytest.approx(kld, abs=0.005)
        assert res["metrics"]["sim"] == pytest.approx(sim, abs=0.005)
        assert res["inference_ms"] > 0


def test_predict_upload_without_and_with_gt(client):
    need_converted("v1.1")
    img = (SAMPLES_DIR / f"{SAMPLE}.jpeg").read_bytes()
    r = client.post("/api/predict", files={"image": ("street.jpg", img, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gt"] is None and body["results"][0]["metrics"] is None
    assert body["results"][0]["model"] == "v1.1"

    gt = (SAMPLES_DIR / f"{SAMPLE}_fixMap.jpg").read_bytes()
    r = client.post(
        "/api/predict",
        files={"image": ("street.jpg", img, "image/jpeg"), "gt": ("gt.jpg", gt, "image/jpeg")},
    )
    assert r.json()["results"][0]["metrics"]["cc"] == pytest.approx(0.733, abs=0.005)


def test_predict_exif_rotation(client):
    need_converted("v1.1")
    img = Image.open(SAMPLES_DIR / f"{SAMPLE}.jpeg")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees: displayed as portrait
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    r = client.post("/api/predict", files={"image": ("phone.jpg", buf.getvalue(), "image/jpeg")})
    body = r.json()
    assert (body["image"]["width"], body["image"]["height"]) == (768, 1024)
    assert decode_png(body["results"][0]["map_png"]).shape == (1024, 768)


def test_predict_rgba_png(client):
    need_converted("v1.1")
    buf = io.BytesIO()
    Image.new("RGBA", (64, 48), (200, 10, 10, 128)).save(buf, format="PNG")
    r = client.post("/api/predict", files={"image": ("a.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert decode_png(r.json()["results"][0]["map_png"]).shape == (48, 64)


def test_bad_inputs(client):
    r = client.post("/api/predict", files={"image": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "not a supported image" in r.json()["detail"]
    assert client.post("/api/predict", data={}).status_code == 400
    assert client.post("/api/predict", data={"sample_id": "nope"}).status_code == 400
    r = client.post("/api/predict", data={"sample_id": SAMPLE, "models": "v9"})
    assert r.status_code == 400 and "Unknown model" in r.json()["detail"]
    big = b"\xff\xd8" + b"0" * (20 * 1024 * 1024 + 10)
    assert client.post("/api/predict", files={"image": ("big.jpg", big, "image/jpeg")}).status_code == 413


def test_missing_weights(tmp_path):
    c = TestClient(create_app(device="cpu", weights_dir=tmp_path))
    health = c.get("/api/health").json()
    assert not any(m["available"] for m in health["models"])
    r = c.post("/api/predict", data={"sample_id": SAMPLE, "models": "v2.0"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "saliency_v2.0.safetensors" in detail and str(tmp_path) in detail
