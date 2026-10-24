import sys
from pathlib import Path

from saliency import paths

from conftest import REPO


def test_default_locations(monkeypatch):
    monkeypatch.delenv("SALIENCY_WEIGHTS_DIR", raising=False)
    assert paths.resource_root() == REPO
    assert paths.weights_dir() == REPO / "weights"
    assert paths.samples_dir() == REPO / "samples"
    assert paths.static_dir() == REPO / "app" / "static"


def test_weights_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SALIENCY_WEIGHTS_DIR", str(tmp_path))
    assert paths.weights_dir() == tmp_path.resolve()


def test_frozen_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv("SALIENCY_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.resource_root() == Path(tmp_path)
    assert paths.weights_dir() == Path(tmp_path) / "weights"
