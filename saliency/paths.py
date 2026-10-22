"""Resource locations, working both from the repo and from a frozen bundle."""
import os
import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def weights_dir() -> Path:
    env = os.environ.get("SALIENCY_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return resource_root() / "weights"


def samples_dir() -> Path:
    return resource_root() / "samples"


def static_dir() -> Path:
    return resource_root() / "app" / "static"
