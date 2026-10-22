"""Convert the per-module ``*_best.pth`` checkpoints into one safetensors file per model.

Usage:
    python tools/convert_weights.py --src /path/to/weights_root --out weights/

``<src>/<version>/`` must contain encoder/decoder3/decoder4/decoder5/final ``_best.pth``
(e.g. ``<src>/1.1/encoder_best.pth``). Float tensors with ndim >= 2 are stored in fp16,
1-D tensors and BatchNorm running statistics stay fp32, integer buffers are unchanged.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saliency.models import SUBMODULES  # noqa: E402
from saliency.registry import REGISTRY  # noqa: E402
from saliency.weights import load_original_pth  # noqa: E402

DTYPE_POLICY = "fp16 for float tensors with ndim>=2; fp32 for 1-D tensors and BN running stats; int unchanged"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_storage(model) -> dict:
    stats = {name for name, _ in model.named_buffers() if name.endswith(("running_mean", "running_var"))}
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().contiguous()
        if v.is_floating_point():
            if v.ndim >= 2 and k not in stats:
                if v.abs().max() > 65504:
                    raise ValueError(f"{k} overflows fp16")
                v = v.half()
            else:
                v = v.float()
        out[k] = v
    return out


def convert(spec, src_root: Path, out_dir: Path) -> dict:
    version = spec.id.lstrip("v")
    run_dir = src_root / version
    model = load_original_pth(spec, run_dir)
    sources = {f"{name}_best.pth": sha256(run_dir / f"{name}_best.pth") for name in SUBMODULES}
    tensors = to_storage(model)
    meta = {
        "version": spec.id,
        "input_size": f"{spec.input_size[0]}x{spec.input_size[1]}",
        "converted": dt.date.today().isoformat(),
        "dtype_policy": DTYPE_POLICY,
        **{f"sha256:{name}": digest for name, digest in sources.items()},
    }
    out = out_dir / spec.weights_file
    save_file(tensors, str(out), metadata=meta)
    n16 = sum(1 for v in tensors.values() if v.dtype == torch.float16)
    print(f"{spec.id}: {out} ({out.stat().st_size / 2**20:.1f} MiB, {len(tensors)} tensors, {n16} fp16)")
    return {
        "file": spec.weights_file,
        "version": spec.id,
        "sha256": sha256(out),
        "size_bytes": out.stat().st_size,
        "dtype_policy": DTYPE_POLICY,
        "sources": [{"file": f"{version}/{name}", "sha256": digest} for name, digest in sources.items()],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path, help="folder with <version>/*_best.pth")
    ap.add_argument("--out", default=Path("weights"), type=Path, help="output folder")
    ap.add_argument("--models", default=",".join(REGISTRY), help="comma separated ids (default: all)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "MANIFEST.json"
    manifest = {"files": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    entries = {e["version"]: e for e in manifest.get("files", [])}
    for model_id in args.models.split(","):
        entries[model_id] = convert(REGISTRY[model_id.strip()], args.src, args.out)
    manifest["files"] = [entries[k] for k in REGISTRY if k in entries]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
