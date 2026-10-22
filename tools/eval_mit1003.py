"""Evaluate the models on MIT1003, original checkpoints vs converted safetensors.

Usage:
    python tools/eval_mit1003.py --model all --mit-dir work/mit --pth-dir work/weights \
        --weights-dir weights --out-dir eval_out [--limit N] [--batch 4]

``--mit-dir`` contains ALLSTIMULI/ and ALLFIXATIONMAPS/ (``<stem>_fixMap.jpg``).
Per-image values are computed in small batches and then aggregated like the notebooks'
evaluation: mean over consecutive groups of ``batch_size`` images (16 for v1.1, 128 for
v2.0, the training batch size), then mean over groups (the last group may be smaller).
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saliency.inference import preprocess_image, preprocess_map  # noqa: E402
from saliency.metrics import cc_loss, kld_loss, sim_loss, total_loss  # noqa: E402
from saliency.registry import REGISTRY  # noqa: E402
from saliency.weights import load_model, load_original_pth  # noqa: E402

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
METRICS = ("total", "kld", "cc", "sim")


def list_pairs(mit_dir: Path):
    img_dir, map_dir = mit_dir / "ALLSTIMULI", mit_dir / "ALLFIXATIONMAPS"
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(VALID_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        gt = map_dir / f"{stem}_fixMap.jpg"
        if not gt.is_file():
            raise FileNotFoundError(f"missing fixation map for {f}: {gt}")
        pairs.append((stem, img_dir / f, gt))
    return pairs


def per_image_metrics(model, x, gt):
    """Notebook metrics for each image of the batch (lists of floats) and the final maps."""
    final, s3, s4, s5 = model.forward_all(x)
    rows = []
    for i in range(x.shape[0]):
        sl = slice(i, i + 1)
        g = gt[sl]
        rows.append({
            "total": float(total_loss(final[sl], s3[sl], s4[sl], s5[sl], g)),
            "kld": float(kld_loss(final[sl], g)),
            "cc": float(1 - cc_loss(final[sl], g)),
            "sim": float(1 - sim_loss(final[sl], g)),
        })
    return rows, final


def grouped_mean(values, group):
    groups = [values[i:i + group] for i in range(0, len(values), group)]
    return sum(sum(g) / len(g) for g in groups) / len(groups)


def evaluate(spec, args):
    pairs = list_pairs(args.mit_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    group = args.group or spec.training["hparams"]["batch_size"]

    orig = load_original_pth(spec, args.pth_dir)
    conv = load_model(spec, "cpu", args.weights_dir)

    rows = []
    t0 = time.perf_counter()
    for start in range(0, len(pairs), args.batch):
        chunk = pairs[start:start + args.batch]
        x = torch.cat([preprocess_image(Image.open(p), spec.input_size) for _, p, _ in chunk])
        gt = torch.cat([preprocess_map(Image.open(g), spec.input_size) for _, _, g in chunk])
        with torch.inference_mode():
            r_orig, m_orig = per_image_metrics(orig, x, gt)
            r_conv, m_conv = per_image_metrics(conv, x, gt)
            diff = (m_orig - m_conv).abs().flatten(1).max(dim=1).values.tolist()
        for (stem, _, _), ro, rc, d in zip(chunk, r_orig, r_conv, diff):
            row = {"stem": stem}
            row.update({f"{k}_orig": ro[k] for k in METRICS})
            row.update({f"{k}_conv": rc[k] for k in METRICS})
            row["max_abs_diff"] = d
            rows.append(row)
        done = len(rows)
        if done % (args.batch * 25) == 0 or done == len(pairs):
            el = time.perf_counter() - t0
            print(f"  {spec.id}: {done}/{len(pairs)} images, {el:.0f}s ({el / done:.2f}s/img for both models)",
                  flush=True)

    summary = {"model": spec.id, "images": len(rows), "group": group,
               "seconds": round(time.perf_counter() - t0, 1),
               "notebook": {"total": spec.reference["total_loss"], "kld": spec.reference["kld"],
                            "cc": spec.reference["cc"], "sim": spec.reference["sim"]},
               "orig": {}, "conv": {},
               "max_abs_pixel_diff": max(r["max_abs_diff"] for r in rows)}
    for k in METRICS:
        summary["orig"][k] = grouped_mean([r[f"{k}_orig"] for r in rows], group)
        summary["conv"][k] = grouped_mean([r[f"{k}_conv"] for r in rows], group)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    csv_path = args.out_dir / f"mit1003_{spec.id}{suffix}_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {csv_path}")

    del orig, conv
    gc.collect()
    return summary


def table(summaries) -> str:
    names = {"total": "Total loss", "kld": "KLD", "cc": "CC", "sim": "SIM"}
    lines = ["| Model | Metric | Notebook | Reproduced (orig .pth) | Converted (safetensors) "
             "| \\|orig - notebook\\| | \\|conv - orig\\| |",
             "|---|---|---|---|---|---|---|"]
    for s in summaries:
        for k in METRICS:
            nb, o, c = s["notebook"][k], s["orig"][k], s["conv"][k]
            lines.append(f"| {s['model']} | {names[k]} | {nb:.4f} | {o:.4f} | {c:.4f} "
                         f"| {abs(o - nb):.4f} | {abs(c - o):.6f} |")
    lines.append("")
    for s in summaries:
        lines.append(f"- {s['model']}: {s['images']} images, groups of {s['group']}, "
                     f"max per-pixel |orig - conv| = {s['max_abs_pixel_diff']:.6f}, {s['seconds']:.0f}s")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="all", help="v1.1, v2.0 or all")
    ap.add_argument("--mit-dir", required=True, type=Path)
    ap.add_argument("--pth-dir", required=True, type=Path, help="folder with <version>/*_best.pth")
    ap.add_argument("--weights-dir", default=Path("weights"), type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--batch", type=int, default=4, help="images per forward pass")
    ap.add_argument("--group", type=int, default=0,
                    help="aggregation group size (default: the model's training batch size)")
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    ids = list(REGISTRY) if args.model == "all" else [m.strip() for m in args.model.split(",")]
    summaries = []
    for model_id in ids:
        print(f"Evaluating {model_id} ...", flush=True)
        summaries.append(evaluate(REGISTRY[model_id], args))

    md = table(summaries)
    print()
    print(md)
    tag = "_".join(s["model"] for s in summaries) + (f"_limit{args.limit}" if args.limit else "")
    (args.out_dir / f"mit1003_{tag}.md").write_text(md)
    (args.out_dir / f"mit1003_{tag}.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
