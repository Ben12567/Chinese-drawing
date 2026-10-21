"""Evaluation for the v7 protocol.

Differences from evaluate_experiments.py (v6):
  * ALL generated images are used (83 test samples x 3 inference seeds = 249 per run). v6 used only
    the seed-42 image of each sample ("primary prediction") for every metric except LPIPS.
  * per-image metrics are averaged over inference seeds to give one value per test sample,
    which is the unit used for paired tests;
  * FID, KID and CMMD with 95% bootstrap CIs (resampling test samples and real images);
  * style accuracy, macro-F1 and confusion matrix from a pre-trained, frozen style classifier;
  * optional sensitivity variants: seal-inpainted and centre-cropped images (both sides).
Use --real-as-predictions to evaluate the real test paintings themselves (reference row).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from clpgen.data.mounting import inpaint_seals
from clpgen.data.schema import load_manifest
from clpgen.evaluation.metrics import (
    blank_space_iou,
    blank_space_ssim,
    compute_clipscore,
    compute_hpsv2,
    compute_pickscore,
    edge_consistency_score,
    load_rgb,
)
from clpgen.evaluation.v7_metrics import (
    bootstrap_set_metric,
    cmmd_from_embeddings,
    fid_from_features,
    kid_from_features,
)

Image.MAX_IMAGE_PIXELS = None
PROMPT_FIELD_MAP = {"short": "prompt_short_zh", "structured": "prompt_structured_zh", "dense": "dense_caption_zh"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def centre_crop(arr: np.ndarray, frac: float = 0.8) -> np.ndarray:
    h, w = arr.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return arr[dh : h - dh, dw : w - dw]


TRANSFORMS = {
    "plain": None,
    "seal_inpainted": lambda a: inpaint_seals(np.ascontiguousarray(a)),
    "centre_crop": lambda a: np.ascontiguousarray(centre_crop(a)),
}


def inception_features(files: list[Path], variant: str) -> np.ndarray:
    from cleanfid import fid

    if not hasattr(inception_features, "model"):
        inception_features.model = fid.build_feature_extractor("clean", DEVICE)
    return fid.get_files_features(
        [str(f) for f in files],
        model=inception_features.model,
        num_workers=0,
        batch_size=32,
        device=DEVICE,
        mode="clean",
        custom_image_tranform=TRANSFORMS[variant],
        verbose=False,
    )


@torch.no_grad()
def clip_l336_embeddings(files: list[Path], variant: str) -> np.ndarray:
    from transformers import CLIPImageProcessor, CLIPModel

    if not hasattr(clip_l336_embeddings, "model"):
        name = "models/clip-vit-large-patch14-336"  # openai/clip-vit-large-patch14-336 @ ce19dc9
        clip_l336_embeddings.model = CLIPModel.from_pretrained(name).to(DEVICE).eval()
        clip_l336_embeddings.proc = CLIPImageProcessor.from_pretrained(name)
    model, proc = clip_l336_embeddings.model, clip_l336_embeddings.proc
    out = []
    for start in range(0, len(files), 16):
        batch = []
        for f in files[start : start + 16]:
            arr = np.asarray(Image.open(f).convert("RGB"))
            if TRANSFORMS[variant] is not None:
                arr = TRANSFORMS[variant](arr)
            batch.append(Image.fromarray(arr).resize((336, 336), Image.BICUBIC))
        inputs = proc(images=batch, return_tensors="pt", do_resize=False, do_center_crop=False).to(DEVICE)
        emb = model.get_image_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


def cached(cache: Path, name: str, files: list[Path], fn) -> np.ndarray:
    path = cache / f"{name}.npz"
    key = np.array([str(f) for f in files])
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if len(data["files"]) == len(key) and (data["files"] == key).all():
            return data["feats"]
    feats = fn(files)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, files=key, feats=feats)
    return feats


@torch.no_grad()
def lpips_per_sample(groups: dict[str, list[Path]]) -> dict[str, float]:
    import lpips

    metric = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
    result = {}
    for sid, paths in groups.items():
        if len(paths) < 2:
            continue
        ts = []
        for p in paths:
            arr = np.asarray(load_rgb(p, image_size=256)).astype(np.float32) / 127.5 - 1.0
            ts.append(torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE))
        result[sid] = float(np.mean([metric(a, b).item() for a, b in combinations(ts, 2)]))
    return result


@torch.no_grad()
def style_predictions(files: list[Path], classifier_path: str) -> np.ndarray:
    from clpgen.evaluation.metrics import _clip_image_features, _load_clip

    bundle = pickle.loads(Path(classifier_path).read_bytes())
    processor, model, device = _load_clip(bundle["clip_model"])
    feats = _clip_image_features(files, processor, model, device)
    return bundle["encoder"].inverse_transform(bundle["classifier"].predict(feats))


def main(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest_path)
    root = manifest_path.parent
    samples = {s.sample_id: s for s in load_manifest(manifest_path)}
    test = [s for s in samples.values() if s.split == args.split]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    # Generated images grouped by test sample.
    groups: dict[str, list[Path]] = defaultdict(list)
    if args.real_as_predictions:
        for s in test:
            groups[s.sample_id].append(root / s.image_path)
    else:
        for p in sorted(Path(args.prediction_dir).glob("*.png")):
            sid = p.stem.split("__seed")[0]
            if sid in samples and samples[sid].split == args.split:
                groups[sid].append(p)
    files = [p for sid in sorted(groups) for p in groups[sid]]
    file_sids = [sid for sid in sorted(groups) for _ in groups[sid]]
    if not files:
        raise SystemExit("no predictions found")

    prompt_field = PROMPT_FIELD_MAP[args.prompt_mode]
    prompts = [getattr(samples[sid], prompt_field) for sid in file_sids]
    rows = pd.DataFrame({"sample_id": file_sids, "path": [str(f) for f in files], "prompt": prompts})
    rows["clipscore"] = compute_clipscore(files, prompts)["per_image"]
    torch.cuda.empty_cache()
    rows["pickscore"] = compute_pickscore(files, prompts)["per_image"]
    torch.cuda.empty_cache()
    hps = compute_hpsv2(files, prompts)["per_image"]
    rows["hpsv2"] = hps if len(hps) == len(files) else np.nan
    torch.cuda.empty_cache()
    struct = [root / samples[sid].structure_map_path for sid in file_sids]
    rows["edge_iou"] = [edge_consistency_score(s, f) for s, f in zip(struct, files)]
    rows["blank_iou"] = [blank_space_iou(s, f) for s, f in zip(struct, files)]
    rows["blank_ssim"] = [blank_space_ssim(s, f) for s, f in zip(struct, files)]
    rows["target_style"] = [samples[sid].style_label for sid in file_sids]
    rows["pred_style"] = style_predictions(files, args.style_classifier)
    rows["style_correct"] = (rows["target_style"] == rows["pred_style"]).astype(float)
    rows.to_csv(out / "per_image_metrics.csv", index=False)

    per_sample = rows.groupby("sample_id")[["clipscore", "pickscore", "hpsv2", "edge_iou", "blank_iou", "blank_ssim", "style_correct"]].mean()
    lp = lpips_per_sample(groups)
    per_sample["lpips_diversity"] = pd.Series(lp)
    per_sample.to_csv(out / "per_sample_metrics.csv")

    # Set-level fidelity metrics.
    real_test = [root / s.image_path for s in sorted(test, key=lambda s: s.sample_id)]
    summary: dict = {"n_samples": len(groups), "n_images": len(files)}
    variants = ["plain"] + (["seal_inpainted", "centre_crop"] if args.sensitivity else [])
    for variant in variants:
        g_inc = cached(out / "cache", f"inception_{variant}", files, lambda fs: inception_features(fs, variant))
        r_inc = cached(cache, f"real_{args.split}_inception_{variant}", real_test, lambda fs: inception_features(fs, variant))
        g_clip = cached(out / "cache", f"clipL336_{variant}", files, lambda fs: clip_l336_embeddings(fs, variant))
        r_clip = cached(cache, f"real_{args.split}_clipL336_{variant}", real_test, lambda fs: clip_l336_embeddings(fs, variant))
        if args.real_as_predictions:
            # Real-vs-real floor: validation paintings against test paintings.
            val = [root / s.image_path for s in sorted(samples.values(), key=lambda s: s.sample_id) if s.split == "val"]
            g_inc = cached(cache, f"real_val_inception_{variant}", val, lambda fs: inception_features(fs, variant))
            g_clip = cached(cache, f"real_val_clipL336_{variant}", val, lambda fs: clip_l336_embeddings(fs, variant))
            gen_groups = [str(i) for i in range(len(val))]
        else:
            gen_groups = file_sids
        rng = np.random.default_rng(0)
        fid = bootstrap_set_metric(g_inc, gen_groups, r_inc, fid_from_features, n_boot=args.n_boot)
        kid = bootstrap_set_metric(g_inc, gen_groups, r_inc, lambda a, b: kid_from_features(a, b, 10, rng), n_boot=args.n_boot)
        kid_point = kid_from_features(g_inc, r_inc, 100, np.random.default_rng(0))
        cmmd = bootstrap_set_metric(g_clip, gen_groups, r_clip, cmmd_from_embeddings, n_boot=args.n_boot)
        summary[variant] = {
            "fid": fid[0], "fid_ci": fid[1:],
            "kid": kid_point, "kid_ci": kid[1:],
            "cmmd": cmmd[0], "cmmd_ci": cmmd[1:],
        }

    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    labels = sorted(set(rows["target_style"]) | set(rows["pred_style"]))
    summary["style"] = {
        "accuracy": float(rows["style_correct"].mean()),
        "macro_f1": float(f1_score(rows["target_style"], rows["pred_style"], labels=labels, average="macro", zero_division=0)),
        "per_class_recall": dict(zip(labels, recall_score(rows["target_style"], rows["pred_style"], labels=labels, average=None, zero_division=0).tolist())),
        "labels": labels,
        "confusion": confusion_matrix(rows["target_style"], rows["pred_style"], labels=labels).tolist(),
        "n_distinct_predicted": int(rows["pred_style"].nunique()),
    }
    for col in per_sample.columns:
        summary[col] = float(per_sample[col].mean())
    summary["prompt_truncated_fraction"] = float(np.mean([len(p.split()) > 60 for p in prompts]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "style"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-mode", choices=list(PROMPT_FIELD_MAP), default="structured")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--style-classifier", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--real-as-predictions", action="store_true")
    main(parser.parse_args())
