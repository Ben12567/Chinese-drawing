"""Colour-treatment style metric (v8).

The v7 style classifier (five labels) is degenerate: in the training split the labels are
705 ink-and-colour / 262 monochrome / 7 light-colour / 5 Mi / 3 Dong Yuan, and on the real test
paintings it reaches macro-F1 0.23 by predicting the majority class almost always, so every
method obtained the majority-class accuracy. Here the labels are collapsed to the distinction the
data can support, monochrome ink vs. coloured (light colour or ink and colour), taken from the
VLM colour field of each painting (style label when the VLM response was not parsed; Mi -> mono-
chrome). The classifier (CLIP ViT-B/32 features + logistic regression) uses balanced class
weights and is reported with balanced accuracy and macro-F1 on the real test paintings.

  python scripts/style_v8.py train
  python scripts/style_v8.py score --runs ours_s42 lora_only_s42 ...
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from clpgen.data.schema import load_manifest
from clpgen.evaluation.metrics import _clip_image_features, _load_clip

CLIP_MODEL = "openai/clip-vit-base-patch32"
MANIFEST = Path("data/processed/clp_v7/manifest.jsonl")
CLASSIFIER = Path("outputs/revision/style_classifier_v8.pkl")
MONO_LABELS = {"ink_monochrome", "mi_style"}


def colour_group(sample) -> str:
    colour = (sample.extra.get("prompt_fields") or {}).get("colour")
    if colour:
        return "monochrome" if colour == "monochrome ink" else "coloured"
    return "monochrome" if sample.style_label in MONO_LABELS else "coloured"


def features(paths: list[Path], cache: Path | None) -> np.ndarray:
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["paths"]) == [str(p) for p in paths]:
            return data["x"]
    processor, model, device = _load_clip(CLIP_MODEL)
    x = _clip_image_features(paths, processor, model, device)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x, paths=np.array([str(p) for p in paths]))
    return x


def train() -> None:
    root = MANIFEST.parent
    samples = load_manifest(MANIFEST)
    tr = [s for s in samples if s.split == "train"]
    te = [s for s in samples if s.split == "test"]
    x_tr = features([root / s.image_path for s in tr], Path("outputs/revision/cache/real_train_clipB32.npz"))
    x_te = features([root / s.image_path for s in te], Path("outputs/revision/cache/real_test_clipB32.npz"))
    y_tr = np.array([colour_group(s) for s in tr])
    y_te = np.array([colour_group(s) for s in te])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    labels = ["coloured", "monochrome"]
    majority = Counter(y_tr).most_common(1)[0][0]
    report = {
        "clip_model": CLIP_MODEL,
        "train_counts": dict(Counter(y_tr)),
        "test_counts": dict(Counter(y_te)),
        "real_test_balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "real_test_macro_f1": float(f1_score(y_te, pred, labels=labels, average="macro")),
        "real_test_accuracy": float((pred == y_te).mean()),
        "real_test_confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "labels": labels,
        "majority_class": str(majority),
        "majority_baseline_balanced_accuracy": 0.5,
        "majority_baseline_macro_f1": float(f1_score(y_te, np.full_like(y_te, majority), labels=labels, average="macro")),
    }
    CLASSIFIER.write_bytes(pickle.dumps({"clip_model": CLIP_MODEL, "classifier": clf}))
    Path("outputs/revision/style_classifier_v8_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def score(runs: list[str], gen_root: Path, eval_root: Path, split: str) -> None:
    root = MANIFEST.parent
    samples = {s.sample_id: s for s in load_manifest(MANIFEST)}
    clf = pickle.loads(CLASSIFIER.read_bytes())["classifier"]
    for run in runs:
        out = eval_root / run
        if (out / "style_v8_summary.json").exists():
            continue
        if run == "real_test":
            paths = [root / s.image_path for s in sorted(samples.values(), key=lambda s: s.sample_id) if s.split == split]
            sids = [s.sample_id for s in sorted(samples.values(), key=lambda s: s.sample_id) if s.split == split]
        else:
            paths = sorted((gen_root / run).glob("*.png"))
            sids = [p.stem.split("__seed")[0] for p in paths]
            keep = [i for i, sid in enumerate(sids) if sid in samples and samples[sid].split == split]
            paths, sids = [paths[i] for i in keep], [sids[i] for i in keep]
        x = features(paths, out / "cache" / "clipB32.npz")
        df = pd.DataFrame({
            "sample_id": sids,
            "target": [colour_group(samples[s]) for s in sids],
            "reference": [colour_group(samples[samples[s].extra["style_reference_id"]]) for s in sids],
            "pred": clf.predict(x),
            "p_monochrome": clf.predict_proba(x)[:, list(clf.classes_).index("monochrome")],
        })
        df["correct"] = (df.pred == df.target).astype(float)
        df.to_csv(out / "style_v8_per_image.csv", index=False)
        summary = {
            "balanced_accuracy": float(balanced_accuracy_score(df.target, df.pred)),
            "macro_f1": float(f1_score(df.target, df.pred, labels=["coloured", "monochrome"], average="macro")),
            "accuracy": float(df.correct.mean()),
            "reference_agreement": float((df.pred == df.reference).mean()),
            "p_monochrome_by_target": df.groupby("target").p_monochrome.mean().to_dict(),
            "n_images": int(len(df)),
        }
        (out / "style_v8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(run, json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "score"])
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--gen-root", default="outputs/revision/gen")
    parser.add_argument("--eval-root", default="outputs/revision/eval")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    if args.action == "train":
        train()
    else:
        score(args.runs, Path(args.gen_root), Path(args.eval_root), args.split)
