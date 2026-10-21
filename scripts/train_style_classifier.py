"""Train and freeze the style classifier used for "style accuracy" (v7).

Same design as v6 (CLIP ViT-B/32 image features + multinomial logistic regression trained on
the real training-split paintings and their metadata-derived style labels), but the classifier
is now trained once, saved, and validated on the real test paintings (upper reference), with a
majority-class baseline, macro-F1, per-class recall and the confusion matrix reported.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

from clpgen.data.schema import load_manifest
from clpgen.evaluation.metrics import _clip_image_features, _load_clip

CLIP_MODEL = "openai/clip-vit-base-patch32"


def main(manifest_path: str, out_path: str, report_path: str) -> None:
    root = Path(manifest_path).parent
    samples = load_manifest(manifest_path)
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    processor, model, device = _load_clip(CLIP_MODEL)
    x_train = _clip_image_features([root / s.image_path for s in train], processor, model, device)
    x_test = _clip_image_features([root / s.image_path for s in test], processor, model, device)
    encoder = LabelEncoder().fit([s.style_label for s in samples])
    y_train = encoder.transform([s.style_label for s in train])
    y_test = encoder.transform([s.style_label for s in test])
    clf = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    pred = clf.predict(x_test)
    labels = list(range(len(encoder.classes_)))
    majority = Counter(y_train).most_common(1)[0][0]
    report = {
        "clip_model": CLIP_MODEL,
        "classes": encoder.classes_.tolist(),
        "train_counts": {encoder.classes_[k]: int(v) for k, v in Counter(y_train).items()},
        "test_counts": {encoder.classes_[k]: int(v) for k, v in Counter(y_test).items()},
        "real_test_accuracy": float((pred == y_test).mean()),
        "real_test_macro_f1": float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)),
        "real_test_per_class_recall": dict(zip(encoder.classes_.tolist(), recall_score(y_test, pred, labels=labels, average=None, zero_division=0).tolist())),
        "real_test_confusion": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "majority_class": encoder.classes_[majority],
        "majority_baseline_accuracy_test": float((y_test == majority).mean()),
        "majority_baseline_macro_f1_test": float(f1_score(y_test, np.full_like(y_test, majority), labels=labels, average="macro", zero_division=0)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(pickle.dumps({"clip_model": CLIP_MODEL, "encoder": encoder, "classifier": clf}))
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    main(args.manifest_path, args.out, args.report)
