"""Structure- and style-controllability experiments (v7), with randomly sampled conditions.

Structure: for K randomly drawn test paintings P, keep P's prompt and style reference fixed and
swap in the structure maps of M other randomly drawn test paintings Q. Each output is scored
against the map it received (edge IoU, blank IoU) and against the M-1 maps it did not receive;
the difference ("specificity") measures how much the output follows the structure input when
the text is fixed.

Style: for K randomly drawn test paintings, keep prompt and structure fixed and use one random
training painting from each of the three largest style groups as reference. Outputs are scored
by the frozen style classifier (agreement with the reference's group) and by colour statistics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.data.schema import load_manifest
from clpgen.evaluation.metrics import _clip_image_features, _load_clip, blank_space_iou, edge_consistency_score
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline

STYLE_GROUPS = ["ink_monochrome", "ink_and_colour", "light_colour"]


def load_pipeline(config_path: str, ckpt: str) -> LandscapeGenerationPipeline:
    config = ProjectConfig.from_yaml(config_path)
    config.raw["model"]["lora"]["weights_path"] = ckpt
    adapter = Path(ckpt) / "structure_adapter.pt"
    if adapter.exists():
        config.raw["model"]["hierarchical_adapter"]["weights_path"] = str(adapter)
    branch = Path(ckpt) / "structure_branch.pt"
    if branch.exists():
        config.raw["model"].setdefault("structure_branch", {})["weights_path"] = str(branch)
    return LandscapeGenerationPipeline(config)


def colour_stats(path: Path) -> dict[str, float]:
    hsv = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2HSV).astype(np.float32)
    return {"mean_saturation": float(hsv[..., 1].mean() / 255.0), "mean_value": float(hsv[..., 2].mean() / 255.0)}


def main(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest_path)
    root = manifest.parent
    samples = load_manifest(manifest)
    test = sorted([s for s in samples if s.split == "test"], key=lambda s: s.sample_id)
    train = sorted([s for s in samples if s.split == "train"], key=lambda s: s.sample_id)
    rng = random.Random(args.sampling_seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.config, args.checkpoint_dir)
    # Baselines without a style reference (e.g. LoRA+ControlNet) get no reference image, as in generate_v7.py.
    use_style = pipe.config.get("model", "style_reference", "enabled", default=True)

    def gen(prompt_sample, structure_sample, reference_path: Path, path: Path) -> None:
        if path.exists():
            return
        image = pipe.generate(GenerationRequest(
            prompt=prompt_sample.prompt_structured_zh,
            structure_map=Image.open(root / structure_sample.structure_map_path).convert("RGBA"),
            style_reference=Image.open(reference_path).convert("RGB") if use_style else None,
            seed=zlib.crc32(f"{prompt_sample.sample_id}|{structure_sample.sample_id}|{reference_path.name}".encode()) % (2**31 - 1),
        ))
        image.save(path)

    # Structure experiment.
    rows = []
    prompt_samples = rng.sample(test, args.k)
    for p in prompt_samples:
        others = rng.sample([s for s in test if s.sample_id != p.sample_id], args.m)
        for q in others:
            path = out / "structure" / f"P_{p.sample_id}__S_{q.sample_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            gen(p, q, root / p.style_reference_path, path)
            row = {"prompt_sample": p.sample_id, "structure_sample": q.sample_id, "path": str(path)}
            row["edge_iou_used"] = edge_consistency_score(root / q.structure_map_path, path)
            row["blank_iou_used"] = blank_space_iou(root / q.structure_map_path, path)
            unused = [o for o in others if o.sample_id != q.sample_id]
            row["edge_iou_unused"] = float(np.mean([edge_consistency_score(root / o.structure_map_path, path) for o in unused]))
            row["blank_iou_unused"] = float(np.mean([blank_space_iou(root / o.structure_map_path, path) for o in unused]))
            rows.append(row)
    structure_df = pd.DataFrame(rows)
    structure_df.to_csv(out / "structure_controllability.csv", index=False)
    structure_summary = {
        "n": int(len(structure_df)),
        "edge_iou_used": float(structure_df.edge_iou_used.mean()),
        "edge_iou_unused": float(structure_df.edge_iou_unused.mean()),
        "blank_iou_used": float(structure_df.blank_iou_used.mean()),
        "blank_iou_unused": float(structure_df.blank_iou_unused.mean()),
        "edge_used_gt_unused_fraction": float((structure_df.edge_iou_used > structure_df.edge_iou_unused).mean()),
        "blank_used_gt_unused_fraction": float((structure_df.blank_iou_used > structure_df.blank_iou_unused).mean()),
    }
    if not use_style:
        # The style experiment varies only the reference image, which this method does not take.
        summary = {"structure": structure_summary, "style": None}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    # Style experiment.
    bundle = pickle.loads(Path(args.style_classifier).read_bytes())
    rows = []
    for p in rng.sample(test, args.k):
        for group in STYLE_GROUPS:
            pool = [s for s in train if s.style_label == group]
            ref = rng.choice(pool)
            path = out / "style" / f"P_{p.sample_id}__R_{ref.sample_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            gen(p, p, root / ref.image_path, path)
            rows.append({"prompt_sample": p.sample_id, "reference_sample": ref.sample_id, "reference_group": group,
                         "target_group": p.style_label, "path": str(path), **colour_stats(path),
                         **{f"ref_{k}": v for k, v in colour_stats(root / ref.image_path).items()}})
    style_df = pd.DataFrame(rows)
    processor, model, device = _load_clip(bundle["clip_model"])
    feats = _clip_image_features([Path(p) for p in style_df["path"]], processor, model, device)
    style_df["predicted_group"] = bundle["encoder"].inverse_transform(bundle["classifier"].predict(feats))
    style_df.to_csv(out / "style_controllability.csv", index=False)

    summary = {
        "structure": structure_summary,
        "style": {
            "n": int(len(style_df)),
            "prediction_matches_reference_group": float((style_df.predicted_group == style_df.reference_group).mean()),
            "prediction_matches_target_group": float((style_df.predicted_group == style_df.target_group).mean()),
            "mean_saturation_by_reference_group": style_df.groupby("reference_group").mean_saturation.mean().to_dict(),
            "reference_saturation_by_group": style_df.groupby("reference_group").ref_mean_saturation.mean().to_dict(),
            "saturation_corr_output_reference": float(np.corrcoef(style_df.mean_saturation, style_df.ref_mean_saturation)[0, 1]),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/revision/ours.yaml")
    parser.add_argument("--checkpoint-dir", default="outputs/revision/train/ours_s42/checkpoint-500")
    parser.add_argument("--manifest-path", default="data/processed/clp_v7/manifest.jsonl")
    parser.add_argument("--style-classifier", default="outputs/revision/style_classifier.pkl")
    parser.add_argument("--output-dir", default="outputs/revision/controllability")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    main(parser.parse_args())
