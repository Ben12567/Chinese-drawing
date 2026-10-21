"""Revised cleaning stage (v7): subject-matter verification and mounting removal.

Order of operations for every downloaded candidate:
  1. metadata object-type filter (screens, furniture, ceramics, textiles, prints, rubbings, ...)
  2. crop to the painted area (mounting borders / photographic background removed)
  3. original quality filter and perceptual-hash de-duplication (unchanged from v6)
  4. CLIP zero-shot subject verification (landscape painting vs. other object/genre classes)
  5. seal statistics (reported, not removed)
Writes a cleaned report compatible with build_landscape_paper_dataset.py, a rejection
breakdown table and contact sheets for visual audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from clpgen.data.cleaning import deduplicate_reports, evaluate_image_quality
from clpgen.data.collection import load_jsonl, save_jsonl
from clpgen.data.mounting import detect_painting_box, red_seal_mask

Image.MAX_IMAGE_PIXELS = None

NON_PAINTING_TERMS = [
    "screen", "furniture", "cabinet", "table", "chair", "lacquer", "porcelain", "ceramic", "stoneware",
    "vase", "jar", "bowl", "dish", "plate", "cup", "box", "textile", "embroider", "kesi", "tapestry",
    "silk tapestry", "rubbing", "woodblock", "print", "book", "printed", "inkstone", "ink stick", "jade",
    "bronze", "sculpture", "mirror", "plaque", "wallpaper", "map", "photograph", "costume", "robe",
    "snuff bottle", "brush pot", "brush holder", "seal paste", "carving", "enamel", "glass",
]

POSITIVE_PROMPTS = [
    "a Chinese landscape painting with mountains and water",
    "a Chinese ink landscape painting on a hanging scroll",
    "a Chinese landscape painting on a handscroll",
    "a Chinese landscape painting on an album leaf",
    "a Chinese landscape painting on a fan",
]
NEGATIVE_PROMPTS = [
    "a photograph of a folding screen",
    "a photograph of a table screen on a wooden stand",
    "a photograph of a decorative art object",
    "a brocade album cover with a title label",
    "a mounted hanging scroll photographed with its silk mounting",
    "a photograph of a ceramic vase",
    "a photograph of a lacquer box",
    "an embroidered silk textile",
    "a page of Chinese calligraphy without a picture",
    "a Chinese figure painting of people",
    "a Chinese bird-and-flower painting",
    "a Chinese painting of bamboo or plum blossoms",
    "a Chinese painting of animals",
    "a woodblock-printed book page",
    "a black-and-white stone rubbing",
    "a photograph of a sculpture",
    "a Japanese painting",
    "a European landscape drawing or etching",
    "a Western oil painting of a landscape",
    "a wooden album cover with a paper title slip",
    "a rough sketch of human figures",
]


def metadata_non_painting(row: dict) -> str:
    extra = row.get("extra") or {}
    fields = " ".join(
        str(x)
        for x in [
            row.get("title", ""),
            extra.get("object_name", ""),
            extra.get("classification", ""),
            extra.get("classification_title", ""),
            extra.get("type", ""),
            row.get("department", ""),
        ]
    ).lower()
    medium = str(row.get("medium", "")).lower()
    for term in NON_PAINTING_TERMS:
        if term in fields:
            # "fan" paintings and "album leaf" are paintings; screens/prints/etc. are not.
            return term
    if any(term in medium for term in ["lacquer", "porcelain", "woodblock", "rubbing", "embroidery", "kesi"]):
        return "medium"
    # Museum records carry an explicit culture field; require it to be Chinese. (The v6 rule matched
    # "ming"/"song" as substrings anywhere, which admitted e.g. European drawings.)
    culture = str(row.get("culture", "")).lower()
    if row.get("source") in {"met", "cma", "artic"} and not any(t in culture for t in ["china", "chinese"]):
        return "culture_not_chinese"
    return ""


def load_clip(device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_landscape_probability(paths: list[Path], device: torch.device, batch_size: int = 16) -> list[dict]:
    model, preprocess, tokenizer = load_clip(device)
    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    text = tokenizer(prompts).to(device)
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    results: list[dict] = []
    for start in range(0, len(paths), batch_size):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[start : start + batch_size]]
        images = torch.stack(batch).to(device)
        if device.type == "cuda":
            images = images.half()
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).float().softmax(dim=-1).cpu().numpy()
        for row in probs:
            pos = float(row[: len(POSITIVE_PROMPTS)].sum())
            top = int(row.argmax())
            results.append({"p_landscape": pos, "top_prompt": prompts[top], "top_prob": float(row[top])})
    return results


def contact_sheet(items: list[tuple[Path, str]], out: Path, cols: int = 8, cell: int = 220) -> None:
    if not items:
        return
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(items):
        im = Image.open(path).convert("RGB")
        im.thumbnail((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + 28)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        draw.text((x + 4, y + cell + 2), label[:34], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def main(manifest: str, crop_dir: str, report_path: str, audit_dir: str, threshold: float, device_name: str) -> None:
    rows = [r for r in load_jsonl(manifest) if r.get("download_ok") and r.get("download_path")]
    crop_root, audit_root = Path(crop_dir), Path(audit_dir)
    reports: list[dict] = []
    reasons = Counter()

    # 1-2: metadata filter and mounting crop
    for row in rows:
        sample_id = f"{row['source']}_{row['source_id']}"
        term = metadata_non_painting(row)
        if term:
            reasons["metadata_non_painting"] += 1
            reports.append({"sample_id": sample_id, "metadata": row, "keep": False, "reason": f"metadata_non_painting:{term}",
                            "image_path": row["download_path"], "quality": {}, "duplicate_of": ""})
            continue
        rgb = np.asarray(Image.open(row["download_path"]).convert("RGB"))
        crop = detect_painting_box(rgb)
        x0, y0, x1, y1 = crop.box
        dst = crop_root / row["source"] / f"{row['source_id']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb[y0:y1, x0:x1]).save(dst, quality=95)
        seal_mask, seal_count = red_seal_mask(rgb[y0:y1, x0:x1])
        reports.append({
            "sample_id": sample_id,
            "source": row["source"],
            "source_id": row["source_id"],
            "title": row.get("title", ""),
            "image_path": str(dst),
            "original_path": row["download_path"],
            "crop": crop.to_dict(),
            "seal_count": int(seal_count),
            "seal_area_fraction": float((seal_mask > 0).mean()),
            "metadata": row,
        })

    # 3: quality + dedup (unchanged v6 criteria)
    candidates = [r for r in reports if "keep" not in r]
    for r in candidates:
        r["quality"] = evaluate_image_quality(r["image_path"]).to_dict()
    deduplicate_reports(candidates)
    for r in candidates:
        flags = r["quality"]["flags"]
        if r.get("duplicate_of"):
            r["keep"], r["reason"] = False, "duplicate"
        elif any(f in {"low_resolution", "extreme_aspect_ratio"} for f in flags) or r["quality"]["quality_score"] < 0.42:
            r["keep"], r["reason"] = False, "quality_filter"
        else:
            r["keep"], r["reason"] = None, ""

    # 4: CLIP zero-shot subject verification
    pending = [r for r in candidates if r["keep"] is None]
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    scores = clip_landscape_probability([Path(r["image_path"]) for r in pending], device)
    for r, s in zip(pending, scores):
        r["subject_check"] = s
        if s["p_landscape"] >= threshold:
            r["keep"], r["reason"] = True, "pass"
        else:
            r["keep"], r["reason"] = False, "subject_not_landscape"
    for r in candidates:
        reasons[r["reason"]] += 1

    save_jsonl(reports, report_path)
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "rejection_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reason", "count"])
        for reason, count in sorted(reasons.items()):
            writer.writerow([reason, count])
    kept = [r for r in reports if r.get("keep")]
    by_source = Counter(r["source"] for r in kept)
    summary = {
        "downloaded": len(rows),
        "kept": len(kept),
        "kept_by_source": dict(by_source),
        "reasons": dict(reasons),
        "cropped_fraction_of_kept": float(np.mean([r["crop"]["cropped"] for r in kept])) if kept else 0.0,
        "seal_prevalence_of_kept": float(np.mean([r["seal_count"] > 0 for r in kept])) if kept else 0.0,
        "mean_seal_area_fraction_of_kept": float(np.mean([r["seal_area_fraction"] for r in kept])) if kept else 0.0,
    }
    (audit_root / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    subject_rejects = [r for r in candidates if r["reason"] == "subject_not_landscape"]
    contact_sheet([(Path(r["image_path"]), f"{r['sample_id']} p={r['subject_check']['p_landscape']:.2f}") for r in subject_rejects],
                  audit_root / "rejected_subject.jpg")
    borderline = sorted([r for r in kept], key=lambda r: r["subject_check"]["p_landscape"])[:96]
    contact_sheet([(Path(r["image_path"]), f"{r['sample_id']} p={r['subject_check']['p_landscape']:.2f}") for r in borderline],
                  audit_root / "kept_lowest_subject_score.jpg")
    cropped = [r for r in kept if r["crop"]["cropped"]]
    for page in range(0, len(cropped), 48):
        contact_sheet([(Path(r["original_path"]), r["sample_id"] + " orig") for r in cropped[page : page + 24]]
                      + [(Path(r["image_path"]), r["sample_id"] + " crop") for r in cropped[page : page + 24]],
                      audit_root / f"crops_{page // 48:02d}.jpg")
    for page in range(0, len(kept), 96):
        contact_sheet([(Path(r["image_path"]), r["sample_id"]) for r in kept[page : page + 96]],
                      audit_root / f"kept_{page // 96:02d}.jpg", cols=12, cell=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--crop-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.manifest, args.crop_dir, args.report_path, args.audit_dir, args.threshold, args.device)
