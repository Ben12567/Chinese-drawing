"""Build the v7 benchmark manifest from the v7 cleaning report and VLM prompt fields.

Differences from build_landscape_paper_dataset.py (v6):
  * images are the mounting-cropped versions produced by verify_and_crop.py;
  * structured prompts use per-image VLM fields (nine-field template, see prompts.build_v7_prompts);
  * style labels use the same metadata keyword rules as v6, renamed to describe what they encode
    (medium-based groups rather than art-historical styles);
  * every sample's style reference is drawn from the TRAINING split (same style label), so no
    test-split painting is ever given to a model as input.
The painter-grouped 80/10/10 split (seed 42) is unchanged.
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from clpgen.data.collection import load_jsonl
from clpgen.data.prompts import build_v7_prompts
from extract_prompt_fields import COLOUR_CLASSES
from clpgen.data.schema import PaintingSample, save_manifest
from clpgen.data.structure_maps import extract_structure_maps, load_rgb, save_structure_map

# Same keyword rules and order as v6 (scripts/build_landscape_paper_dataset.py::STYLE_KEYWORDS);
# only the label names change: blue_green_landscape -> ink_and_colour,
# light_reddish_landscape -> light_colour, ink_wash_freehand -> ink_monochrome,
# mi_style_cloud_mountains -> mi_style.
STYLE_RULES = [
    ("mi fu", "mi_style"),
    ("mi youren", "mi_style"),
    ("dong yuan", "dong_yuan_style"),
    ("light color", "light_colour"),
    ("slight color", "light_colour"),
    ("ink and color", "ink_and_colour"),
    ("color on silk", "ink_and_colour"),
    ("color on paper", "ink_and_colour"),
    ("ink on silk", "ink_monochrome"),
    ("ink on paper", "ink_monochrome"),
]
DEFAULT_STYLE = "ink_monochrome"


def infer_style(meta: dict) -> tuple[str, str]:
    text = " ".join(
        [str(meta.get("title", "")), str(meta.get("artist", "")), str(meta.get("medium", "")), " ".join(meta.get("tags", []) or [])]
    ).lower()
    for keyword, style in STYLE_RULES:
        if keyword in text:
            return style, keyword
    return DEFAULT_STYLE, "default"


def assign_split(groups: list[tuple[str, list[dict]]], seed: int = 42) -> dict[str, str]:
    from build_landscape_paper_dataset import assign_split as v6_assign_split  # identical grouping logic

    return v6_assign_split(groups, seed=seed)


def fallback_fields(meta: dict) -> dict[str, str]:
    from build_landscape_paper_dataset import infer_prompt_fields

    f = infer_prompt_fields(meta)
    return {
        "subject": "landscape with mountains and water",
        "composition": "layered landscape composition",
        "foreground": ", ".join(f["foreground"]),
        "midground": ", ".join(f["midground"]),
        "background": ", ".join(f["background"]),
        "blank_space": f["blankness"],
        "brush_ink": f"{f['ink_tone']}, {f['palette']}",
        "atmosphere": f"{f['season']}, {f['weather']}, {f['mood']}",
    }


def v6_template_prompt(meta: dict) -> str:
    """The v6 structured prompt for this painting (keyword rules + defaults); used only in the
    LoRA-only diagnostic that reproduces the v6 evaluation conditions."""
    from build_landscape_paper_dataset import infer_prompt_fields
    from clpgen.data.prompts import build_prompt_bundle

    return build_prompt_bundle(**infer_prompt_fields(meta)).structured_zh


def assign_train_style_references(samples: list[PaintingSample]) -> None:
    train_by_style: dict[str, list[PaintingSample]] = defaultdict(list)
    for s in sorted(samples, key=lambda s: s.sample_id):
        if s.split == "train":
            train_by_style[s.style_label].append(s)
    all_train = [s for s in sorted(samples, key=lambda s: s.sample_id) if s.split == "train"]
    for s in samples:
        pool = [c for c in train_by_style.get(s.style_label, []) if c.sample_id != s.sample_id] or [
            c for c in all_train if c.sample_id != s.sample_id
        ]
        choice = pool[zlib.crc32(s.sample_id.encode()) % len(pool)]
        s.style_reference_path = choice.image_path
        s.extra["style_reference_id"] = choice.sample_id


def colour_agreement(samples: list[PaintingSample]) -> dict:
    """Agreement between VLM colour class and the museum-medium keyword class, on records whose
    medium field matched an explicit keyword (i.e. not the default rule)."""
    from sklearn.metrics import cohen_kappa_score

    pairs = [(s.extra["style_label_metadata"], s.style_label) for s in samples
             if s.extra.get("style_label_source") == "vlm_colour" and s.extra.get("style_rule") not in ("default",)]
    if len(pairs) < 2:
        return {"n": len(pairs)}
    a, b = zip(*pairs)
    return {"n": len(pairs), "agreement": float(np.mean([x == y for x, y in pairs])), "cohen_kappa": float(cohen_kappa_score(a, b))}


def main(report: str, fields_path: str, output_dir: str, target_size: int, seed: int) -> None:
    rows = [r for r in load_jsonl(report) if r.get("keep")]
    vlm = {r["sample_id"]: r for r in load_jsonl(fields_path)} if fields_path else {}
    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        meta = row["metadata"]
        key = str(meta.get("artist", "")).strip() or str(meta.get("title", "")).strip() or "unknown"
        row["v7_id"] = f"clp_v7_{index:06d}"
        row["sample_id_v7"] = row["v7_id"]
        grouped.setdefault(key, []).append(row)
    split_map = assign_split([(k, [{**r, "sample_id": r["v7_id"]} for r in v]) for k, v in grouped.items()], seed=seed)

    root = Path(output_dir)
    samples: list[PaintingSample] = []
    style_rule_hits = Counter()
    prompt_sources = Counter()
    for row in rows:
        meta = row["metadata"]
        sid = row["v7_id"]
        image = Image.open(row["image_path"]).convert("RGB")
        w_raw, h_raw = image.size
        scale = min(1.0, target_size / max(w_raw, h_raw))
        if scale < 1.0:
            image = image.resize((round(w_raw * scale), round(h_raw * scale)), Image.LANCZOS)
        image_out = root / "images" / f"{sid}.png"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_out)
        structure_out = root / "structure_maps" / f"{sid}.png"
        save_structure_map(extract_structure_maps(load_rgb(image_out)), structure_out)

        metadata_style, rule = infer_style(meta)
        v = vlm.get(row["sample_id"])
        if v and v.get("parse_ok"):
            fields, source = v["fields"], "vlm"
        else:
            fields, source = fallback_fields(meta), "keyword_fallback"
        # Attribution groups (Mi, Dong Yuan) come from metadata keywords; the three colour-treatment
        # groups come from the VLM's judgement of the image, because many records (notably Wikimedia)
        # have no usable medium field and would otherwise all fall into the default group.
        if metadata_style in ("mi_style", "dong_yuan_style"):
            style, label_source = metadata_style, f"metadata:{rule}"
        elif source == "vlm":
            style, label_source = COLOUR_CLASSES[fields["colour"]], "vlm_colour"
        else:
            style, label_source = metadata_style, f"metadata:{rule}"
        style_rule_hits[label_source] += 1
        prompt_sources[source] += 1
        prompts = build_v7_prompts(fields, style)
        samples.append(
            PaintingSample(
                sample_id=sid,
                image_path=str(image_out.relative_to(root)),
                structure_map_path=str(structure_out.relative_to(root)),
                width=image.size[0],
                height=image.size[1],
                width_raw=int(meta.get("width_original") or w_raw),
                height_raw=int(meta.get("height_original") or h_raw),
                source=str(meta.get("source", "")),
                painter=str(meta.get("artist", "")) or "unknown",
                era=str(meta.get("date_display", "")),
                title=str(meta.get("title", "")),
                object_id=str(meta.get("source_id", "")),
                object_url=str(meta.get("object_url", "")),
                image_url=str(meta.get("image_url", "")),
                license=str(meta.get("license", "")),
                style_label=style,
                brushwork_label="",
                culture=str(meta.get("culture", "")),
                department=str(meta.get("department", "")),
                medium=str(meta.get("medium", "")),
                period=str(meta.get("period", "")),
                dynasty=str(meta.get("dynasty", "")),
                tags=list(meta.get("tags", []) or []),
                dense_caption_zh=prompts.dense_zh,
                dense_caption_en=prompts.dense_en,
                prompt_short_zh=prompts.short_zh,
                prompt_structured_zh=prompts.structured_zh,
                prompt_structured_en=prompts.structured_en,
                split=split_map[sid],
                quality_score=float(row["quality"]["quality_score"]),
                quality_flags=list(row["quality"]["flags"]),
                duplicate_group=str(row.get("duplicate_of", "")),
                extra={
                    "v7_source_key": row["sample_id"],
                    "style_rule": rule,
                    "style_label_metadata": metadata_style,
                    "style_label_source": label_source,
                    "prompt_source": source,
                    "prompt_fields": fields,
                    "vlm_has_inscription": (v or {}).get("fields", {}).get("has_inscription") if v and v.get("parse_ok") else None,
                    "vlm_has_seal": (v or {}).get("fields", {}).get("has_seal") if v and v.get("parse_ok") else None,
                    "seal_count_colour": row.get("seal_count"),
                    "seal_area_fraction": row.get("seal_area_fraction"),
                    "crop": row.get("crop"),
                    "subject_check": row.get("subject_check"),
                    "credit_line": (meta.get("extra") or {}).get("credit_line", ""),
                    "prompt_v6_template": v6_template_prompt(meta),
                },
            )
        )

    assign_train_style_references(samples)
    save_manifest(samples, root / "manifest.jsonl")
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (split_dir / f"{split}.txt").write_text("\n".join(s.sample_id for s in samples if s.split == split), encoding="utf-8")
    stats = {
        "n": len(samples),
        "split": dict(Counter(s.split for s in samples)),
        "source": dict(Counter(s.source for s in samples)),
        "style": dict(Counter(s.style_label for s in samples)),
        "style_by_split": {sp: dict(Counter(s.style_label for s in samples if s.split == sp)) for sp in ["train", "val", "test"]},
        "style_label_sources": dict(style_rule_hits),
        "vlm_vs_metadata_colour_agreement": colour_agreement(samples),
        "prompt_sources": dict(prompt_sources),
        "n_painters": len({s.painter for s in samples}),
        "unique_structured_prompts_test": len({s.prompt_structured_zh for s in samples if s.split == "test"}),
        "n_test": sum(s.split == "test" for s in samples),
        "licenses": dict(Counter(s.license for s in samples)),
    }
    (root / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--fields", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.report, args.fields, args.output_dir, args.target_size, args.seed)
