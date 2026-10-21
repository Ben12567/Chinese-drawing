"""Per-image structured prompt fields from a vision-language model (v7).

In v6 the structured prompt was filled from keyword rules on museum titles/media, with fixed
defaults for most fields, so most test prompts were near-identical. In v7 each image is
described by Qwen2.5-VL-3B-Instruct (greedy decoding), which returns eight content fields;
the ninth field (style) comes from the metadata-derived style label.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from clpgen.data.collection import load_jsonl, save_jsonl

FIELDS = ["subject", "composition", "foreground", "midground", "background", "blank_space", "brush_ink", "atmosphere"]
COLOUR_CLASSES = {"monochrome ink": "ink_monochrome", "ink with light colour": "light_colour", "ink and colour": "ink_and_colour"}

INSTRUCTION = (
    'You are annotating a Chinese landscape painting for a text-to-image dataset. '
    'Look carefully at THIS image and describe only what is visible in it. Write in English only. '
    'Return ONLY a JSON object with the keys below; every text value is a short phrase of at most 8 words '
    'written in your own words for this particular painting.\n'
    '"subject": the main scene;\n'
    '"composition": the overall layout and format (e.g. vertical or horizontal, where the main masses are);\n'
    '"foreground": main elements in the foreground;\n'
    '"midground": main elements in the middle ground;\n'
    '"background": main elements in the background;\n'
    '"blank_space": where empty, unpainted or misty areas are and how large they are;\n'
    '"brush_ink": how ink and colour are applied;\n'
    '"atmosphere": season, weather and mood;\n'
    '"colour": exactly one of "monochrome ink", "ink with light colour", "ink and colour" '
    '(use "ink and colour" only for strong mineral blue/green or other saturated pigments);\n'
    '"has_inscription": "yes" if calligraphic writing appears anywhere on the image, else "no";\n'
    '"has_seal": "yes" if red seal impressions appear, else "no".'
)


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(data.get(key), str) and data.get(key).strip() for key in FIELDS):
        return None
    if data.get("colour") not in COLOUR_CLASSES:
        return None
    if any(not data[key].isascii() for key in FIELDS):
        return None  # e.g. Chinese text leaking into a field; resample
    return data


def main(report_path: str, out_path: str, model_name: str, max_side: int, device: str = "cuda", limit: int = 0, threads: int = 0) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = [r for r in load_jsonl(report_path) if r.get("keep")]
    if limit:
        rows = rows[:limit]
    done: dict[str, dict] = {}
    out = Path(out_path)
    if out.exists():
        done = {r["sample_id"]: r for r in load_jsonl(out)}
    if threads:
        torch.set_num_threads(threads)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=max_side * max_side)
    results = list(done.values())
    for index, row in enumerate(rows):
        if row["sample_id"] in done:
            continue
        image = Image.open(row["image_path"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": INSTRUCTION}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        fields = None
        raw = ""
        for attempt in range(3):
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=attempt > 0, temperature=0.7 if attempt else None)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            fields = parse_json(raw)
            if fields is not None:
                break
        results.append({"sample_id": row["sample_id"], "fields": fields, "raw": raw, "parse_ok": fields is not None})
        if (index + 1) % 25 == 0:
            save_jsonl(results, out)
            print(f"[{index + 1}/{len(rows)}]", flush=True)
    save_jsonl(results, out)
    ok = sum(r["parse_ok"] for r in results)
    print(f"Wrote {len(results)} rows ({ok} parsed) to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")  # Qwen/Qwen2.5-VL-3B-Instruct @ 6628554
    parser.add_argument("--max-side", type=int, default=672)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    main(args.report_path, args.out_path, args.model, args.max_side, args.device, args.limit, args.threads)
