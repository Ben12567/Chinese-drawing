from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline


def _load_tifa_prompts(path: Path) -> list[tuple[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[tuple[str, str]] = []
    for row in rows:
        prompt_id = str(row["id"])
        prompt = str(row["caption"])
        prompts.append((prompt_id, prompt))
    return prompts


def _load_geneval_prompts(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            prompts.append((f"{index:05d}", str(row["prompt"])))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def _load_t2i_compbench_prompts(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append((f"{index:06d}", prompt))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def _zero_structure_map(size: int) -> Image.Image:
    return Image.fromarray(np.zeros((size, size, 4), dtype=np.uint8), mode="RGBA")


def main(
    config_path: str,
    benchmark: str,
    prompt_file: str,
    output_dir: str,
    checkpoint_dir: str | None,
    seed: int,
    limit: int | None,
) -> None:
    config = ProjectConfig.from_yaml(config_path)
    if checkpoint_dir:
        config.raw.setdefault("model", {}).setdefault("lora", {})["weights_path"] = checkpoint_dir
        config.raw.setdefault("model", {}).setdefault("hierarchical_adapter", {})["weights_path"] = str(
            Path(checkpoint_dir) / "structure_adapter.pt"
        )
    config.raw.setdefault("model", {}).setdefault("style_reference", {})["enabled"] = False
    pipeline = LandscapeGenerationPipeline(config)
    prompt_path = Path(prompt_file)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if benchmark == "tifa":
        prompts = _load_tifa_prompts(prompt_path)
    elif benchmark == "geneval":
        prompts = _load_geneval_prompts(prompt_path, limit=limit)
    else:
        prompts = _load_t2i_compbench_prompts(prompt_path, limit=limit)
    if limit is not None and benchmark == "tifa":
        prompts = prompts[:limit]

    structure_map = _zero_structure_map(config.get("dataset", "image_size", default=768))
    id_to_image: dict[str, str] = {}

    for prompt_id, prompt in prompts:
        if benchmark == "geneval":
            sample_dir = output_root / prompt_id / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = sample_dir / "0000.png"
            meta_path = output_root / prompt_id / "metadata.jsonl"
            meta_path.write_text(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n", encoding="utf-8")
        elif benchmark == "t2i_compbench":
            image_path = output_root / f"{prompt}_{prompt_id}.png"
        else:
            image_path = output_root / f"{prompt_id}.png"
            id_to_image[prompt_id] = image_path.name
        request = GenerationRequest(
            prompt=prompt,
            structure_map=structure_map if config.get("model", "hierarchical_adapter", "enabled", default=True) else None,
            style_reference=None,
            seed=seed,
        )
        image = pipeline.generate(request)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        print(f"Saved {image_path}")

    if benchmark == "tifa":
        (output_root / "id2img.json").write_text(json.dumps(id_to_image, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {(output_root / 'id2img.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["tifa", "geneval", "t2i_compbench"], required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(
        config_path=args.config,
        benchmark=args.benchmark,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        limit=args.limit,
    )
