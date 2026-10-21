from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.data.schema import load_manifest
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline


PROMPT_FIELD_MAP = {
    "short": "prompt_short_zh",
    "structured": "prompt_structured_zh",
    "dense": "dense_caption_zh",
}


def _load_image(path: Path, mode: str) -> Image.Image:
    return Image.open(path).convert(mode)


def main(
    config_path: str,
    manifest_path: str,
    split: str,
    output_dir: str,
    prompt_mode: str,
    seeds: list[int],
    limit: int | None,
    checkpoint_dir: str | None,
    skip_existing: bool,
) -> None:
    config = ProjectConfig.from_yaml(config_path)
    if checkpoint_dir:
        config.raw.setdefault("model", {}).setdefault("lora", {})["weights_path"] = checkpoint_dir
        config.raw.setdefault("model", {}).setdefault("hierarchical_adapter", {})["weights_path"] = str(
            Path(checkpoint_dir) / "structure_adapter.pt"
        )
    pipeline = LandscapeGenerationPipeline(config)
    samples = [sample for sample in load_manifest(manifest_path) if sample.split == split]
    if limit is not None:
        samples = samples[:limit]
    prompt_field = PROMPT_FIELD_MAP[prompt_mode]
    manifest_root = Path(manifest_path).parent
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    use_structure_condition = config.get("model", "hierarchical_adapter", "enabled", default=True) or config.get(
        "model",
        "controlnet",
        "enabled",
        default=False,
    )

    for sample in samples:
        structure_map = _load_image(manifest_root / sample.structure_map_path, "RGBA")
        style_reference = None
        if config.get("model", "style_reference", "enabled", default=True) and sample.style_reference_path:
            style_reference = _load_image(manifest_root / sample.style_reference_path, "RGB")
        prompt = getattr(sample, prompt_field)
        for seed in seeds:
            output_path = output_root / f"{sample.sample_id}__seed{seed}.png"
            if skip_existing and output_path.exists():
                continue
            request = GenerationRequest(
                prompt=prompt,
                structure_map=structure_map if use_structure_condition else None,
                style_reference=style_reference,
                seed=seed,
            )
            image = pipeline.generate(request)
            image.save(output_path)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-mode", choices=["short", "structured", "dense"], default="structured")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    main(
        config_path=args.config,
        manifest_path=args.manifest_path,
        split=args.split,
        output_dir=args.output_dir,
        prompt_mode=args.prompt_mode,
        seeds=seeds,
        limit=args.limit,
        checkpoint_dir=args.checkpoint_dir,
        skip_existing=args.skip_existing,
    )
