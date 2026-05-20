from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline


def main(
    config_path: str,
    checkpoint_dir: str,
    prompt: str,
    structure_map: str,
    output_path: str,
    style_reference: str | None = None,
    seed: int = 42,
) -> None:
    config = ProjectConfig.from_yaml(config_path)
    config.raw.setdefault("model", {}).setdefault("lora", {})["weights_path"] = checkpoint_dir
    config.raw.setdefault("model", {}).setdefault("hierarchical_adapter", {})["weights_path"] = str(
        Path(checkpoint_dir) / "structure_adapter.pt"
    )

    pipeline = LandscapeGenerationPipeline(config)
    structure_image = Image.open(structure_map).convert("RGBA")
    style_image = Image.open(style_reference).convert("RGB") if style_reference else None
    request = GenerationRequest(
        prompt=prompt,
        structure_map=structure_image,
        style_reference=style_image,
        seed=seed,
    )
    image = pipeline.generate(request)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--structure-map", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--style-reference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        structure_map=args.structure_map,
        output_path=args.output_path,
        style_reference=args.style_reference,
        seed=args.seed,
    )
