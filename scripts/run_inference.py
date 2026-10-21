from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline


def main(config_path: str, prompt: str, structure_map: str | None, style_reference: str | None, dry_run: bool) -> None:
    config = ProjectConfig.from_yaml(config_path)
    pipeline = LandscapeGenerationPipeline(config)
    structure_image = Image.open(structure_map).convert("RGBA") if structure_map else None
    style_image = Image.open(style_reference).convert("RGB") if style_reference else None
    request = GenerationRequest(prompt=prompt, structure_map=structure_image, style_reference=style_image)
    if dry_run:
        print(pipeline.dry_run(request))
        return
    image = pipeline.generate(request)
    out_path = Path(config.get("project", "output_root", default="outputs")) / "inference" / "sample.png"
    pipeline.save(image, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prompt",
        default="题材对象：秋山；构图层次：近景松树与溪亭，中景山径与云雾水岸，远景层叠山峦；笔墨浓淡：淡墨；设色：浅绛；留白/气韵：大留白、空灵；季节天气：秋日云雾。",
    )
    parser.add_argument("--structure-map")
    parser.add_argument("--style-reference")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.prompt, args.structure_map, args.style_reference, args.dry_run)
