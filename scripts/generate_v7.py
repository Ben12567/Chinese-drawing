"""Generate evaluation images (v7 protocol).

Changes relative to generate_eval_set.py (v6):
  * --seed-mode per_sample (default) derives the initial-noise seed from (sample_id, seed), so
    different test samples never share initial noise; the same (sample, seed) pair receives the
    same noise for every method, which keeps the comparison paired. v6 used the same seed for
    every sample ("shared"), which made text-only methods with near-identical prompts produce
    near-identical images.
  * the style reference is the sample's manifest entry, which in v7 points to a training image.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.data.schema import load_manifest
from clpgen.pipelines.generator import GenerationRequest, LandscapeGenerationPipeline

PROMPT_FIELD_MAP = {"short": "prompt_short_zh", "structured": "prompt_structured_zh", "dense": "dense_caption_zh"}


def effective_seed(sample_id: str, seed: int, mode: str) -> int:
    if mode == "shared":
        return seed
    return (zlib.crc32(sample_id.encode("utf-8")) + 1_000_003 * seed) % (2**31 - 1)


def main(args: argparse.Namespace) -> None:
    config = ProjectConfig.from_yaml(args.config)
    if args.checkpoint_dir:
        config.raw.setdefault("model", {}).setdefault("lora", {})["weights_path"] = args.checkpoint_dir
        adapter_path = Path(args.checkpoint_dir) / "structure_adapter.pt"
        if adapter_path.exists():
            config.raw.setdefault("model", {}).setdefault("hierarchical_adapter", {})["weights_path"] = str(adapter_path)
        branch_path = Path(args.checkpoint_dir) / "structure_branch.pt"
        if branch_path.exists():
            config.raw.setdefault("model", {}).setdefault("structure_branch", {})["weights_path"] = str(branch_path)
    pipeline = LandscapeGenerationPipeline(config)
    samples = [s for s in load_manifest(args.manifest_path) if s.split == args.split]
    if args.sample_ids:
        wanted = [t.strip() for t in Path(args.sample_ids).read_text().split()] if Path(args.sample_ids).exists() else args.sample_ids.split(",")
        index = {s.sample_id: s for s in samples}
        samples = [index[t] for t in wanted if t in index]
    if args.limit:
        samples = samples[: args.limit]
    root = Path(args.manifest_path).parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_structure = (
        config.get("model", "hierarchical_adapter", "enabled", default=True)
        or config.get("model", "controlnet", "enabled", default=False)
        or config.get("model", "t2i_adapter", "enabled", default=False)
        or config.get("model", "structure_branch", "enabled", default=False)
    )
    use_style = config.get("model", "style_reference", "enabled", default=True)
    seeds = [int(t) for t in args.seeds.split(",") if t.strip()]
    for sample in samples:
        todo = [s for s in seeds if not (args.skip_existing and (out / f"{sample.sample_id}__seed{s}.png").exists())]
        if not todo:
            continue
        structure = Image.open(root / sample.structure_map_path).convert("RGBA")
        reference = Image.open(root / sample.style_reference_path).convert("RGB") if use_style else None
        if args.prompt_mode == "v6template":
            prompt = sample.extra["prompt_v6_template"]
        else:
            prompt = getattr(sample, PROMPT_FIELD_MAP[args.prompt_mode])
        for seed in todo:
            image = pipeline.generate(
                GenerationRequest(
                    prompt=prompt,
                    structure_map=structure if use_structure else None,
                    style_reference=reference,
                    seed=effective_seed(sample.sample_id, seed, args.seed_mode),
                )
            )
            image.save(out / f"{sample.sample_id}__seed{seed}.png")
        print(f"done {sample.sample_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-mode", choices=list(PROMPT_FIELD_MAP) + ["v6template"], default="structured")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--seed-mode", choices=["per_sample", "shared"], default="per_sample")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    main(parser.parse_args())
