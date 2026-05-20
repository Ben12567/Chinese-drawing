from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _set_nested(data: dict, path: list[str], value) -> None:
    cursor = data
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _apply_ablation(config: dict, ablation: dict) -> dict:
    result = deepcopy(config)
    if "use_hierarchical_adapter" in ablation:
        _set_nested(result, ["model", "hierarchical_adapter", "enabled"], ablation["use_hierarchical_adapter"])
    if "use_style_reference" in ablation:
        _set_nested(result, ["model", "style_reference", "enabled"], ablation["use_style_reference"])
    if "structure_channels" in ablation:
        _set_nested(result, ["dataset", "structure_channels_active"], ablation["structure_channels"])
    if "prompt_mode" in ablation:
        _set_nested(result, ["dataset", "prompt_mode_train"], ablation["prompt_mode"])
        _set_nested(result, ["dataset", "prompt_mode_eval"], ablation["prompt_mode"])
    if "train_text_encoder" in ablation:
        _set_nested(result, ["model", "lora", "train_text_encoder"], ablation["train_text_encoder"])
    result["project"]["name"] = ablation["name"]
    result["project"]["output_root"] = f"outputs/{ablation['name']}"
    return result


def main(base_config_path: str, ablation_config_path: str, output_dir: str) -> None:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    matrix = yaml.safe_load(Path(ablation_config_path).read_text(encoding="utf-8"))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    base_out = output_root / "full_model.yaml"
    base_out.write_text(yaml.safe_dump(base_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for ablation in matrix.get("ablations", []):
        materialized = _apply_ablation(base_config, ablation)
        out_path = output_root / f"{ablation['name']}.yaml"
        out_path.write_text(yaml.safe_dump(materialized, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Wrote {out_path}")
    print(f"Wrote {base_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ablation-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.base_config, args.ablation_config, args.output_dir)
