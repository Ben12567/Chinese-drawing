"""Write the v7 experiment configs into configs/revision/.

All methods share the same base model, data, prompts, inference settings and seeds; each
config differs from "ours" only in the components named in its file name. Keys that the code
never read in v6 (adapter_conditioning_scale, style_reference.encoder_name, warmup_steps) are
removed so that the configs describe what is run.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DATA = "data/processed/clp_v7"
ALL_CHANNELS = ["lineart", "quantized_depth", "blank_space_mask", "salient_composition_mask"]

BASE = {
    "project": {"name": "", "seed": 42, "output_root": ""},
    "dataset": {
        "manifest_path": f"{DATA}/manifest.jsonl",
        "split_dir": f"{DATA}/splits",
        "image_size": 768,
        "prompt_mode_train": "structured",
        "prompt_mode_eval": "structured",
        "structure_channels_active": ALL_CHANNELS,
    },
    "model": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "variant": "fp16",
        "use_safetensors": True,
        "local_files_only": False,
        "resolution_train": 768,
        "resolution_infer": 1024,
        "lora": {"rank": 8, "alpha": 8, "unet_target_modules": ["to_q", "to_k", "to_v", "to_out.0"], "train_text_encoder": False, "weights_path": None},
        "hierarchical_adapter": {"enabled": True, "in_channels": 4, "base_channels": 32, "token_dim": 256, "levels": 4, "pooled_sizes": [12, 6, 3, 2], "weights_path": None},
        "controlnet": {"enabled": False, "repo": "models/controlnet-canny-sdxl-1.0", "channel": "lineart", "invert_lineart": True, "variant": "fp16"},
        "t2i_adapter": {"enabled": False, "repo": "models/t2i-adapter-lineart-sdxl-1.0", "channel": "lineart", "invert_lineart": True, "variant": "fp16"},
        "style_reference": {"enabled": True, "ip_adapter_repo": "h94/IP-Adapter", "ip_adapter_subfolder": "sdxl_models", "ip_adapter_weight_name": "ip-adapter_sdxl.bin"},
    },
    "training": {
        "batch_size": 1, "grad_accum_steps": 4, "learning_rate": 5.0e-5, "adapter_learning_rate": 2.0e-5,
        "num_epochs": 12, "max_train_steps": 500, "mixed_precision": "bf16", "checkpoint_every": 250,
        "gradient_checkpointing": True, "max_grad_norm": 1.0, "num_workers": 0,
    },
    "inference": {"num_inference_steps": 30, "guidance_scale": 7.5, "style_conditioning_scale": 0.55,
                  "controlnet_conditioning_scale": 0.8, "t2i_adapter_conditioning_scale": 0.8},
}


def variant(name: str, *, adapter: bool, style: bool, train: bool, controlnet: bool = False, t2i: bool = False,
            channels: list[str] | None = None, **training) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["project"]["name"] = name
    cfg["project"]["output_root"] = f"outputs/revision/train/{name}"
    cfg["model"]["hierarchical_adapter"]["enabled"] = adapter
    cfg["model"]["style_reference"]["enabled"] = style
    cfg["model"]["controlnet"]["enabled"] = controlnet
    cfg["model"]["t2i_adapter"]["enabled"] = t2i
    if channels:
        cfg["dataset"]["structure_channels_active"] = channels
    if not train:
        cfg["training"]["max_train_steps"] = 0
        cfg["training"]["num_epochs"] = 0
    for key, value in training.items():
        if key == "prompt_mode_train":
            cfg["dataset"]["prompt_mode_train"] = value
        else:
            cfg["training"][key] = value
    return cfg


CONFIGS = {
    # trained models
    "ours": variant("ours", adapter=True, style=True, train=True),
    "lora_only": variant("lora_only", adapter=False, style=False, train=True),
    # inference-time combinations that reuse the weights above
    "ours_no_style": variant("ours_no_style", adapter=True, style=False, train=False),
    "lora_ip_adapter": variant("lora_ip_adapter", adapter=False, style=True, train=False),
    "lora_controlnet": variant("lora_controlnet", adapter=False, style=False, train=False, controlnet=True),
    # off-the-shelf baselines (no domain adaptation)
    "controlnet": variant("controlnet", adapter=False, style=False, train=False, controlnet=True),
    "ip_adapter_only": variant("ip_adapter_only", adapter=False, style=True, train=False),
    "t2i_adapter": variant("t2i_adapter", adapter=False, style=False, train=False, t2i=True),
    "sdxl_base": variant("sdxl_base", adapter=False, style=False, train=False),
    # channel ablation (adapter retrained with a channel subset; IP-Adapter on, as in the full model)
    "ch_lineart": variant("ch_lineart", adapter=True, style=True, train=True, channels=["lineart"]),
    "ch_depth": variant("ch_depth", adapter=True, style=True, train=True, channels=["quantized_depth"]),
    "ch_blank": variant("ch_blank", adapter=True, style=True, train=True, channels=["blank_space_mask"]),
    "ch_saliency": variant("ch_saliency", adapter=True, style=True, train=True, channels=["salient_composition_mask"]),
    "ch_lineart_depth": variant("ch_lineart_depth", adapter=True, style=True, train=True, channels=["lineart", "quantized_depth"]),
    "ch_no_saliency": variant("ch_no_saliency", adapter=True, style=True, train=True, channels=["lineart", "quantized_depth", "blank_space_mask"]),
    # LoRA-only diagnostics
    "lora_only_short_train": variant("lora_only_short_train", adapter=False, style=False, train=True, prompt_mode_train="short"),
    "lora_only_1000": variant("lora_only_1000", adapter=False, style=False, train=True, max_train_steps=1000),
    "lora_only_lr1e4": variant("lora_only_lr1e4", adapter=False, style=False, train=True, learning_rate=1.0e-4),
}


def main() -> None:
    out = Path("configs/revision")
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(CONFIGS)} configs to {out}")


if __name__ == "__main__":
    main()
