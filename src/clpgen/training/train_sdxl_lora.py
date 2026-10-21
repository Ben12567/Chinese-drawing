from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from clpgen.config import ProjectConfig
from clpgen.data.dataset import ChineseLandscapeDataset
from clpgen.models.hierarchical_adapter import HierarchicalStructureAdapter
from clpgen.utils.randomness import set_global_seed


@dataclass
class TrainingSummary:
    num_trainable_params: int
    num_samples: int
    batch_size: int
    image_size: int
    device: str


def _count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def upcast_trainable_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def validate_training_stack() -> None:
    missing: list[str] = []
    try:
        import diffusers  # noqa: F401
    except ImportError:
        missing.append("diffusers")
    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("peft")
    if missing:
        raise ImportError(
            "Training stack is incomplete. Missing packages: "
            + ", ".join(missing)
            + ". Install with `python -m pip install diffusers peft`."
        )


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(config: ProjectConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    precision = config.get("training", "mixed_precision", default="fp16")
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


def build_dataset(config: ProjectConfig, split: str) -> ChineseLandscapeDataset:
    return ChineseLandscapeDataset(
        manifest_path=config.get("dataset", "manifest_path"),
        split=split,
        image_size=config.get("dataset", "image_size", default=768),
        prompt_mode=config.get("dataset", "prompt_mode_train", default="structured"),
        structure_channels_active=config.get("dataset", "structure_channels_active"),
    )


def build_structure_adapter(
    config: ProjectConfig,
    cross_attention_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HierarchicalStructureAdapter | None:
    adapter_cfg = config.get("model", "hierarchical_adapter", default={})
    if not adapter_cfg.get("enabled", True):
        return None
    adapter = HierarchicalStructureAdapter(
        in_channels=adapter_cfg.get("in_channels", 4),
        base_channels=adapter_cfg.get("base_channels", 64),
        levels=adapter_cfg.get("levels", 4),
        token_dim=adapter_cfg.get("token_dim", 512),
        pooled_sizes=adapter_cfg.get("pooled_sizes", [16, 8, 4, 2]),
        cross_attention_dim=cross_attention_dim,
    ).to(device=device, dtype=torch.float32 if device.type == "cuda" and dtype == torch.float16 else dtype)
    weights_path = adapter_cfg.get("weights_path")
    if weights_path and Path(weights_path).exists():
        adapter.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return adapter


def apply_lora(module: torch.nn.Module, rank: int, alpha: int, target_modules: list[str]) -> torch.nn.Module:
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian",
    )
    return get_peft_model(module, peft_config)


def prepare_training_components(config: ProjectConfig) -> TrainingSummary:
    seed = config.get("project", "seed", default=42)
    set_global_seed(seed, deterministic=config.get("project", "deterministic", default=False))
    dataset = build_dataset(config, split="train")
    device = resolve_device()
    return TrainingSummary(
        num_trainable_params=0,
        num_samples=len(dataset),
        batch_size=config.get("training", "batch_size", default=1),
        image_size=config.get("dataset", "image_size", default=768),
        device=str(device),
    )


def dry_run_epoch(config: ProjectConfig, max_steps: int = 4) -> dict[str, int]:
    seed = config.get("project", "seed", default=42)
    set_global_seed(seed, deterministic=config.get("project", "deterministic", default=False))
    dataset = build_dataset(config, split="train")
    if len(dataset) == 0:
        return {"num_batches_checked": 0, "num_samples": 0}

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("training", "batch_size", default=1),
        shuffle=True,
        num_workers=config.get("training", "num_workers", default=0),
        generator=torch.Generator().manual_seed(seed),
    )
    consumed = 0
    for batch in tqdm(dataloader, desc="dry-run"):
        _ = batch["pixel_values"].shape
        _ = batch["structure_map"].shape
        consumed += 1
        if consumed >= max_steps:
            break
    return {"num_batches_checked": consumed, "num_samples": len(dataset)}


def _move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=dtype)
    batch["structure_map"] = batch["structure_map"].to(device=device, dtype=torch.float32)
    batch["style_reference"] = batch["style_reference"].to(device=device, dtype=dtype)
    return batch


def _build_add_time_ids(pipeline: Any, batch_size: int, image_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    projection_dim = pipeline.text_encoder_2.config.projection_dim
    time_ids = pipeline._get_add_time_ids(
        original_size=(image_size, image_size),
        crops_coords_top_left=(0, 0),
        target_size=(image_size, image_size),
        dtype=dtype,
        text_encoder_projection_dim=projection_dim,
    )
    return time_ids.to(device=device).repeat(batch_size, 1)


def save_checkpoint(
    config: ProjectConfig,
    pipeline: Any,
    structure_adapter: HierarchicalStructureAdapter | None,
    output_dir: Path,
    step: int,
) -> None:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers

    step_dir = output_dir / f"checkpoint-{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.unet))
    text_encoder_lora_layers = None
    text_encoder_2_lora_layers = None
    if config.get("model", "lora", "train_text_encoder", default=True):
        text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.text_encoder))
        text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.text_encoder_2))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=step_dir / "lora",
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    )
    pipeline.unet.save_pretrained(step_dir / "peft_unet")
    if config.get("model", "lora", "train_text_encoder", default=True):
        pipeline.text_encoder.save_pretrained(step_dir / "peft_text_encoder")
        pipeline.text_encoder_2.save_pretrained(step_dir / "peft_text_encoder_2")
    if structure_adapter is not None:
        torch.save(structure_adapter.state_dict(), step_dir / "structure_adapter.pt")


def train(config: ProjectConfig) -> None:
    validate_training_stack()
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline

    seed = config.get("project", "seed", default=42)
    set_global_seed(seed, deterministic=config.get("project", "deterministic", default=False))
    device = resolve_device()
    dtype = resolve_dtype(config, device)
    output_dir = Path(config.get("project", "output_root", default="outputs/main_sdxl_landscape"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config, split="train")
    if len(dataset) == 0:
        raise ValueError("Training split is empty.")
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("training", "batch_size", default=1),
        shuffle=True,
        num_workers=config.get("training", "num_workers", default=0),
        generator=torch.Generator().manual_seed(seed),
    )

    base_model = config.get("model", "base_model")
    pretrained_vae = config.get("model", "pretrained_vae")
    local_files_only = config.get("model", "local_files_only", default=False)
    variant = config.get("model", "variant")
    use_safetensors = config.get("model", "use_safetensors", default=True)

    vae = None
    if pretrained_vae:
        vae = AutoencoderKL.from_pretrained(
            pretrained_vae,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    pipe_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "use_safetensors": use_safetensors,
    }
    if vae is not None:
        pipe_kwargs["vae"] = vae
    if variant and device.type == "cuda":
        pipe_kwargs["variant"] = variant

    pipeline = StableDiffusionXLPipeline.from_pretrained(base_model, **pipe_kwargs).to(device)
    pipeline.set_progress_bar_config(disable=True)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    lora_cfg = config.get("model", "lora", default={})
    pipeline.unet.requires_grad_(False)
    pipeline.unet = apply_lora(
        pipeline.unet,
        rank=lora_cfg.get("rank", 16),
        alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("unet_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
    )
    upcast_trainable_params(pipeline.unet)
    if config.get("training", "gradient_checkpointing", default=True):
        pipeline.unet.enable_gradient_checkpointing()

    if lora_cfg.get("train_text_encoder", True):
        pipeline.text_encoder = apply_lora(
            pipeline.text_encoder,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 16),
            target_modules=lora_cfg.get("text_encoder_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]),
        )
        pipeline.text_encoder_2 = apply_lora(
            pipeline.text_encoder_2,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 16),
            target_modules=lora_cfg.get("text_encoder_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]),
        )
        upcast_trainable_params(pipeline.text_encoder)
        upcast_trainable_params(pipeline.text_encoder_2)

    structure_adapter = build_structure_adapter(
        config=config,
        cross_attention_dim=pipeline.unet.config.cross_attention_dim,
        device=device,
        dtype=dtype,
    )
    if structure_adapter is not None:
        structure_adapter.train()
        upcast_trainable_params(structure_adapter)

    total_trainable_params = _count_params(pipeline.unet)
    if lora_cfg.get("train_text_encoder", True):
        total_trainable_params += _count_params(pipeline.text_encoder)
        total_trainable_params += _count_params(pipeline.text_encoder_2)
    if structure_adapter is not None:
        total_trainable_params += _count_params(structure_adapter)
    print(
        TrainingSummary(
            num_trainable_params=total_trainable_params,
            num_samples=len(dataset),
            batch_size=config.get("training", "batch_size", default=1),
            image_size=config.get("dataset", "image_size", default=768),
            device=str(device),
        )
    )

    param_groups = [
        {
            "params": [param for param in pipeline.unet.parameters() if param.requires_grad],
            "lr": config.get("training", "learning_rate", default=1.0e-4),
        },
    ]
    if structure_adapter is not None:
        param_groups.append(
            {
                "params": [param for param in structure_adapter.parameters() if param.requires_grad],
                "lr": config.get("training", "adapter_learning_rate", default=5.0e-5),
            }
        )
    if lora_cfg.get("train_text_encoder", True):
        param_groups.append(
            {
                "params": [param for param in pipeline.text_encoder.parameters() if param.requires_grad]
                + [param for param in pipeline.text_encoder_2.parameters() if param.requires_grad],
                "lr": config.get("training", "learning_rate", default=1.0e-4),
            }
        )
    optimizer = AdamW(param_groups)

    global_step = 0
    num_epochs = config.get("training", "num_epochs", default=1)
    max_train_steps = config.get("training", "max_train_steps")
    checkpoint_every = config.get("training", "checkpoint_every", default=1000)
    max_grad_norm = config.get("training", "max_grad_norm", default=1.0)
    grad_accum_steps = config.get("training", "grad_accum_steps", default=1)
    image_size = config.get("dataset", "image_size", default=768)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for step_in_epoch, batch in enumerate(progress):
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            batch = _move_batch(batch, device=device, dtype=dtype)
            with torch.no_grad():
                latents = pipeline.vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                prompt=list(batch["prompt"]),
                prompt_2=list(batch["prompt"]),
                device=device,
                do_classifier_free_guidance=False,
            )
            encoder_hidden_states = prompt_embeds
            if structure_adapter is not None:
                structure_tokens = structure_adapter(batch["structure_map"])["cross_attention_tokens"].to(
                    dtype=prompt_embeds.dtype
                )
                encoder_hidden_states = torch.cat([prompt_embeds, structure_tokens], dim=1)
            add_time_ids = _build_add_time_ids(
                pipeline=pipeline,
                batch_size=latents.shape[0],
                image_size=image_size,
                dtype=prompt_embeds.dtype,
                device=device,
            )

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                model_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean") / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(dataloader)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in param_groups for param in group["params"]],
                    max_grad_norm,
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(loss=float(loss.detach().cpu() * grad_accum_steps), step=global_step)

            if should_step and checkpoint_every and global_step > 0 and global_step % checkpoint_every == 0:
                save_checkpoint(config, pipeline, structure_adapter, output_dir, global_step)

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    save_checkpoint(config, pipeline, structure_adapter, output_dir, global_step or 0)


def main(config_path: str | Path, dry_run: bool = True) -> None:
    config = ProjectConfig.from_yaml(config_path)
    if dry_run:
        summary = prepare_training_components(config)
        print(summary)
        print(dry_run_epoch(config))
        return
    train(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
