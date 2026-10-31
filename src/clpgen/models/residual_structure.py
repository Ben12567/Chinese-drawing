"""Residual structure branch (v8).

The four-channel structural condition is encoded into multi-scale features that are added to the
U-Net activations, h_l + lambda_s * P_l(F_s^l), i.e. the injection described in the original
manuscript (the v7 code appended pooled structure tokens to the cross-attention context instead).

Two backbones are supported. Both start from a public SDXL edge/line-art model whose input layer
is extended from 3 to 4 channels: the line-art channel (inverted to white lines on black, as the
public models expect) receives the sum of the pretrained RGB filters, and the layer, blank-space
and salience channels receive zero-initialised filters. At initialisation the branch therefore
reproduces the public model on line art exactly; the other channels only contribute what is
learned during fine-tuning.

  kind="t2i"         T2I-Adapter (FullAdapterXL, ~79M parameters), all parameters trained.
  kind="controlnet"  ControlNet; LoRA on its attention projections plus the conditioning
                     embedding and the zero convolutions are trained.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from clpgen.data.structure_maps import STRUCTURE_CHANNELS

CONTROLNET_TRAINABLE_PREFIXES = ("controlnet_cond_embedding", "controlnet_down_blocks", "controlnet_mid_block")


def condition_tensor(structure: torch.Tensor, active: list[str] | None = None) -> torch.Tensor:
    """Raw structure map (B, 4, H, W) in [0, 1], line art stored as dark lines on white, ->
    branch input with line art inverted; inactive channels are zeroed after the inversion, so a
    disabled channel always means "no signal"."""
    cond = structure.clone()
    cond[:, 0] = 1.0 - cond[:, 0]
    if active is not None:
        for index, name in enumerate(STRUCTURE_CHANNELS):
            if name not in active:
                cond[:, index] = 0.0
    return cond


def _expand_t2i_conv_in(adapter: nn.Module) -> None:
    body = adapter.adapter
    old = body.conv_in
    r2 = body.unshuffle.downscale_factor ** 2
    if old.in_channels != 3 * r2:
        raise ValueError(f"unexpected T2I-Adapter input width {old.in_channels}")
    new = nn.Conv2d(4 * r2, old.out_channels, kernel_size=old.kernel_size, padding=old.padding)
    with torch.no_grad():
        new.weight.zero_()
        # PixelUnshuffle orders output channels as c * r^2 + offset.
        new.weight[:, :r2] = old.weight[:, :r2] + old.weight[:, r2 : 2 * r2] + old.weight[:, 2 * r2 :]
        new.bias.copy_(old.bias)
    body.conv_in = new.to(dtype=old.weight.dtype, device=old.weight.device)
    adapter.register_to_config(in_channels=4)


def _expand_controlnet_conv_in(controlnet: nn.Module) -> None:
    emb = controlnet.controlnet_cond_embedding
    old = emb.conv_in
    if old.in_channels != 3:
        raise ValueError(f"unexpected ControlNet conditioning width {old.in_channels}")
    new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, padding=old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, 0] = old.weight.sum(dim=1)
        new.bias.copy_(old.bias)
    emb.conv_in = new.to(dtype=old.weight.dtype, device=old.weight.device)
    controlnet.register_to_config(conditioning_channels=4)


def input_layer_name(kind: str) -> str:
    return "adapter.conv_in" if kind == "t2i" else "controlnet_cond_embedding.conv_in"


def build_structure_branch(cfg: dict, dtype: torch.dtype, trainable: bool = False) -> nn.Module:
    """Load the public backbone, extend it to 4 input channels, attach LoRA (ControlNet) and load
    fine-tuned weights from cfg["weights_path"] if given."""
    from diffusers import ControlNetModel, T2IAdapter

    kind = cfg.get("kind", "t2i")
    kwargs = {"torch_dtype": dtype}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    if kind == "t2i":
        branch = T2IAdapter.from_pretrained(cfg["repo"], **kwargs)
        _expand_t2i_conv_in(branch)
    elif kind == "controlnet":
        branch = ControlNetModel.from_pretrained(cfg["repo"], use_safetensors=True, **kwargs)
        _expand_controlnet_conv_in(branch)
        from peft import LoraConfig, inject_adapter_in_model

        rank = cfg.get("lora_rank", 16)
        inject_adapter_in_model(
            LoraConfig(r=rank, lora_alpha=rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                       lora_dropout=0.0, init_lora_weights="gaussian"),
            branch,
        )
    else:
        raise ValueError(f"unknown structure branch kind {kind}")

    branch.requires_grad_(False)
    if trainable:
        for name, param in branch.named_parameters():
            if kind == "t2i" or "lora_" in name or name.startswith(CONTROLNET_TRAINABLE_PREFIXES):
                param.requires_grad_(True)
    if kind == "controlnet":
        # LoRA layers are injected in place, so the module is still a ControlNetModel and can be
        # used by the diffusers pipeline directly; their weights follow the module dtype.
        for name, param in branch.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype)

    weights_path = cfg.get("weights_path")
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        missing = set(state) - set(dict(branch.named_parameters()))
        if missing:
            raise KeyError(f"structure branch weights contain unknown keys: {sorted(missing)[:5]}")
        own = dict(branch.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value.to(own[name].dtype))
    return branch


def trainable_state_dict(branch: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in branch.named_parameters() if p.requires_grad}


def unet_residual_kwargs(
    branch: nn.Module,
    kind: str,
    cond: torch.Tensor,
    scale: float,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    dtype: torch.dtype,
) -> dict:
    """Training-time forward of the branch; returns the residual keyword arguments of the U-Net."""
    if kind == "t2i":
        features = branch(cond)
        return {"down_intrablock_additional_residuals": [f.to(dtype) * scale for f in features]}
    down, mid = branch(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=cond,
        conditioning_scale=scale,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )
    return {"down_block_additional_residuals": [d.to(dtype) for d in down], "mid_block_additional_residual": mid.to(dtype)}


# ---------------------------------------------------------------------------------------------
# Auxiliary structural consistency loss, L_struct = || Psi(x_hat) - Psi_s(s) ||_1 restricted to
# the regions the condition specifies (one-sided): line-art pixels should be edges of x_hat and
# blank-space pixels should be light and edge-free in x_hat. x_hat is the one-step estimate of the
# clean image decoded with a tiny VAE decoder; the loss is weighted by alpha_bar_t so that it acts
# mainly at low noise levels where x_hat is meaningful.
# ---------------------------------------------------------------------------------------------

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return (k[:, None] * k[None, :])[None, None]


def soft_structure(rgb01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable counterparts of the line-art and blank-space rules: soft edge map from the
    Sobel magnitude of the blurred grey image (threshold between the Canny thresholds 40/255 and
    120/255) and soft blank map (grey > 190/255 and no edge)."""
    g = rgb01.float().mean(dim=1, keepdim=True)
    g = F.conv2d(g, _gaussian_kernel().to(g.device), padding=2)
    sx = _SOBEL_X.to(g.device)[None, None]
    gx = F.conv2d(g, sx, padding=1)
    gy = F.conv2d(g, sx.transpose(-1, -2), padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    edge = torch.sigmoid((mag - 0.31) / 0.05)
    blank = torch.sigmoid((g - 190.0 / 255.0) / 0.02) * (1.0 - edge)
    return edge, blank


def structural_loss(
    x_hat01: torch.Tensor, structure: torch.Tensor, weights: torch.Tensor, active: list[str] | None = None
) -> torch.Tensor:
    """structure: raw map (B, 4, H, W) in [0, 1]; weights: per-sample alpha_bar_t."""
    structure = F.interpolate(structure.float(), size=x_hat01.shape[-2:], mode="nearest")
    edge, blank = soft_structure(x_hat01)
    terms = []
    active = active or list(STRUCTURE_CHANNELS)
    if "lineart" in active:
        target = (structure[:, 0:1] < 0.5).float()
        terms.append((target * (1.0 - edge)).sum(dim=(1, 2, 3)) / target.sum(dim=(1, 2, 3)).clamp_min(1.0))
    if "blank_space_mask" in active:
        target = (structure[:, 2:3] > 0.5).float()
        terms.append((target * (1.0 - blank)).sum(dim=(1, 2, 3)) / target.sum(dim=(1, 2, 3)).clamp_min(1.0))
    if not terms:
        return x_hat01.sum() * 0.0
    per_sample = torch.stack(terms, dim=0).sum(dim=0)
    return (weights.float() * per_sample).mean()
