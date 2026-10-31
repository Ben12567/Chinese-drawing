"""Checks of the v8 residual structure branch (CPU; needs the local model folders)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from clpgen.models.residual_structure import build_structure_branch, condition_tensor, structural_loss

T2I = Path("models/t2i-adapter-lineart-sdxl-1.0")
CN = Path("models/controlnet-canny-sdxl-1.0")
TAESD = Path("models/taesdxl")
MANIFEST = Path("data/processed/clp_v7/manifest.jsonl")


def _structure_batch(size: int = 256) -> torch.Tensor:
    from clpgen.data.schema import load_manifest

    s = [x for x in load_manifest(MANIFEST) if x.split == "train"][0]
    arr = np.asarray(Image.open(MANIFEST.parent / s.structure_map_path).convert("RGBA").resize((size, size)))
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def test_condition_tensor_masks_after_inversion():
    raw = torch.ones(1, 4, 8, 8)  # line art all white = no lines
    cond = condition_tensor(raw, ["quantized_depth"])
    assert torch.all(cond[:, 0] == 0) and torch.all(cond[:, 1] == 1) and torch.all(cond[:, 2:] == 0)


@pytest.mark.skipif(not T2I.exists(), reason="model not available")
def test_t2i_extension_is_exact_on_lineart():
    from diffusers import T2IAdapter

    raw = _structure_batch()
    ref = T2IAdapter.from_pretrained(T2I, torch_dtype=torch.float32, variant="fp16")
    ext = build_structure_branch({"kind": "t2i", "repo": str(T2I), "variant": "fp16"}, dtype=torch.float32)
    line = 1.0 - raw[:, :1]
    with torch.no_grad():
        a = ref(line.repeat(1, 3, 1, 1))
        b = ext(condition_tensor(raw, ["lineart"]))
        c = ext(condition_tensor(raw))
    for x, y in zip(a, b):
        assert (x - y).abs().max() <= 1e-4 * x.abs().max(), float((x - y).abs().max())
    # with zero-initialised filters the extra channels have no effect before training
    for x, y in zip(b, c):
        assert (x - y).abs().max() <= 1e-4 * x.abs().max(), float((x - y).abs().max())


@pytest.mark.skipif(not TAESD.exists(), reason="model not available")
def test_structural_loss_prefers_matching_condition():
    from clpgen.data.schema import load_manifest
    from diffusers import AutoencoderTiny

    vae = AutoencoderTiny.from_pretrained(TAESD, torch_dtype=torch.float32)
    samples = [x for x in load_manifest(MANIFEST) if x.split == "train"][:6]
    losses_match, losses_other = [], []
    for i, s in enumerate(samples):
        img = np.asarray(Image.open(MANIFEST.parent / s.image_path).convert("RGB").resize((512, 512)))
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        with torch.no_grad():
            rec = (vae.decode(vae.encode(x).latents).sample.clamp(-1, 1) + 1) / 2
        own = np.asarray(Image.open(MANIFEST.parent / s.structure_map_path).convert("RGBA").resize((512, 512)))
        other_s = samples[(i + 1) % len(samples)]
        other = np.asarray(Image.open(MANIFEST.parent / other_s.structure_map_path).convert("RGBA").resize((512, 512)))
        w = torch.ones(1)
        for arr, bucket in [(own, losses_match), (other, losses_other)]:
            t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            bucket.append(float(structural_loss(rec, t, w)))
    assert np.mean(losses_match) < np.mean(losses_other)


@pytest.mark.skipif(not CN.exists(), reason="model not available")
def test_controlnet_extension_is_exact_on_lineart():
    from diffusers import ControlNetModel

    torch.manual_seed(0)
    raw = _structure_batch(256)
    ref = ControlNetModel.from_pretrained(CN, torch_dtype=torch.float32, variant="fp16")
    ext = build_structure_branch({"kind": "controlnet", "repo": str(CN), "variant": "fp16"}, dtype=torch.float32)
    latents = torch.randn(1, 4, 32, 32)
    kwargs = dict(
        timestep=torch.tensor([500]),
        encoder_hidden_states=torch.randn(1, 77, 2048),
        added_cond_kwargs={"text_embeds": torch.randn(1, 1280), "time_ids": torch.tensor([[256.0, 256, 0, 0, 256, 256]])},
        return_dict=False,
    )
    with torch.no_grad():
        down_a, mid_a = ref(latents, controlnet_cond=(1.0 - raw[:, :1]).repeat(1, 3, 1, 1), **kwargs)
        down_b, mid_b = ext(latents, controlnet_cond=condition_tensor(raw), **kwargs)
    for x, y in zip(list(down_a) + [mid_a], list(down_b) + [mid_b]):
        assert (x - y).abs().max() <= 1e-4 * x.abs().max() + 1e-6
