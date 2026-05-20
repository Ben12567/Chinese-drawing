from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from clpgen.config import ProjectConfig
from clpgen.data.structure_maps import STRUCTURE_CHANNELS
from clpgen.models.hierarchical_adapter import HierarchicalStructureAdapter


@dataclass
class GenerationRequest:
    prompt: str
    prompt_2: str | None = None
    structure_map: Image.Image | np.ndarray | None = None
    style_reference: Image.Image | np.ndarray | None = None
    negative_prompt: str = ""
    negative_prompt_2: str | None = None
    seed: int = 42


class LandscapeGenerationPipeline:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = self._resolve_dtype()
        self.use_structure_adapter = config.get("model", "hierarchical_adapter", "enabled", default=True)
        self.use_controlnet = config.get("model", "controlnet", "enabled", default=False)
        self.structure_channels_active = config.get("dataset", "structure_channels_active") or list(
            STRUCTURE_CHANNELS
        )
        adapter_cfg = config.get("model", "hierarchical_adapter", default={})
        self.structure_adapter = (
            HierarchicalStructureAdapter(
                in_channels=adapter_cfg.get("in_channels", 4),
                base_channels=adapter_cfg.get("base_channels", 64),
                levels=adapter_cfg.get("levels", 4),
                token_dim=adapter_cfg.get("token_dim", 512),
                pooled_sizes=adapter_cfg.get("pooled_sizes", [16, 8, 4, 2]),
            )
            if self.use_structure_adapter
            else None
        )
        self.pipe: Any | None = None

    def _resolve_dtype(self) -> torch.dtype:
        mixed_precision = self.config.get("training", "mixed_precision", default="fp16")
        if self.device.type != "cuda":
            return torch.float32
        if mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float16

    def _require_diffusers(self) -> None:
        try:
            from diffusers import StableDiffusionXLPipeline  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Diffusers is required for SDXL generation. "
                "Install with `python -m pip install diffusers peft`."
            ) from exc

    def load(self) -> None:
        self._require_diffusers()
        from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline

        base_model = self.config.get("model", "base_model")
        pretrained_vae = self.config.get("model", "pretrained_vae")
        local_files_only = self.config.get("model", "local_files_only", default=False)
        variant = self.config.get("model", "variant")
        use_safetensors = self.config.get("model", "use_safetensors", default=True)

        vae = None
        if pretrained_vae:
            vae = AutoencoderKL.from_pretrained(
                pretrained_vae,
                torch_dtype=self.weight_dtype,
                local_files_only=local_files_only,
            )

        pipe_kwargs: dict[str, Any] = {
            "torch_dtype": self.weight_dtype,
            "local_files_only": local_files_only,
            "use_safetensors": use_safetensors,
        }
        if vae is not None:
            pipe_kwargs["vae"] = vae
        if variant and self.device.type == "cuda":
            pipe_kwargs["variant"] = variant

        if self.use_controlnet:
            controlnet_cfg = self.config.get("model", "controlnet", default={})
            controlnet_model = ControlNetModel.from_pretrained(
                controlnet_cfg.get("repo", "diffusers/controlnet-canny-sdxl-1.0"),
                torch_dtype=self.weight_dtype,
                local_files_only=local_files_only,
                use_safetensors=use_safetensors,
            )
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model,
                controlnet=controlnet_model,
                **pipe_kwargs,
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(base_model, **pipe_kwargs)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self._load_lora_weights_if_available()
        self._load_structure_adapter_if_available()
        self._load_ip_adapter_if_available()

    def _load_lora_weights_if_available(self) -> None:
        lora_weights_path = self.config.get("model", "lora", "weights_path")
        if self.pipe is None or not lora_weights_path:
            return
        path = Path(lora_weights_path)
        if self._try_load_peft_bundle(path):
            return
        self.pipe.load_lora_weights(lora_weights_path)

    def _try_load_peft_bundle(self, path: Path) -> bool:
        try:
            from peft import PeftModel
        except ImportError:
            return False

        bundle_root = path if path.is_dir() and (path / "peft_unet").exists() else path.parent
        unet_dir = bundle_root / "peft_unet"
        if not unet_dir.exists() or self.pipe is None:
            return False

        self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, unet_dir, is_trainable=False)
        text_encoder_dir = bundle_root / "peft_text_encoder"
        text_encoder_2_dir = bundle_root / "peft_text_encoder_2"
        if text_encoder_dir.exists():
            self.pipe.text_encoder = PeftModel.from_pretrained(
                self.pipe.text_encoder,
                text_encoder_dir,
                is_trainable=False,
            )
        if text_encoder_2_dir.exists():
            self.pipe.text_encoder_2 = PeftModel.from_pretrained(
                self.pipe.text_encoder_2,
                text_encoder_2_dir,
                is_trainable=False,
            )
        return True

    def _load_structure_adapter_if_available(self) -> None:
        if not self.use_structure_adapter:
            return
        adapter_weights_path = self.config.get("model", "hierarchical_adapter", "weights_path")
        cross_attention_dim = self.pipe.unet.config.cross_attention_dim if self.pipe is not None else 2048
        if self.structure_adapter is None:
            return
        if self.structure_adapter.cross_attention_dim != cross_attention_dim:
            adapter_cfg = self.config.get("model", "hierarchical_adapter", default={})
            self.structure_adapter = HierarchicalStructureAdapter(
                in_channels=adapter_cfg.get("in_channels", 4),
                base_channels=adapter_cfg.get("base_channels", 64),
                levels=adapter_cfg.get("levels", 4),
                token_dim=adapter_cfg.get("token_dim", 512),
                pooled_sizes=adapter_cfg.get("pooled_sizes", [16, 8, 4, 2]),
                cross_attention_dim=cross_attention_dim,
            )
        self.structure_adapter.to(self.device, dtype=self.weight_dtype)
        self.structure_adapter.eval()
        if adapter_weights_path and Path(adapter_weights_path).exists():
            state = torch.load(adapter_weights_path, map_location="cpu")
            self.structure_adapter.load_state_dict(state, strict=True)

    def _load_ip_adapter_if_available(self) -> None:
        if self.pipe is None:
            return
        style_cfg = self.config.get("model", "style_reference", default={})
        if not style_cfg.get("enabled", True):
            return
        repo = style_cfg.get("ip_adapter_repo")
        weight_name = style_cfg.get("ip_adapter_weight_name")
        subfolder = style_cfg.get("ip_adapter_subfolder")
        if not repo or not weight_name:
            return
        kwargs: dict[str, Any] = {
            "weight_name": weight_name,
            "local_files_only": self.config.get("model", "local_files_only", default=False),
        }
        if subfolder:
            kwargs["subfolder"] = subfolder
        image_encoder_folder = style_cfg.get("ip_adapter_image_encoder_folder")
        if image_encoder_folder:
            kwargs["image_encoder_folder"] = image_encoder_folder
        self.pipe.load_ip_adapter(repo, **kwargs)
        self.pipe.set_ip_adapter_scale(self.config.get("inference", "style_conditioning_scale", default=0.6))

    def _mask_structure_channels(self, array: np.ndarray) -> np.ndarray:
        masked = array.copy()
        inactive = [index for index, name in enumerate(STRUCTURE_CHANNELS) if name not in self.structure_channels_active]
        if inactive:
            masked[..., inactive] = 0
        return masked

    def _to_tensor(self, image: Image.Image | np.ndarray, channels: int) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGBA" if channels == 4 else "RGB"))
        else:
            array = image
        if array.shape[-1] != channels:
            raise ValueError(f"Expected {channels} channels, got {array.shape[-1]}.")
        if channels == 4:
            array = self._mask_structure_channels(array)
        tensor = torch.from_numpy(array.astype(np.float32))
        tensor = tensor / (255.0 if channels == 4 else 127.5)
        if channels != 4:
            tensor = tensor - 1.0
        return tensor.permute(2, 0, 1).unsqueeze(0)

    def _build_structure_tokens(self, structure_map: Image.Image | np.ndarray | None) -> torch.Tensor | None:
        if structure_map is None or self.structure_adapter is None:
            return None
        structure_tensor = self._to_tensor(structure_map, channels=4).to(self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            outputs = self.structure_adapter(structure_tensor)
        return outputs["cross_attention_tokens"]

    def _prepare_control_image(self, structure_map: Image.Image | np.ndarray | None) -> Image.Image | None:
        if structure_map is None or not self.use_controlnet:
            return None
        if isinstance(structure_map, Image.Image):
            array = np.asarray(structure_map.convert("RGBA"))
        else:
            array = structure_map
        if array.shape[-1] != 4:
            raise ValueError("ControlNet baseline expects a 4-channel structure map.")
        control_channel_name = self.config.get("model", "controlnet", "channel", default="lineart")
        try:
            channel_index = STRUCTURE_CHANNELS.index(control_channel_name)
        except ValueError:
            channel_index = 0
        gray = array[..., channel_index].astype(np.uint8)
        return Image.fromarray(gray, mode="L").convert("RGB")

    def _encode_prompt_with_structure(self, request: GenerationRequest) -> dict[str, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError("Pipeline is not loaded.")
        guidance_scale = self.config.get("inference", "guidance_scale", default=7.5)
        do_cfg = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            self.pipe.encode_prompt(
                prompt=request.prompt,
                prompt_2=request.prompt_2,
                device=self.device,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=request.negative_prompt,
                negative_prompt_2=request.negative_prompt_2,
            )
        )
        structure_tokens = self._build_structure_tokens(request.structure_map)
        if structure_tokens is not None:
            structure_tokens = structure_tokens.to(dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, structure_tokens], dim=1)
            if do_cfg and negative_prompt_embeds is not None:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, structure_tokens], dim=1)
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    def dry_run(self, request: GenerationRequest) -> dict[str, Any]:
        summary: dict[str, Any] = {"prompt": request.prompt, "seed": request.seed}
        if request.structure_map is not None and self.structure_adapter is not None:
            structure_tensor = self._to_tensor(request.structure_map, channels=4)
            structure_features = self.structure_adapter(structure_tensor)
            summary["structure_token_shapes"] = [tuple(token.shape) for token in structure_features["tokens"]]
            summary["merged_structure_tokens"] = tuple(structure_features["cross_attention_tokens"].shape)
        if request.style_reference is not None:
            summary["style_reference_enabled"] = True
        return summary

    def generate(self, request: GenerationRequest) -> Image.Image:
        if self.pipe is None:
            self.load()
        embeds = self._encode_prompt_with_structure(request)
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        pipe_kwargs: dict[str, Any] = {
            "prompt": None,
            "prompt_2": None,
            "prompt_embeds": embeds["prompt_embeds"],
            "negative_prompt_embeds": embeds["negative_prompt_embeds"],
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"],
            "negative_pooled_prompt_embeds": embeds["negative_pooled_prompt_embeds"],
            "ip_adapter_image": request.style_reference,
            "num_inference_steps": self.config.get("inference", "num_inference_steps", default=30),
            "guidance_scale": self.config.get("inference", "guidance_scale", default=7.5),
            "height": self.config.get("model", "resolution_infer", default=1024),
            "width": self.config.get("model", "resolution_infer", default=1024),
            "generator": generator,
        }
        control_image = self._prepare_control_image(request.structure_map)
        if control_image is not None:
            pipe_kwargs["image"] = control_image
            pipe_kwargs["controlnet_conditioning_scale"] = self.config.get(
                "inference",
                "controlnet_conditioning_scale",
                default=0.8,
            )
        output = self.pipe(
            **pipe_kwargs,
        )
        return output.images[0]

    def save(self, image: Image.Image, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
