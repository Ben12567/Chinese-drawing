from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional dependency
    structural_similarity = None


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class MetricResult:
    name: str
    value: float
    details: dict[str, float] | None = None


def _nan() -> float:
    return float("nan")


def _safe_mean(values: Sequence[float]) -> float:
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else _nan()


def load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _resize_gray(gray: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if gray.shape == target_shape:
        return gray
    resized = Image.fromarray(gray.astype(np.uint8)).resize((target_shape[1], target_shape[0]))
    return np.asarray(resized, dtype=np.float32)


def load_rgb(path: str | Path, image_size: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    return image


def edge_consistency_score(reference_structure_map: str | Path, generated_image: str | Path) -> float:
    structure = np.asarray(Image.open(reference_structure_map).convert("RGBA"))
    lineart = structure[..., 0].astype(np.float32)
    gray = _resize_gray(load_gray(generated_image), lineart.shape)
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120).astype(np.float32)
    lineart_bin = (255.0 - lineart) > 127
    edges_bin = edges > 0
    intersection = np.logical_and(lineart_bin, edges_bin).sum()
    union = np.logical_or(lineart_bin, edges_bin).sum()
    return float(intersection / max(union, 1))


def blank_space_iou(reference_structure_map: str | Path, generated_image: str | Path) -> float:
    structure = np.asarray(Image.open(reference_structure_map).convert("RGBA"))
    blank_ref = structure[..., 2] > 127
    gray = _resize_gray(load_gray(generated_image), blank_ref.shape)
    blank_pred = gray > 220
    intersection = np.logical_and(blank_ref, blank_pred).sum()
    union = np.logical_or(blank_ref, blank_pred).sum()
    return float(intersection / max(union, 1))


def blank_space_ssim(reference_structure_map: str | Path, generated_image: str | Path) -> float:
    if structural_similarity is None:
        return _nan()
    structure = np.asarray(Image.open(reference_structure_map).convert("RGBA"))
    blank_ref = (structure[..., 2] > 127).astype(np.uint8) * 255
    gray = _resize_gray(load_gray(generated_image), blank_ref.shape)
    blank_pred = ((gray > 220).astype(np.uint8) * 255).astype(np.uint8)
    return float(structural_similarity(blank_ref, blank_pred, data_range=255))


def structure_consistency(reference_structure_map: str | Path, generated_image: str | Path) -> dict[str, float]:
    return {
        "edge_consistency": edge_consistency_score(reference_structure_map, generated_image),
        "blank_space_iou": blank_space_iou(reference_structure_map, generated_image),
        "blank_space_ssim": blank_space_ssim(reference_structure_map, generated_image),
    }


def aggregate_structure_consistency(pairs: Sequence[tuple[str | Path, str | Path]]) -> dict[str, float]:
    rows = [structure_consistency(reference, generated) for reference, generated in pairs]
    if not rows:
        return {"edge_consistency": _nan(), "blank_space_iou": _nan(), "blank_space_ssim": _nan()}
    return {key: _safe_mean([row[key] for row in rows]) for key in rows[0]}


def compute_fid_kid(reference_dir: str | Path, prediction_dir: str | Path) -> dict[str, float]:
    try:
        from cleanfid import fid
    except ImportError:  # pragma: no cover - optional dependency
        return {"fid": _nan(), "kid": _nan()}
    reference_dir = Path(reference_dir)
    prediction_dir = Path(prediction_dir)
    return {
        "fid": float(fid.compute_fid(str(reference_dir), str(prediction_dir), mode="clean", num_workers=0)),
        "kid": float(fid.compute_kid(str(reference_dir), str(prediction_dir), mode="clean", num_workers=0)),
    }


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batched(items: Sequence, batch_size: int) -> Sequence[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_clip(model_name: str = "openai/clip-vit-base-patch32") -> tuple:
    from transformers import AutoProcessor, CLIPModel

    device = _resolve_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def _clip_image_features(
    image_paths: Sequence[str | Path],
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for batch_paths in _batched(list(image_paths), batch_size):
        images = [load_rgb(path, image_size=224) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            batch_features = model.get_image_features(**inputs)
        if not isinstance(batch_features, torch.Tensor):
            batch_features = getattr(batch_features, "image_embeds", None) or getattr(batch_features, "pooler_output")
        batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
        features.append(batch_features.cpu().numpy())
    return np.concatenate(features, axis=0) if features else np.zeros((0, 512), dtype=np.float32)


def compute_clipscore(
    image_paths: Sequence[str | Path],
    prompts: Sequence[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 8,
) -> dict[str, float | list[float]]:
    if not image_paths:
        return {"clipscore": _nan(), "per_image": []}
    processor, model, device = _load_clip(model_name=model_name)
    scores: list[float] = []
    for batch_paths, batch_prompts in zip(_batched(list(image_paths), batch_size), _batched(list(prompts), batch_size)):
        images = [load_rgb(path, image_size=224) for path in batch_paths]
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=list(batch_prompts), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = getattr(image_features, "image_embeds", None) or getattr(image_features, "pooler_output")
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "text_embeds", None) or getattr(text_features, "pooler_output")
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        batch_scores = (image_features * text_features).sum(dim=-1)
        scores.extend(batch_scores.detach().cpu().numpy().tolist())
    return {"clipscore": _safe_mean(scores), "per_image": scores}


def compute_image_reward(image_paths: Sequence[str | Path], prompts: Sequence[str]) -> dict[str, float | list[float]]:
    try:
        import types
        import packaging.version
        import pkg_resources
        if not hasattr(pkg_resources, "packaging"):
            pkg_resources.packaging = types.SimpleNamespace(version=packaging.version)
        import ImageReward as RM
    except Exception:  # pragma: no cover - optional dependency
        return {"image_reward": _nan(), "per_image": []}
    model = RM.load("ImageReward-v1.0")
    scores = [float(model.score(prompt, str(path))) for path, prompt in zip(image_paths, prompts)]
    return {"image_reward": _safe_mean(scores), "per_image": scores}


def compute_hpsv2(image_paths: Sequence[str | Path], prompts: Sequence[str], hps_version: str = "v2.1") -> dict[str, float | list[float]]:
    try:
        import hpsv2
    except Exception:  # pragma: no cover - optional dependency
        return {"hpsv2": _nan(), "per_image": []}
    scores: list[float] = []
    for path, prompt in zip(image_paths, prompts):
        try:
            value = hpsv2.score(str(path), prompt, hps_version=hps_version)
        except Exception:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        scores.append(float(value))
    return {"hpsv2": _safe_mean(scores), "per_image": scores}


def compute_pickscore(
    image_paths: Sequence[str | Path],
    prompts: Sequence[str],
    model_name: str = "yuvalkirstain/PickScore_v1",
    batch_size: int = 8,
) -> dict[str, float | list[float]]:
    try:
        from transformers import AutoModel, AutoProcessor
    except Exception:  # pragma: no cover - optional dependency
        return {"pickscore": _nan(), "per_image": []}
    if not image_paths:
        return {"pickscore": _nan(), "per_image": []}
    device = _resolve_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    scores: list[float] = []
    logit_scale = None
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp()
    for batch_paths, batch_prompts in zip(_batched(list(image_paths), batch_size), _batched(list(prompts), batch_size)):
        images = [load_rgb(path, image_size=224) for path in batch_paths]
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=list(batch_prompts), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = getattr(image_features, "image_embeds", None) or getattr(image_features, "pooler_output")
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "text_embeds", None) or getattr(text_features, "pooler_output")
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        batch_scores = (image_features * text_features).sum(dim=-1)
        if logit_scale is not None:
            batch_scores = batch_scores * logit_scale
        scores.extend(batch_scores.detach().cpu().numpy().tolist())
    return {"pickscore": _safe_mean(scores), "per_image": scores}


def compute_lpips_diversity(seed_groups: dict[str, list[str | Path]], image_size: int = 256) -> dict[str, float]:
    try:
        import lpips
    except ImportError:  # pragma: no cover - optional dependency
        return {"lpips_diversity": _nan()}
    device = _resolve_device()
    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    scores: list[float] = []
    for image_paths in seed_groups.values():
        if len(image_paths) < 2:
            continue
        tensors: list[torch.Tensor] = []
        for path in image_paths:
            image = load_rgb(path, image_size=image_size)
            array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device))
        for left, right in combinations(tensors, 2):
            with torch.no_grad():
                score = metric(left, right).mean().item()
            scores.append(float(score))
    return {"lpips_diversity": _safe_mean(scores)}


def compute_style_accuracy(
    reference_image_paths: Sequence[str | Path],
    reference_labels: Sequence[str],
    prediction_image_paths: Sequence[str | Path],
    target_labels: Sequence[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 8,
) -> dict[str, float | list[int]]:
    unique_labels = sorted(set(reference_labels))
    if len(unique_labels) < 2 or not prediction_image_paths:
        return {"style_accuracy": _nan(), "per_image": []}
    processor, model, device = _load_clip(model_name=model_name)
    train_features = _clip_image_features(reference_image_paths, processor, model, device, batch_size=batch_size)
    pred_features = _clip_image_features(prediction_image_paths, processor, model, device, batch_size=batch_size)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(list(reference_labels))
    known_labels = set(encoder.classes_.tolist())
    valid_indices = [index for index, label in enumerate(target_labels) if label in known_labels]
    if not valid_indices:
        return {"style_accuracy": _nan(), "per_image": []}
    pred_features = pred_features[valid_indices]
    y_target = encoder.transform([target_labels[index] for index in valid_indices])
    classifier = LogisticRegression(max_iter=2000, multi_class="auto")
    classifier.fit(train_features, y_train)
    predictions = classifier.predict(pred_features)
    matches = (predictions == y_target).astype(np.int32)
    match_map = {index: int(match) for index, match in zip(valid_indices, matches.tolist())}
    per_image = [match_map.get(index, -1) for index in range(len(target_labels))]
    return {"style_accuracy": float(matches.mean()), "per_image": per_image}


def cronbach_alpha(score_matrix: np.ndarray) -> float:
    if score_matrix.ndim != 2 or score_matrix.shape[1] < 2:
        return _nan()
    item_variances = score_matrix.var(axis=0, ddof=1)
    total_variance = score_matrix.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return _nan()
    n_items = score_matrix.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def group_prediction_paths(prediction_dir: str | Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(Path(prediction_dir).glob("*.png")):
        sample_id = path.stem.split("__seed")[0]
        groups[sample_id].append(path)
    return groups
