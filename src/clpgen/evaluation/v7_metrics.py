"""Set-level fidelity metrics with bootstrap confidence intervals (v7).

FID uses the exact low-rank identity Tr sqrt(S1 S2) = ||M||_* with
M = Xc2 Xc1^T / sqrt((n1-1)(n2-1)) (Xc = centred features), which is valid because both
covariance matrices have rank < 2048 at our sample sizes and makes bootstrap resampling cheap.
KID follows clean-fid (cubic polynomial kernel, unbiased MMD^2 over random subsets of size
min(n1, n2)). CMMD follows Jayasumana et al. (CVPR 2024): normalised CLIP ViT-L/14@336
embeddings, Gaussian RBF kernel with sigma = 10, scaled by 1000.
"""
from __future__ import annotations

import numpy as np


def fid_from_features(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    n1, n2 = len(x1), len(x2)
    mu1, mu2 = x1.mean(0), x2.mean(0)
    c1, c2 = x1 - mu1, x2 - mu2
    tr1 = float((c1**2).sum() / (n1 - 1))
    tr2 = float((c2**2).sum() / (n2 - 1))
    m = (c2 @ c1.T) / np.sqrt((n1 - 1) * (n2 - 1))
    nuclear = float(np.linalg.svd(m, compute_uv=False).sum())
    return float(((mu1 - mu2) ** 2).sum() + tr1 + tr2 - 2.0 * nuclear)


def _poly_mmd_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m = len(x)
    return float(
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
    )


def kid_from_features(x1: np.ndarray, x2: np.ndarray, num_subsets: int = 100, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    m = min(len(x1), len(x2), 1000)
    values = []
    for _ in range(num_subsets):
        a = x1[rng.choice(len(x1), m, replace=False)]
        b = x2[rng.choice(len(x2), m, replace=False)]
        values.append(_poly_mmd_unbiased(a, b))
    return float(np.mean(values))


def cmmd_from_embeddings(x: np.ndarray, y: np.ndarray, sigma: float = 10.0, scale: float = 1000.0) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    gamma = 1.0 / (2 * sigma**2)
    xs, ys = (x**2).sum(1), (y**2).sum(1)
    kxx = np.exp(-gamma * (-2 * x @ x.T + xs[:, None] + xs[None, :])).mean()
    kyy = np.exp(-gamma * (-2 * y @ y.T + ys[:, None] + ys[None, :])).mean()
    kxy = np.exp(-gamma * (-2 * x @ y.T + xs[:, None] + ys[None, :])).mean()
    return float(scale * (kxx + kyy - 2 * kxy))


def bootstrap_set_metric(
    gen_feats: np.ndarray,
    gen_groups: list[str],
    real_feats: np.ndarray,
    fn,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI.

    Generated images are resampled by test sample (all inference seeds of a drawn sample are
    kept together); real images are resampled independently.
    """
    rng = np.random.default_rng(seed)
    point = fn(gen_feats, real_feats)
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(gen_groups):
        groups.setdefault(g, []).append(i)
    keys = list(groups)
    values = []
    for _ in range(n_boot):
        drawn = rng.choice(len(keys), len(keys), replace=True)
        idx = [i for k in drawn for i in groups[keys[k]]]
        ridx = rng.choice(len(real_feats), len(real_feats), replace=True)
        values.append(fn(gen_feats[idx], real_feats[ridx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(point), float(lo), float(hi)
