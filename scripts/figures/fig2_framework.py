"""Fig 2: framework schematic as implemented (token conditioning, frozen IP-Adapter)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from common import save_tiff

TRAINED = "#f6d7b0"
FROZEN = "#dbe7f3"
INPUT = "#e6f0e3"


def box(ax, x, y, w, h, text, color, fontsize=7.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=color, ec="#333333", lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))


def image(ax, path, x, y, w, h, gray=False):
    if path and Path(path).exists():
        im = Image.open(path).convert("L" if gray else "RGB")
        ax.imshow(im, extent=(x, x + w, y, y + h), cmap="gray" if gray else None, aspect="auto", zorder=3)
    else:
        box(ax, x, y, w, h, "image", "#eeeeee", 6)


def main(args: argparse.Namespace) -> None:
    fig = plt.figure(figsize=(7.5, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.56)
    ax.axis("off")

    # Inputs
    ax.text(0.09, 0.535, "Inputs", ha="center", fontsize=8.5, weight="bold")
    box(ax, 0.01, 0.43, 0.16, 0.08, "Structured nine-field\ntext prompt (VLM)", INPUT, 7)
    box(ax, 0.01, 0.27, 0.16, 0.12, "", INPUT)
    ax.text(0.09, 0.375, "4-channel structural map", ha="center", fontsize=7)
    for i, ch in enumerate(args.channels or [None] * 4):
        image(ax, ch, 0.017 + i * 0.038, 0.285, 0.034, 0.07, gray=True)
    ax.text(0.09, 0.275, "line | layer | blank | salience", ha="center", fontsize=5.5, va="top")
    box(ax, 0.01, 0.06, 0.16, 0.17, "", INPUT)
    ax.text(0.09, 0.215, "Style reference\n(training split)", ha="center", fontsize=7, va="top")
    image(ax, args.reference, 0.04, 0.07, 0.10, 0.10)

    # Encoders
    ax.text(0.30, 0.535, "Encoders", ha="center", fontsize=8.5, weight="bold")
    box(ax, 0.22, 0.43, 0.16, 0.08, "SDXL text encoders\n(frozen) → 77 tokens", FROZEN, 7)
    box(ax, 0.22, 0.27, 0.16, 0.12, "Structure adapter (trained)\n4 conv levels → pooled\n12²+6²+3²+2² = 193 tokens\n→ linear to 2048-d", TRAINED, 6.5)
    box(ax, 0.22, 0.09, 0.16, 0.11, "IP-Adapter image\nencoder ViT-bigG/14\n(frozen, inference only)", FROZEN, 6.5)
    arrow(ax, 0.17, 0.47, 0.22, 0.47)
    arrow(ax, 0.17, 0.33, 0.22, 0.33)
    arrow(ax, 0.17, 0.145, 0.22, 0.145)

    # Concatenation
    box(ax, 0.43, 0.36, 0.13, 0.10, "Concatenate\n[77 text ; 193 structure]\ntokens", "#ffffff", 6.5)
    arrow(ax, 0.38, 0.47, 0.43, 0.43)
    arrow(ax, 0.38, 0.33, 0.43, 0.39)

    # U-Net
    ax.text(0.71, 0.535, "SDXL U-Net (frozen) + LoRA (trained)", ha="center", fontsize=8.5, weight="bold")
    box(ax, 0.60, 0.08, 0.22, 0.43, "", "#fbfbfb")
    box(ax, 0.62, 0.36, 0.18, 0.11, "Cross-attention\nQ from image features;\nK, V from text+structure tokens", FROZEN, 6.3)
    box(ax, 0.62, 0.22, 0.18, 0.10, "Decoupled image attention\n+ λr·Attn(Q, K_r, V_r), λr = 0.55", FROZEN, 6.3)
    box(ax, 0.62, 0.10, 0.18, 0.08, "LoRA r = 8 on to_q, to_k,\nto_v, to_out (all attention)", TRAINED, 6.3)
    arrow(ax, 0.56, 0.41, 0.62, 0.41)
    arrow(ax, 0.38, 0.145, 0.62, 0.26)
    ax.text(0.71, 0.085, "30 Euler steps, CFG 7.5; structure tokens in\nboth conditional and unconditional branch", ha="center", va="top", fontsize=5.5)

    # Output
    ax.text(0.92, 0.535, "Output", ha="center", fontsize=8.5, weight="bold")
    image(ax, args.output, 0.85, 0.22, 0.14, 0.14 * 4.2 / 7.5 * (7.5 / 4.2))
    ax.text(0.92, 0.20, "VAE decoder →\n1024 × 1024 image\n(model output)", ha="center", va="top", fontsize=6.5)
    arrow(ax, 0.82, 0.30, 0.85, 0.30)

    # Legend
    for i, (c, t) in enumerate([(TRAINED, "trained"), (FROZEN, "frozen"), (INPUT, "input")]):
        ax.add_patch(FancyBboxPatch((0.43 + i * 0.055, 0.02), 0.012, 0.012, boxstyle="square,pad=0", fc=c, ec="#333"))
        ax.text(0.445 + i * 0.055, 0.026, t, fontsize=6, va="center")
    save_tiff(fig, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--channels", nargs="*", default=None)
    main(parser.parse_args())
