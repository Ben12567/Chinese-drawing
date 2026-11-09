"""Figures 1-8 of the revised manuscript as editable PowerPoint files in the visual style of the original submission.

Layout, colours and typography follow the original figures (pastel panels, Arial, grey image captions); the content
is the revised results. Examples are the same random draws as in make_revision_figures.py; image captions give
per-image metrics instead of the evaluative labels of the original figures.
Writes outputs/revision/figures_ppt/FigN.pptx (one slide each) and FigN.png renders via export_ppt_figures.ps1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import make_revision_figures as F

OUT = F.ROOT / "outputs/revision/figures_ppt"
ASSETS = OUT / "assets"
W = 7.5  # figure width in inches (PLOS maximum)

DARK, GREY, BORDER, PANEL = "1F2933", "667085", "D7D2CA", "FBFAF7"
BLUE, ORANGE, PURPLE, GREEN, RED, TEAL = "0F4D92", "C96B32", "9A4D8E", "4F8A5E", "B5332E", "3F8F8A"
BG_TEAL, BG_PINK, BG_GREEN, BG_BLUE, BG_ORANGE, BG_RED = "E4F0EF", "F2E6EF", "E8F0EA", "DDE8F4", "F7E8DC", "F9ECEA"
STYLE_NAMES = {"ink_monochrome": "ink monochrome", "ink_and_colour": "ink and colour", "light_colour": "light colour",
               "mi_style": "Mi-style", "dong_yuan_style": "Dong Yuan-style"}


# ---------------------------------------------------------------- pptx helpers
class Fig:
    def __init__(self, height: float):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(height)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n_assets = 0

    def box(self, x, y, w, h, fill=None, line=None, lw=0.75, rounded=True, dash=False):
        shape = self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                            Inches(x), Inches(y), Inches(w), Inches(h))
        if rounded:
            shape.adjustments[0] = min(0.08, 0.12 / max(min(w, h), 0.1))
        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        else:
            shape.fill.background()
        if line:
            shape.line.color.rgb = RGBColor.from_string(line)
            shape.line.width = Pt(lw)
        else:
            shape.line.fill.background()
        shape.shadow.inherit = False
        shape.text_frame.text = ""
        return shape

    def text(self, x, y, w, h, text, size=7, bold=False, color=DARK, align="center", anchor="middle", italic=False):
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.02)
        tf.margin_top = tf.margin_bottom = Inches(0.0)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT}[align]
            r = p.add_run()
            r.text = line
            r.font.name = "Arial"
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = RGBColor.from_string(color)
        return tb

    def line(self, x0, y0, x1, y1, color=BLUE, lw=0.9):
        c = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
        c.line.color.rgb = RGBColor.from_string(color)
        c.line.width = Pt(lw)
        return c

    def image(self, img, x, y, w, h, frame=None, crop_square=False):
        """Place an image (path or array) centred in the box (x, y, w, h), keeping its aspect ratio."""
        im = Image.fromarray(img) if isinstance(img, np.ndarray) else Image.open(img)
        im = im.convert("RGB")
        if crop_square:
            s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
        im.thumbnail((1200, 1200))
        self.n_assets += 1
        p = ASSETS / f"{id(self)}_{self.n_assets}.png"
        im.save(p)
        if frame:
            self.box(x, y, w, h, fill=frame, line=BORDER, lw=0.5, rounded=False)
        ar = im.width / im.height
        iw, ih = (w, w / ar) if ar > w / h else (h * ar, h)
        self.slide.shapes.add_picture(str(p), Inches(x + (w - iw) / 2), Inches(y + (h - ih) / 2), Inches(iw), Inches(ih))

    def letter(self, x, y, s="a"):
        self.text(x, y, 0.25, 0.2, s, size=9, bold=True, align="left")

    def save(self, name):
        self.prs.save(OUT / f"{name}.pptx")


def gen(run, sid, seed="42"):
    return F.GEN / run / f"{sid}__seed{seed}.png"


def lineart(m, sid):
    return F.channel_images(F.DATA / m[sid]["structure_map_path"])[0]


def metric_note(df, path):
    return F.annotate(df, str(path)).replace("E ", "Line-art IoU ").replace("  B ", " · Blank IoU ")


# ---------------------------------------------------------------- figures
def fig1(m):
    src = json.loads((F.OUT / "Fig1_sources.json").read_text(encoding="utf-8"))
    src["panel_A_real_painting"] = "clp_v7_000031"  # the painting of the original Fig 1 (MET, Zhao Zuo, Landscape)
    f = Fig(3.96)
    for x, bg, letter, title in [(0.13, BG_TEAL, "a", "Cultural pictorial language"), (2.59, BG_PINK, "b", "Why generic diffusion fails"),
                                 (5.06, BG_GREEN, "c", "Controllable generation")]:
        f.box(x, 0.19, 2.31, 3.64, fill=bg, line="D6DEE2", lw=0.75)
        f.letter(x + 0.06, 0.24, letter)
        f.text(x + 0.2, 0.22, 2.05, 0.24, title, size=7.5, bold=True)
    # a
    f.image(F.DATA / m["clp_v7_000031"]["image_path"], 0.24, 0.62, 0.96, 3.0)
    for i, t in enumerate(["Spatial hierarchy", "Blank-space\norganisation", "Brush-and-ink\nrhythm", "Poetic\natmosphere", "Compositional\nbalance"]):
        f.box(1.27, 0.67 + 0.5 * i, 1.0, 0.36, fill="FFFFFF", line=TEAL, lw=0.75)
        f.text(1.27, 0.67 + 0.5 * i, 1.0, 0.36, t, size=6.3 if i else 6.8)
    # b
    f.box(2.77, 0.62, 1.96, 0.36, fill="FFFFFF", line=PURPLE, lw=0.75)
    f.text(2.77, 0.62, 1.96, 0.36, "Generic text-to-image diffusion", size=6.8, bold=True)
    f.box(2.77, 1.14, 1.96, 0.36, fill="FFFFFF", line=PURPLE, lw=0.75)
    f.text(2.77, 1.14, 1.96, 0.36, "Surface texture imitation", size=6.8)
    f.box(2.77, 1.68, 1.96, 0.74, fill=BG_RED, line="C9605A", lw=0.75)
    f.text(2.77, 1.68, 1.96, 0.74, "Hierarchy loss\nBlank-space errors\nStyle drift", size=6.8, color="A6352F")
    for i, sid in enumerate(src["panel_B_sdxl_prompts"]):
        f.image(gen("sdxl_base", sid), 2.75 + 0.67 * i, 2.58, 0.6, 0.6)
    f.text(2.63, 3.3, 2.2, 0.3, "Unadapted SDXL outputs for three\nbenchmark prompts", size=5.8, color=GREY)
    # c
    for i, (t, bg, ln) in enumerate([("Structured text", BG_BLUE, BLUE), ("Four-channel\nstructure", BG_ORANGE, ORANGE),
                                     ("Style reference", BG_PINK, PURPLE)]):
        f.box(5.19, 0.6 + 0.53 * i, 0.98, 0.38, fill=bg, line=ln, lw=0.9)
        f.text(5.19, 0.6 + 0.53 * i, 0.98, 0.38, t, size=6.6, bold=True)
    f.box(6.33, 0.6, 0.98, 1.44, fill="FFFFFF", line=GREEN, lw=0.9)
    f.text(6.33, 0.6, 0.98, 1.44, "SDXL backbone\n\nLoRA adaptation\nStructure branch\nImage prompting", size=6.3, bold=True)
    f.image(gen("ours_v8_s42", src["panel_C_condition_and_output"]), 5.66, 2.18, 1.14, 1.14, crop_square=True)
    f.text(5.13, 3.38, 2.17, 0.36, "Composition follows the condition map;\ncolour follows the style reference", size=6.3, bold=True, color="3E7B4F")
    f.save("Fig1")
    (F.OUT / "Fig1_sources.json").write_text(json.dumps(src, indent=2), encoding="utf-8")


def fig2(m):
    src = json.loads((F.OUT / "Fig2_sources.json").read_text(encoding="utf-8"))
    sid, ref = src["condition_and_output"], src["style_reference"]
    f = Fig(4.22)
    cols = [(0.19, 1.35, BG_GREEN, "5B8A6B", "Inputs"), (1.75, 1.31, BG_BLUE, BLUE, "Condition encoders"),
            (3.29, 1.37, BG_ORANGE, ORANGE, "SDXL + LoRA"), (4.88, 1.24, BG_PINK, PURPLE, "Denoising"), (6.35, 0.96, BG_RED, "C0504D", "Output")]
    for x, w, bg, ln, t in cols:
        f.box(x, 0.19, w, 3.85, fill=bg, line=ln, lw=0.75)
        f.text(x, 0.24, w, 0.22, t, size=7.5, bold=True)
    f.letter(0.06, 0.06)
    # inputs
    f.box(0.29, 0.6, 1.15, 0.36, fill="FFFFFF", line="5B8A6B", lw=0.75)
    f.text(0.29, 0.6, 1.15, 0.36, "Structured text\nprompt", size=6.6, bold=True)
    f.box(0.29, 1.3, 1.15, 0.36, fill="FFFFFF", line="5B8A6B", lw=0.75)
    f.text(0.29, 1.3, 1.15, 0.36, "Hierarchical\nstructure map", size=6.6, bold=True)
    chans = F.channel_images(F.DATA / m[sid]["structure_map_path"])
    for k, (c, name) in enumerate(zip(chans, ["line art", "layer", "blank", "salience"])):
        f.image(c, 0.25 + 0.31 * k, 1.78, 0.28, 0.42)
        f.text(0.2 + 0.31 * k, 2.22, 0.38, 0.14, name, size=4.8, color=GREY)
    f.box(0.29, 2.5, 1.15, 0.36, fill="FFFFFF", line="5B8A6B", lw=0.75)
    f.text(0.29, 2.5, 1.15, 0.36, "Style-reference\nimage", size=6.6, bold=True)
    f.image(F.DATA / m[ref]["image_path"], 0.34, 2.95, 1.05, 0.98)
    # encoders
    enc = [(0.68, "Text encoders\n(frozen)"), (1.67, "Structure branch\nControlNet copy of the\nSDXL encoder, 4-ch input\n(trained)"),
           (2.78, "Image-prompt encoder\n(IP-Adapter, frozen)")]
    for y, t in enc:
        f.box(1.88, y, 1.05, 0.62 if "Structure" not in t else 0.72, fill="FFFFFF", line=BLUE, lw=0.9)
        f.text(1.88, y, 1.05, 0.62 if "Structure" not in t else 0.72, t, size=6.0, bold=True)
    for (yi, ye), col in zip([(0.78, 0.99), (1.48, 2.03), (2.68, 3.09)], [BLUE, ORANGE, PURPLE]):
        f.line(1.44, yi, 1.88, ye, color=col)
    # SDXL + LoRA
    f.text(3.29, 0.55, 1.37, 0.2, "Text semantics", size=6.6, color=BLUE)
    f.box(3.48, 0.84, 0.99, 2.18, fill="FFFFFF", line=ORANGE, lw=0.9)
    f.text(3.48, 0.84, 0.99, 2.18, "SDXL U-Net\n\nLoRA domain\nadaptation\n\nResidual structure\ninjection\nh + λs·P(F)\n\nDecoupled image\ncross-attention",
           size=6.0, bold=True)
    f.text(3.29, 3.18, 1.37, 0.3, "Structure + style\ncoordination", size=6.6, color=ORANGE)
    for y0, y1, col in [(0.99, 1.2, BLUE), (2.03, 1.93, ORANGE), (3.09, 2.6, PURPLE)]:
        f.line(2.93, y0, 3.48, y1, color=col)
    # denoising
    f.box(5.04, 0.99, 0.92, 1.87, fill="FFFFFF", line=PURPLE, lw=0.9)
    f.text(5.04, 0.99, 0.92, 1.87, "Noise latent\n↓\nIterative\ndenoising\n↓\nImage latent", size=6.6, bold=True)
    f.text(4.92, 3.0, 1.16, 0.4, "Conditioned at each\ndenoising step", size=6.3, color=GREY)
    f.line(4.47, 1.93, 5.04, 1.93, color=PURPLE)
    # output
    f.image(F.GEN / "ours_v8_s42" / f"{sid}__seed42.png", 6.45, 1.2, 0.76, 1.42)
    f.box(6.42, 3.31, 0.82, 0.47, fill="FFFFFF", line="C0504D", lw=0.9)
    f.text(6.42, 3.31, 0.82, 0.47, "Generated\nlandscape painting", size=5.8, bold=True)
    f.line(5.96, 1.93, 6.42, 3.5, color="C0504D")
    f.text(0.19, 4.07, 7.1, 0.14, "Trained on the benchmark: LoRA and the structure branch. Frozen: SDXL, text encoders, VAE, IP-Adapter.",
           size=5.4, color=GREY, align="left")
    f.save("Fig2")


def fig3_chart():
    """Dot plot of KID and CMMD in the style of the original Fig 3."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 7})
    main = pd.read_csv(F.ROOT / "outputs/revision/stats_v8/main_table.csv").set_index("method")
    order = ["SDXL", "T2I-Adapter", "LoRA-only", "IP-Adapter-only", "LoRA+IP-Adapter", "Token adapter", "ControlNet",
             "LoRA+ControlNet", "Ours w/o style ref.", "Ours, line art only", "Ours"]
    real = F.summary("real_test")
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.25))
    cmap = plt.get_cmap("Blues")
    for ax, key, label, title, letter, scale, fmt in [
            (axes[0], "kid", "KID ×10³ (lower is better)", "Distribution distance (KID)", "a", 1000, "{:.1f}"),
            (axes[1], "cmmd", "CMMD (lower is better)", "CLIP-embedding distance (CMMD)", "b", 1, "{:.3f}")]:
        for i, meth in enumerate(order):
            r = main.loc[meth]
            mean = r[f"{key}_mean"] * scale
            if r["n_train_runs"] > 1:
                lo, hi = mean - r[f"{key}_sd"] * scale, mean + r[f"{key}_sd"] * scale
            else:
                lo, hi = r[f"{key}_ci_lo"] * scale, r[f"{key}_ci_hi"] * scale
            col = "#0F4D92" if meth == "Ours" else cmap(0.3 + 0.5 * i / len(order))
            ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=col, ms=4.5, capsize=2.5, lw=1)
            ax.text(hi, i - 0.3, " " + fmt.format(mean), fontsize=6.3, va="center", ha="left")
        ref = real["plain"][key] * scale  # real validation vs. real test paintings
        ax.axvline(ref, ls="--", lw=0.8, color="#888888")
        ax.text(ref, -0.75, " real vs. real", fontsize=5.8, color="#666666", ha="left")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(len(order) - 0.3, -0.9)  # worst method at the top, Ours at the bottom (as in the original Fig 3)
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        ax.text(-0.42, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=0)
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    path = ASSETS / "fig3_chart.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def fig3():
    f = Fig(3.25)
    f.image(fig3_chart(), 0, 0, W, 3.25)
    f.save("Fig3")


def fig4(m):
    runs = [("LoRA-only", "lora_only_s42"), ("ControlNet", "controlnet"), ("IP-Adapter only", "ip_adapter_only"),
            ("LoRA+ControlNet", "lora_controlnet_s42"), ("Ours", "ours_v8_s42")]
    ids = F.pick_random_tests(4)
    dfs = {run: F.per_image(run) for _, run in runs}
    cw, gap, x0, y0, row = 1.13, 0.06, 1.47, 0.36, 1.47
    f = Fig(y0 + 4 * row + 0.05)
    f.letter(0.03, 0.02)
    f.text(0.08, 0.1, 1.3, 0.22, "Prompt / condition", size=7.2, bold=True)
    for j, (name, _) in enumerate(runs):
        f.text(x0 + j * (cw + gap), 0.1, cw, 0.22, name, size=7.2, bold=True, color=BLUE if name == "Ours" else DARK)
    for i, sid in enumerate(ids):
        y = y0 + i * row
        r = m[sid]
        f.box(0.08, y, 1.3, 1.34, fill="FFFFFF", line=BORDER, lw=0.6)
        subject = r["extra"]["prompt_fields"]["subject"][:60]
        f.text(0.13, y + 0.05, 1.2, 0.36, subject[:1].upper() + subject[1:], size=6.2, bold=True, align="left", anchor="top")
        f.text(0.13, y + 0.42, 1.2, 0.14, "Style: " + STYLE_NAMES.get(r["style_label"], r["style_label"]), size=5.6, color=GREY, align="left")
        f.image(lineart(m, sid), 0.15, y + 0.6, 1.16, 0.68, frame="FFFFFF")
        for j, (name, run) in enumerate(runs):
            p = gen(run, sid)
            f.image(p, x0 + j * (cw + gap), y, cw, cw, frame=PANEL)
            f.text(x0 + j * (cw + gap), y + cw + 0.02, cw, 0.16, metric_note(dfs[run], p).replace("Line-art IoU", "E").replace("Blank IoU", "B"),
                   size=5.8, color=RED if name == "Ours" else GREY, bold=name == "Ours")
    f.save("Fig4")


def fig5(m):
    df = pd.read_csv(F.CTRL / "structure_controllability.csv")
    prompt = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt].iloc[:4]
    rh, y0 = 1.62, 0.36
    f = Fig(y0 + 4 * rh + 0.05)
    f.letter(0.03, 0.02)
    f.text(0.15, 0.1, 4.2, 0.22, "Input structure (swapped-in condition map)", size=7.2, bold=True)
    f.text(4.55, 0.1, 2.8, 0.22, "Generated output (Ours)", size=7.2, bold=True)
    for i, (_, r) in enumerate(rows.iterrows()):
        y = y0 + i * rh
        f.text(0.15, y, 5, 0.18, f"Case {'ABCD'[i]}: condition of test #{r.structure_sample[-6:]}", size=6.6, bold=True, align="left")
        f.box(0.15, y + 0.22, 4.25, 1.32, fill="F7F8FA", line=BORDER, lw=0.6)
        f.box(4.55, y + 0.22, 2.8, 1.32, fill="F7F8FA", line=BORDER, lw=0.6)
        chans = F.channel_images(F.DATA / m[r.structure_sample]["structure_map_path"])
        for k, (c, name) in enumerate(zip(chans, ["line art", "layer map", "blank-space mask", "salience mask"])):
            f.image(c, 0.25 + k * 1.03, y + 0.28, 0.97, 1.04)
            f.text(0.2 + k * 1.03, y + 1.33, 1.07, 0.16, name, size=5.4, color=GREY)
        f.image(F.ROOT / r.path, 4.62, y + 0.28, 1.2, 1.2)
        f.text(5.87, y + 0.35, 1.45, 1.1,
               f"Line-art IoU\n{r.edge_iou_used:.3f} with this map\n{r.edge_iou_unused:.3f} with the other maps\n\n"
               f"Blank-space IoU\n{r.blank_iou_used:.3f} / {r.blank_iou_unused:.3f}", size=5.6, color=GREY, align="left", anchor="top")
    f.save("Fig5")
    return prompt


def fig6(m):
    df = pd.read_csv(F.CTRL / "style_controllability.csv")
    prompt = df.prompt_sample.iloc[0]
    rows = df[df.prompt_sample == prompt]
    colours = {"ink_monochrome": "5B6B7F", "ink_and_colour": TEAL, "light_colour": ORANGE}
    f = Fig(3.9)
    f.letter(0.03, 0.02)
    f.text(0, 0.1, W, 0.22, f"Fixed text prompt and structure condition (test #{prompt[-6:]})", size=7.5, bold=True)
    for j, (_, r) in enumerate(rows.iterrows()):
        x = 0.25 + j * 2.4
        f.text(x, 0.42, 2.2, 0.2, STYLE_NAMES[r.reference_group].capitalize(), size=7.5, bold=True, color=colours.get(r.reference_group, DARK))
        f.text(x, 0.64, 2.2, 0.18, "Style reference (training split)", size=6.6, color=GREY)
        f.image(F.DATA / m[r.reference_sample]["image_path"], x + 0.55, 0.86, 1.1, 1.0)
        f.text(x, 1.88, 2.2, 0.16, f"mean saturation {r.ref_mean_saturation:.2f}", size=5.8, color=GREY)
        f.text(x, 2.12, 2.2, 0.18, "Generated output (Ours)", size=6.6, color=GREY)
        f.image(F.ROOT / r.path, x + 0.5, 2.34, 1.2, 1.2)
        f.text(x, 3.56, 2.2, 0.16, f"mean saturation {r.mean_saturation:.2f}", size=5.8, color=GREY)
    f.save("Fig6")
    return prompt


def fig7(m):
    runs = [("Full model", "ours_v8_s42"), ("w/o structure branch\n(LoRA+IP-Adapter)", "lora_ip_adapter_s42"),
            ("w/o style reference", "ours_v8_no_style_s42"), ("Line art only", "ours_v8_lineart_s42")]
    ids = F.pick_random_tests(3)
    dfs = {run: F.per_image(run) for _, run in runs}
    cw, gap, x0, y0, row = 1.25, 0.08, 1.85, 0.48, 1.62
    f = Fig(y0 + 3 * row + 0.05)
    f.letter(0.03, 0.02)
    f.text(0.1, 0.12, 1.6, 0.3, "Input condition\n(line art)", size=7.2, bold=True)
    for j, (name, _) in enumerate(runs):
        f.text(x0 + j * (cw + gap), 0.12, cw, 0.3, name, size=7.0, bold=True, color=BLUE if j == 0 else DARK)
    for i, sid in enumerate(ids):
        y = y0 + i * row
        f.text(0.1, y, 1.6, 0.16, f"Test case {i + 1}", size=6.8, bold=True, color="667085", align="left")
        f.image(lineart(m, sid), 0.1, y + 0.2, 1.6, 1.05)
        for j, (name, run) in enumerate(runs):
            p = gen(run, sid)
            f.image(p, x0 + j * (cw + gap), y, cw, cw, frame=PANEL)
            f.text(x0 + j * (cw + gap), y + cw + 0.02, cw, 0.16, metric_note(dfs[run], p).replace("Line-art IoU", "E").replace("Blank IoU", "B"),
                   size=5.8, color=GREY)
    f.save("Fig7")


def fig8(m):
    df = F.per_image("ours_v8_s42")
    df = df[df.path.str.contains("seed42")]
    lowest = list(df.nsmallest(2, "edge_iou").path)
    wrong = df[(df.colour_correct == 0) & ~df.path.isin(lowest)]
    picks = lowest + list(wrong.sample(2, random_state=F.SAMPLING_SEED).path)
    heads = ["Lowest line-art IoU (1)", "Lowest line-art IoU (2)", "Colour misclassified (1)", "Colour misclassified (2)"]
    cw, gap, x0, y0 = 1.62, 0.12, 0.62, 0.36
    f = Fig(5.25)
    f.letter(0.03, 0.02)
    for lab, y in [("Real test\npainting", y0 + 0.6), ("Line-art\ncondition", y0 + 2.05), ("Output\n(Ours)", y0 + 3.5)]:
        f.text(0.0, y, 0.6, 0.4, lab, size=6.4, bold=True, color=GREY)
    for j, (path, head) in enumerate(zip(picks, heads)):
        x = x0 + j * (cw + gap)
        sid = Path(path).stem.split("__seed")[0]
        row = df[df.path == path].iloc[0]
        f.text(x, 0.1, cw, 0.2, head, size=6.8, bold=True)
        f.image(F.DATA / m[sid]["image_path"], x, y0, cw, 1.38, frame=PANEL)
        f.image(lineart(m, sid), x, y0 + 1.45, cw, 1.38, frame="FFFFFF")
        f.image(F.ROOT / path, x, y0 + 2.9, cw, cw, frame=PANEL)
        f.text(x, y0 + 2.9 + cw + 0.03, cw, 0.5,
               f"E {F.num(row.edge_iou)} · B {F.num(row.blank_iou_re)}\ntarget {row.target_colour}, predicted {row.pred_colour}",
               size=5.8, color=GREY, anchor="top")
    f.save("Fig8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    m = F.manifest()
    fig1(m)
    fig2(m)
    fig3()
    fig4(m)
    fig5(m)
    fig6(m)
    fig7(m)
    fig8(m)
    print("wrote", sorted(p.name for p in OUT.glob("*.pptx")))


if __name__ == "__main__":
    main()
