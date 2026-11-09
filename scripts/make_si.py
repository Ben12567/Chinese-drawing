"""Export the supporting-information files of the revised manuscript from existing result files.

  python scripts/make_si.py

Writes revision/SI/ (S1-S7 Tables, S1-S4 Figs, S1-S2 Appendix). No experiment is run; every value is
read from outputs/revision, data/processed/clp_v7 and data/metadata.
"""
from __future__ import annotations

import json
import pickle
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT / "scripts"))
import make_revision_figures as F  # noqa: E402

OUT = ROOT.parent / "revision" / "SI"
EVAL = ROOT / "outputs/revision/eval"
GEN = ROOT / "outputs/revision/gen"
STATS = ROOT / "outputs/revision/stats_v8"
DATA = ROOT / "data/processed/clp_v7"
SPEC = json.loads((ROOT / "outputs/revision/stats_spec_v8.json").read_text(encoding="utf-8"))
ABL = json.loads((ROOT / "outputs/revision/stats_ablation_v8.json").read_text(encoding="utf-8"))


def manifest() -> dict:
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["sample_id"]: r for r in rows}


def s2_table(m: dict) -> None:
    uses = []
    src = json.loads((ROOT / "outputs/revision/figures/Fig2_sources.json").read_text(encoding="utf-8"))
    uses += [("Fig 2", "condition map (test painting)", src["condition_and_output"]), ("Fig 2", "style reference", src["style_reference"])]
    uses += [("Fig 4", "real test painting / condition", s) for s in F.pick_random_tests(4)]
    ctrl = pd.read_csv(F.CTRL / "structure_controllability.csv")
    rows5 = ctrl[ctrl.prompt_sample == ctrl.prompt_sample.iloc[0]].iloc[:4]
    uses += [("Fig 5", "prompt and style reference of", rows5.prompt_sample.iloc[0])]
    uses += [("Fig 5", "condition map", s) for s in rows5.structure_sample]
    style = pd.read_csv(F.CTRL / "style_controllability.csv")
    rows6 = style[style.prompt_sample == style.prompt_sample.iloc[0]]
    uses += [("Fig 6", "prompt and condition of", rows6.prompt_sample.iloc[0])]
    uses += [("Fig 6", "style reference", s) for s in rows6.reference_sample]
    uses += [("Fig 7", "real test painting / condition", s) for s in F.pick_random_tests(3)]
    df = F.per_image("ours_v8_s42")
    df = df[df.path.str.contains("seed42")]
    lowest = list(df.nsmallest(2, "edge_iou").path)
    wrong = df[(df.colour_correct == 0) & ~df.path.isin(lowest)]
    picks = lowest + list(wrong.sample(2, random_state=F.SAMPLING_SEED).path)
    uses += [("Fig 8", "real test painting / condition", Path(p).stem.split("__seed")[0]) for p in picks]
    f1 = json.loads((ROOT / "outputs/revision/figures/Fig1_sources.json").read_text(encoding="utf-8"))
    uses = ([("Fig 1", "real painting (panel A)", f1["panel_A_real_painting"])]
            + [("Fig 1", "prompt of (panel B, SDXL output)", s) for s in f1["panel_B_sdxl_prompts"]]
            + [("Fig 1", "prompt and condition of the output (panel C)", f1["panel_C_condition_and_output"])]) + uses
    rows = []
    for fig, role, sid in uses:
        r = m[sid]
        rows.append({"figure": fig, "role": role, "sample_id": sid, "split": r["split"], "source": r["source"],
                     "object_id": r["object_id"], "title": r["title"], "artist": re.sub(r"<[^>]+>", "", r["painter"] or "").strip(), "object_url": r["object_url"],
                     "image_url": r["image_url"], "licence": r["license"], "credit_line": r["extra"].get("credit_line", "")})
    pd.DataFrame(rows).to_csv(OUT / "S2_Table.csv", index=False, encoding="utf-8-sig")


def s3_table() -> None:
    report = [json.loads(l) for l in (ROOT / "data/metadata/cleaned_report_v7.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = pd.DataFrame([{"source": r.get("source") or "unknown", "keep": r["keep"], "reason": str(r["reason"]).split(":")[0].split("(")[0].strip()} for r in report])
    steps = rep.assign(reason=np.where(rep.keep, "kept", rep.reason)).pivot_table(index="reason", columns="source", aggfunc="size", fill_value=0)
    steps["total"] = steps.sum(axis=1)
    stats = json.loads((DATA / "dataset_stats.json").read_text(encoding="utf-8"))
    m = pd.DataFrame(manifest().values())
    by_split = pd.crosstab(m.style_label, m.split)
    prev = pd.DataFrame({
        "seal detected (colour rule)": [float((m.extra.map(lambda e: e.get("seal_count_colour", 0)) > 0).mean())],
        "inscription (VLM)": [float((m.extra.map(lambda e: (e.get("prompt_fields") or {}).get("has_inscription")) == "yes").mean())],
        "cropped (mounting removal)": [float(m.extra.map(lambda e: bool((e.get("crop") or {}).get("cropped"))).mean())],
    })
    with pd.ExcelWriter(OUT / "S3_Table.xlsx") as w:
        steps.to_excel(w, sheet_name="cleaning steps")
        pd.Series(stats["source"], name="paintings").to_frame().to_excel(w, sheet_name="final by source")
        by_split.to_excel(w, sheet_name="style labels by split")
        prev.to_excel(w, sheet_name="prevalence", index=False)


def run_row(name: str, d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    row = {"method": name, "run": d.name}
    for v in ["plain", "seal_inpainted", "centre_crop"]:
        if v in s:
            for k in ["fid", "kid", "cmmd"]:
                row[f"{k}_{v}"] = s[v][k]
                if f"{k}_ci" in s[v]:
                    row[f"{k}_{v}_ci_lo"], row[f"{k}_{v}_ci_hi"] = s[v][f"{k}_ci"]
    for k in ["clipscore", "pickscore", "hpsv2", "lpips_diversity"]:
        row[k] = s.get(k)
    for extra in ["structure_v8", "style_v8"]:
        p = d / f"{extra}_summary.json"
        if p.exists():
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                if not isinstance(v, dict):
                    row[f"{extra}_{k}"] = v
    return row


def s4_s5_s6_tables() -> None:
    rows = [run_row(name, ROOT / d) for name, dirs in {**SPEC, **ABL}.items() for d in dirs]
    # Channel ablations of the token adapter (v7, seed 42), reported as promised in the development protocol
    v7_channels = {"ch_lineart_s42": "line art only", "ch_depth_s42": "layer map only", "ch_blank_s42": "blank-space mask only",
                   "ch_saliency_s42": "salience mask only", "ch_lineart_depth_s42": "line art + layer map",
                   "ch_no_saliency_s42": "without salience mask"}
    rows += [run_row(f"Token adapter (seed 42), {label}", EVAL / run) for run, label in v7_channels.items()]
    rows.append(run_row("Real test paintings", EVAL / "real_test"))
    pd.DataFrame(rows).drop_duplicates("run").to_csv(OUT / "S4_Table.csv", index=False)
    tests = pd.concat([pd.read_csv(STATS / "paired_tests.csv").assign(family="main (Holm over 90 tests)"),
                       pd.read_csv(STATS / "ablation_tests.csv").assign(family="ablation (Holm over 45 tests)")])
    tests.to_csv(OUT / "S5_Table.csv", index=False)
    diag = {"diag_lora42_v6template_shared": "original template prompts, shared noise",
            "diag_lora42_v6template_per_sample": "original template prompts, per-sample noise",
            "diag_lora42_structured_shared": "structured prompts, shared noise",
            "lora_only_s42": "structured prompts, per-sample noise (revised protocol)",
            "diag_lora_shorttrain_eval_short": "trained on short prompts, evaluated with short prompts",
            "diag_lora_shorttrain_eval_structured": "trained on short prompts, evaluated with structured prompts",
            "diag_lora_lr1e4": "learning rate 1e-4",
            "diag_lora1000_step250": "1000-step run, checkpoint 250", "diag_lora1000_step500": "1000-step run, checkpoint 500",
            "diag_lora1000_step750": "1000-step run, checkpoint 750", "diag_lora1000_step1000": "1000-step run, checkpoint 1000"}
    out = []
    for run, label in diag.items():
        s = json.loads((EVAL / run / "summary.json").read_text(encoding="utf-8"))
        i = json.loads((EVAL / run / "intersample.json").read_text(encoding="utf-8"))
        out.append({"condition": label, "run": run, "kid": s["plain"]["kid"], "cmmd": s["plain"]["cmmd"], "fid": s["plain"]["fid"],
                    "inter_sample_lpips": i["inter_sample_lpips"], "near_duplicate_fraction": i["near_duplicate_fraction"]})
    pd.DataFrame(out).to_csv(OUT / "S6_Table.csv", index=False)


def s7_table(m: dict) -> None:
    from style_v8 import colour_group, features
    clf = pickle.loads((ROOT / "outputs/revision/style_classifier_v8.pkl").read_bytes())["classifier"]
    style = pd.read_csv(F.CTRL / "style_controllability.csv").drop(columns=["predicted_group"])
    style["reference_colour"] = [colour_group(_Sample(m[s])) for s in style.reference_sample]
    style["predicted_colour"] = clf.predict(features([ROOT / p for p in style.path], None))
    with pd.ExcelWriter(OUT / "S7_Table.xlsx") as w:
        pd.read_csv(F.CTRL / "structure_controllability.csv").to_excel(w, sheet_name="structure swap (Ours)", index=False)
        pd.read_csv(ROOT / "outputs/revision/controllability_lora_controlnet/structure_controllability.csv").to_excel(
            w, sheet_name="structure swap (LoRA+CN)", index=False)
        pd.read_csv(ROOT / "outputs/revision/controllability/structure_controllability.csv").to_excel(
            w, sheet_name="structure swap (Token adapter)", index=False)
        style.to_excel(w, sheet_name="style swap (Ours)", index=False)
        summaries = {name: json.loads((ROOT / "outputs/revision" / d / "summary.json").read_text(encoding="utf-8"))["structure"]
                     for name, d in [("Ours", "controllability_v8_ours"), ("LoRA+ControlNet", "controllability_lora_controlnet"),
                                     ("Token adapter", "controllability")]}
        pd.DataFrame(summaries).T.to_excel(w, sheet_name="structure swap summary")


class _Sample:
    """Minimal stand-in for the manifest schema used by style_v8.colour_group."""
    def __init__(self, r: dict):
        self.extra, self.style_label = r["extra"], r["style_label"]


def sheet(paths: list[Path], cols: int, size: int = 256, labels: list[str] | None = None) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        x, y = (k % cols) * size, (k // cols) * (size + 18)
        canvas.paste(im, (x + (size - im.width) // 2, y + (size - im.height) // 2))
        if labels:
            draw.text((x + 4, y + size + 2), labels[k][:40], fill="black")
    return canvas


def s1_fig() -> None:
    report = [json.loads(l) for l in (ROOT / "data/metadata/cleaned_report_v7.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    removed = [r for r in report if str(r["reason"]).startswith("metadata_non_painting")]
    paths = [ROOT / r["original_path"] for r in removed if r.get("original_path") and (ROOT / r["original_path"]).exists()]
    pages = []
    if paths:
        rng = random.Random(2026)
        pick = rng.sample(paths, min(24, len(paths)))
        pages.append(sheet(pick, 6, labels=[p.stem for p in pick]))
    audit = ROOT / "outputs/revision/audit"
    pages += [Image.open(audit / n).convert("RGB") for n in ["rejected_subject.jpg", "kept_lowest_subject_score.jpg", "crops_00.jpg", "crops_01.jpg"]]
    pages[0].save(OUT / "S1_Fig.pdf", save_all=True, append_images=pages[1:], resolution=150)
    print("S1 Fig: object-type removals with images:", len(paths))


def s2_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    labels = ["coloured", "monochrome"]
    entries = [("Real test paintings", [EVAL / "real_test"])] + [(n, [ROOT / d for d in dirs]) for n, dirs in SPEC.items()]
    fig, axes = plt.subplots(3, 4, figsize=(7.5, 6.2))
    for ax, (name, dirs) in zip(axes.flat, entries):
        df = pd.concat([pd.read_csv(d / "style_v8_per_image.csv") for d in dirs])
        cm = confusion_matrix(df.target, df.pred, labels=labels, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["col.", "mono."], fontsize=6)
        ax.set_yticks([0, 1], ["col.", "mono."], fontsize=6)
        ax.set_title(name, fontsize=6.5)
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    fig.supxlabel("predicted", fontsize=7)
    fig.supylabel("target (label of the test painting)", fontsize=7)
    fig.tight_layout()
    F.save_tiff(fig, OUT / "S2_Fig.tif")


def s3_fig(m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [("original template,\nshared noise", "diag_lora42_v6template_shared"),
             ("original template,\nper-sample noise", "diag_lora42_v6template_per_sample"),
             ("structured prompts,\nshared noise", "diag_lora42_structured_shared"),
             ("structured prompts,\nper-sample noise", "lora_only_s42")]
    ids = F.pick_random_tests(5)
    fig, axes = plt.subplots(len(conds), len(ids), figsize=(7.5, 6.4))
    for i, (label, run) in enumerate(conds):
        for j, sid in enumerate(ids):
            F.show(axes[i, j], GEN / run / f"{sid}__seed42.png")
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=6.5)
            if i == 0:
                axes[i, j].set_title(sid, fontsize=6)
    fig.tight_layout(pad=0.3)
    F.save_tiff(fig, OUT / "S3_Fig.tif")


def appendices(m: dict) -> None:
    import extract_prompt_fields as E
    rng = random.Random(2026)
    ids = rng.sample(sorted(m), 10)
    lines = ["# S1 Appendix. Vision-language-model instruction and prompt examples", "",
             "Model: Qwen2.5-VL-3B-Instruct, greedy decoding, max 256 new tokens, image resized to at most 672 x 672 px. "
             "The response is parsed as JSON; on failure one sampled retry (temperature 0.7) is made, and if that also fails the "
             "keyword template of the original submission is used (51 paintings).", "", "## Instruction", "", "```", E.INSTRUCTION, "```", "",
             "## Structured prompts of ten randomly drawn paintings (random.Random(2026))", ""]
    for sid in ids:
        lines += [f"- **{sid}** ({m[sid]['split']}, {m[sid]['source']}): {m[sid]['prompt_structured_en'] or m[sid]['prompt_structured_zh']}"]
    (OUT / "S1_Appendix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sel = ROOT / "outputs/revision/v8_selection"
    text = (ROOT / "configs/revision/v8_preregistration.md").read_text(encoding="utf-8")
    text = ("# S2 Appendix. Development and test protocol of the structure branch\n\n"
            "Part 1 is the protocol as committed to the code repository (commit 6aaf6cd) before any model with the structure "
            "branch was trained. Part 2 gives the validation results and the selection.\n\n## Part 1. Protocol\n\n"
            + text.split("\n", 1)[1] + "\n\n## Part 2. Validation results\n\n### Stage 1\n\n```\n"
            + (sel / "stage1.csv").read_text(encoding="utf-8") + "```\n\n### Stage 2\n\n```\n"
            + (sel / "stage2.csv").read_text(encoding="utf-8") + "```\n\n### Frozen configuration\n\n```\n"
            + (sel / "frozen.json").read_text(encoding="utf-8") + "\n```\n")
    (OUT / "S2_Appendix.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = manifest()
    shutil.copy(ROOT.parent / "revision" / "S1_Table_hyperparameters.md", OUT / "S1_Table.md")
    s2_table(m)
    s3_table()
    s4_s5_s6_tables()
    s7_table(m)
    s1_fig()
    s2_fig()
    s3_fig(m)
    shutil.copy(ROOT / "outputs/revision/figures/S4_Fig_training_curves.tif", OUT / "S4_Fig.tif")
    appendices(m)
    for p in sorted(OUT.iterdir()):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
