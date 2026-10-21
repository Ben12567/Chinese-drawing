"""Numbered contact sheets of all retained images for the visual audit (S1 Fig source)."""
import json, math, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

report, out_dir = sys.argv[1], Path(sys.argv[2])
rows = [json.loads(l) for l in open(report, encoding="utf-8")]
kept = [r for r in rows if r.get("keep")]
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "audit_index.json").write_text(json.dumps([r["sample_id"] for r in kept]), encoding="utf-8")
cols, per, cell = 10, 60, 190
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = None
for page in range(math.ceil(len(kept) / per)):
    items = kept[page * per:(page + 1) * per]
    rows_n = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell, rows_n * (cell + 20)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(items):
        im = Image.open(r["image_path"]).convert("RGB"); im.thumbnail((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + 20)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x + 4, y + cell), f"#{page * per + i}", fill="red", font=font)
    sheet.save(out_dir / f"audit_{page:02d}.jpg", quality=80)
print(len(kept), "images,", math.ceil(len(kept) / per), "sheets")
