"""Download candidate images listed in a metadata JSONL, downscaling on the fly.

Museum "original" scans are frequently 10-40 MB; every image is resized so that its longest
side is at most --max-side pixels and stored as a high-quality JPEG. The script is resumable
and stops when free disk space drops below --min-free-gb.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
from pathlib import Path

import requests
from PIL import Image

from clpgen.data.collection import USER_AGENT, load_jsonl, save_jsonl

Image.MAX_IMAGE_PIXELS = None


def fetch(session: requests.Session, url: str, timeout: int, retries: int) -> bytes:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # The Art Institute of Chicago asks API/IIIF clients to identify themselves with this header.
            headers = {"AIC-User-Agent": USER_AGENT} if "artic.edu" in url else None
            response = session.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def download_one(session: requests.Session, row: dict, root: Path, max_side: int) -> dict:
    dst = root / row["source"] / f"{row['source_id']}.jpg"
    if dst.exists():
        row["download_path"], row["download_ok"] = str(dst), True
        return row
    try:
        payload = fetch(session, row["image_url"], timeout=90, retries=3)
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        row["width_original"], row["height_original"] = image.size
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, quality=95)
        row["download_path"], row["download_ok"] = str(dst), True
    except Exception as exc:  # noqa: BLE001
        row["download_path"], row["download_ok"] = "", False
        row["download_error"] = str(exc)[:300]
    return row


def main(manifest: str, out_dir: str, out_manifest: str, max_side: int, min_free_gb: float, sleep: float, workers: int = 3) -> None:
    from concurrent.futures import ThreadPoolExecutor

    rows = load_jsonl(manifest)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free / 1e9 < min_free_gb:
        raise SystemExit(f"free disk below {min_free_gb} GB")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    done = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(lambda r: download_one(session, r, root, max_side), rows)):
            done.append(row)
            if (i + 1) % 25 == 0:
                ok = sum(r.get("download_ok", False) for r in done)
                print(f"[{i + 1}/{len(rows)}] ok={ok}", flush=True)
                save_jsonl(done + rows[len(done):], out_manifest)
    save_jsonl(done, out_manifest)
    ok = sum(r.get("download_ok", False) for r in done)
    print(json.dumps({"total": len(done), "ok": ok, "failed": len(done) - ok}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    main(args.manifest, args.out_dir, args.out_manifest, args.max_side, args.min_free_gb, args.sleep, args.workers)
