from __future__ import annotations

import argparse
from pathlib import Path

from clpgen.data.collection import HTTPClient, load_jsonl, save_jsonl


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main(manifest_path: str, downloads_dir: str, output_manifest: str, sleep_seconds: float = 0.0) -> None:
    client = HTTPClient(timeout=45, sleep_seconds=sleep_seconds)
    rows = load_jsonl(manifest_path)
    downloads_root = Path(downloads_dir)
    updated: list[dict] = []

    for row in rows:
        image_url = str(row.get("image_url", "")).strip()
        source = str(row.get("source", "src")).strip() or "src"
        source_id = str(row.get("source_id", "")).strip() or "unknown"
        ext = Path(image_url).suffix.lower() or ".jpg"
        row.setdefault("download_path", "")
        row.setdefault("download_ok", False)
        if not image_url or ext not in ALLOWED_SUFFIXES:
            row["download_ok"] = False
            row["download_error"] = f"unsupported_or_missing_url:{ext}"
            updated.append(row)
            continue

        dst = downloads_root / source / f"{source_id}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            row["download_path"] = str(dst)
            row["download_ok"] = True
            updated.append(row)
            continue
        try:
            client.stream_download(image_url, dst)
            row["download_path"] = str(dst)
            row["download_ok"] = True
            row.pop("download_error", None)
        except Exception as exc:  # noqa: BLE001
            row["download_ok"] = False
            row["download_error"] = str(exc)
        updated.append(row)

    save_jsonl(updated, output_manifest)
    print(f"Wrote updated manifest with downloads to {output_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--downloads-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    main(
        manifest_path=args.manifest_path,
        downloads_dir=args.downloads_dir,
        output_manifest=args.output_manifest,
        sleep_seconds=args.sleep_seconds,
    )
