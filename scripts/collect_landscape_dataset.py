from __future__ import annotations

import argparse
from pathlib import Path

from clpgen.data.collection import COLLECTOR_REGISTRY, HTTPClient, looks_like_chinese_landscape, save_jsonl


ALLOWED_DOWNLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main(
    output_manifest: str,
    downloads_dir: str | None = None,
    sources: list[str] | None = None,
    per_source_limit: int = 1200,
    download_images: bool = False,
    sleep_seconds: float = 0.0,
) -> None:
    source_names = sources or ["met", "cma", "artic", "wikimedia"]
    client = HTTPClient(timeout=45, sleep_seconds=sleep_seconds)
    all_records: list[dict] = []

    downloads_root = Path(downloads_dir) if downloads_dir else None
    if download_images and downloads_root is None:
        raise ValueError("--downloads-dir is required when --download-images is used.")

    for source_name in source_names:
        collector_cls = COLLECTOR_REGISTRY[source_name]
        collector = collector_cls(client)
        records = collector.collect(limit=per_source_limit)
        for record in records:
            row = record.to_dict()
            row["accepted_by_rules"] = looks_like_chinese_landscape(record)
            row["download_path"] = ""
            if download_images and downloads_root is not None:
                ext = Path(record.image_url).suffix or ".jpg"
                if ext.lower() in ALLOWED_DOWNLOAD_SUFFIXES:
                    download_path = downloads_root / source_name / f"{record.source_id}{ext}"
                    try:
                        client.stream_download(record.image_url, download_path)
                        row["download_path"] = str(download_path)
                        row["download_ok"] = True
                    except Exception as exc:  # noqa: BLE001
                        row["download_ok"] = False
                        row["download_error"] = str(exc)
                else:
                    row["download_ok"] = False
                    row["download_error"] = f"unsupported_suffix:{ext.lower()}"
            all_records.append(row)

    save_jsonl(all_records, output_manifest)
    print(f"Collected {len(all_records)} candidate records into {output_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--downloads-dir")
    parser.add_argument("--sources", nargs="+", choices=sorted(COLLECTOR_REGISTRY.keys()))
    parser.add_argument("--per-source-limit", type=int, default=1200)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    main(
        output_manifest=args.output_manifest,
        downloads_dir=args.downloads_dir,
        sources=args.sources,
        per_source_limit=args.per_source_limit,
        download_images=args.download_images,
        sleep_seconds=args.sleep_seconds,
    )
