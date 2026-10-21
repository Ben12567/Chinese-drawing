"""Resumable MET collector (same search specs and acceptance rules as MetCollector).

Object JSON is cached on disk so that interrupted runs resume; progress is printed regularly.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from clpgen.data.collection import USER_AGENT, CandidateRecord, MetCollector, looks_like_chinese_landscape, normalize_text, save_jsonl

API = "https://collectionapi.metmuseum.org/public/collection"
PAUSE = 1.0


def search_ids(session: requests.Session, limit: int) -> list[int]:


    search_specs = [
        {"q": "landscape", "geoLocation": "China", "medium": "Paintings"},
        {"q": "mountains", "geoLocation": "China", "medium": "Paintings"},
        {"q": "river", "geoLocation": "China", "medium": "Paintings"},
        {"q": "China", "departmentId": 6, "medium": "Paintings"},
        {"q": "shan shui", "medium": "Paintings"},
        {"q": "ink landscape", "medium": "Paintings"},
        {"q": "pavilion", "geoLocation": "China", "medium": "Paintings"},
        {"q": "waterfall", "geoLocation": "China", "medium": "Paintings"},
        {"q": "mist", "geoLocation": "China", "medium": "Paintings"},
        {"q": "Guo Xi", "medium": "Paintings"},
        {"q": "Fan Kuan", "medium": "Paintings"},
        {"q": "Huang Gongwang", "medium": "Paintings"},
        {"q": "Ni Zan", "medium": "Paintings"},
        {"q": "Wang Meng", "medium": "Paintings"},
        {"q": "Dong Yuan", "medium": "Paintings"},
        {"q": "Mi Fu", "medium": "Paintings"},
        {"q": "Mi Youren", "medium": "Paintings"},
        {"q": "Xia Gui", "medium": "Paintings"},
        {"q": "Ma Yuan", "medium": "Paintings"},
    ]
    ids: list[int] = []
    seen: set[int] = set()
    for spec in search_specs:
        offset = 0
        while True:
            params = {"hasImages": "true", "dateBegin": 900, "dateEnd": 1912, "limit": 100, "offset": offset, **spec}
            data = session.get(f"{API}/v1.1/search", params=params, timeout=30).json()
            page = data.get("objectIDs") or []
            for oid in page:
                if oid not in seen:
                    seen.add(oid)
                    ids.append(oid)
            offset += 100
            if len(ids) >= limit or not page or offset >= int(data.get("total") or 0):
                break
        print(f"search {spec}: {len(ids)} ids", flush=True)
        if len(ids) >= limit:
            break
    return ids[:limit]


def fetch(session: requests.Session, cache: Path, oid: int) -> dict | None:
    import time

    path = cache / f"{oid}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(6):
        try:
            time.sleep(PAUSE)
            r = requests.get(f"{API}/v1/objects/{oid}", headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code == 404:
                path.write_text("{}", encoding="utf-8")
                return {}
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("application/json"):
                print(f"blocked? oid={oid} status={r.status_code} type={r.headers.get('content-type')} -> wait 60s", flush=True)
                time.sleep(60)
                continue
            data = r.json()
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (requests.RequestException, ValueError) as exc:
            print(f"error oid={oid}: {exc!r}"[:200], flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def to_record(obj: dict) -> CandidateRecord | None:
    image_url = normalize_text(obj.get("primaryImage")) or normalize_text(obj.get("primaryImageSmall"))
    if not obj or not obj.get("isPublicDomain") or not image_url:
        return None
    tags = [normalize_text(t.get("term")) for t in (obj.get("tags") or []) if normalize_text(t.get("term"))]
    return CandidateRecord(
        source="met", source_id=str(obj.get("objectID")), title=normalize_text(obj.get("title")),
        artist=normalize_text(obj.get("artistDisplayName")) or normalize_text(obj.get("artistDisplayBio")),
        culture=normalize_text(obj.get("culture")), period=normalize_text(obj.get("period")),
        dynasty=normalize_text(obj.get("dynasty")), medium=normalize_text(obj.get("medium")),
        department=normalize_text(obj.get("department")), object_url=normalize_text(obj.get("objectURL")),
        image_url=image_url, license="CC0 / Public Domain", tags=tags, date_display=normalize_text(obj.get("objectDate")),
        extra={"repository": normalize_text(obj.get("repository")), "classification": normalize_text(obj.get("classification")),
               "object_name": normalize_text(obj.get("objectName")), "country": normalize_text(obj.get("country")),
               "credit_line": normalize_text(obj.get("creditLine"))},
    )


def main(out: str, cache_dir: str, limit: int, workers: int) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ids = search_ids(session, limit)
    print(f"{len(ids)} object ids", flush=True)
    objects: dict[int, dict | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, session, cache, oid): oid for oid in ids}
        for i, fut in enumerate(as_completed(futures)):
            objects[futures[fut]] = fut.result()
            if (i + 1) % 100 == 0:
                print(f"fetched {i + 1}/{len(ids)}", flush=True)
    records = []
    failed = 0
    for oid in ids:
        obj = objects.get(oid)
        if obj is None:
            failed += 1
            continue
        rec = to_record(obj)
        if rec is not None and looks_like_chinese_landscape(rec):
            row = rec.to_dict()
            row["accepted_by_rules"] = True
            row["download_path"] = ""
            records.append(row)
    save_jsonl(records, out)
    print(f"Collected {len(records)} MET records ({failed} fetch failures) into {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default="data/metadata/met_object_cache")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    main(args.out, args.cache_dir, args.limit, args.workers)
