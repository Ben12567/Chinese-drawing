from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import ast
import csv
import json
import os
import time
from io import StringIO

import requests


USER_AGENT = "CLPGenDatasetBuilder/0.1 (+local research use)"


@dataclass
class CandidateRecord:
    source: str
    source_id: str
    title: str
    artist: str
    culture: str
    period: str
    dynasty: str
    medium: str
    department: str
    object_url: str
    image_url: str
    license: str
    tags: list[str]
    width: int = 0
    height: int = 0
    date_display: str = ""
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["extra"] is None:
            data["extra"] = {}
        return data


class HTTPClient:
    def __init__(self, timeout: int = 30, sleep_seconds: float = 0.0, max_retries: int = 3) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(max(self.sleep_seconds, 0.5) * (attempt + 1))
        assert last_error is not None
        raise last_error

    def stream_download(self, url: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.session.get(url, stream=True, timeout=self.timeout) as response:
                    response.raise_for_status()
                    with dst.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return
            except requests.RequestException as exc:
                last_error = exc
                if dst.exists():
                    dst.unlink(missing_ok=True)
                if attempt >= self.max_retries:
                    raise
                time.sleep(max(self.sleep_seconds, 0.5) * (attempt + 1))
        assert last_error is not None
        raise last_error


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(normalize_text(item) for item in value if normalize_text(item))
    if isinstance(value, dict):
        return normalize_text(json.dumps(value, ensure_ascii=False))
    if isinstance(value, str):
        return " ".join(value.replace("\n", " ").split())
    return str(value)


def contains_landscape_semantics(text: str) -> bool:
    text = text.lower()
    keywords = [
        "landscape",
        "mountain",
        "river",
        "shan shui",
        "hanging scroll",
        "ink",
        "ink and color",
        "china",
        "chinese",
        "song",
        "yuan",
        "ming",
        "qing",
    ]
    return any(keyword in text for keyword in keywords)


def looks_like_chinese_landscape(record: CandidateRecord) -> bool:
    haystack = " ".join(
        [
            record.title,
            record.artist,
            record.culture,
            record.period,
            record.dynasty,
            record.medium,
            record.department,
            " ".join(record.tags),
        ]
    )
    haystack_lower = haystack.lower()
    culture_ok = any(
        token in haystack_lower
        for token in ["china", "chinese", "song", "yuan", "ming", "qing", "shan shui"]
    )
    culture_lower = record.culture.lower()
    negative_culture = any(
        token in culture_lower
        for token in ["japan", "japanese", "korea", "korean", "america", "american", "france", "french"]
    )
    medium_ok = any(
        token in haystack_lower
        for token in [
            "painting",
            "hanging scroll",
            "handscroll",
            "album leaf",
            "folding fan",
            "fan mounted as an album leaf",
            "ink",
            "silk",
            "paper",
        ]
    )
    subject_ok = any(
        token in haystack_lower
        for token in [
            "landscape",
            "mountain",
            "river",
            "waterfall",
            "peak",
            "mist",
            "pavilion",
            "cloud",
            "hill",
            "stream",
            "valley",
            "shan shui",
        ]
    )
    return culture_ok and medium_ok and subject_ok and not negative_culture


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MetCollector:
    name = "met"
    CSV_CANDIDATE_PATHS = [
        Path("data/external/met_openaccess/MetObjects.full.csv"),
        Path("data/external/met_openaccess/MetObjects.csv"),
    ]

    def __init__(self, client: HTTPClient) -> None:
        self.client = client

    def _parse_object_id(self, value: Any) -> int:
        digits = "".join(ch for ch in normalize_text(value) if ch.isdigit())
        return int(digits or "0")

    def _csv_rows(self) -> list[dict[str, str]]:
        for path in self.CSV_CANDIDATE_PATHS:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                first_line = f.readline()
                if "Object ID" not in first_line:
                    continue
                f.seek(0)
                return list(csv.DictReader(f))
        return []

    def _fetch_object_record(self, object_id: int) -> CandidateRecord | None:
        try:
            obj = self.client.get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}")
        except requests.RequestException:
            return None
        image_url = normalize_text(obj.get("primaryImage")) or normalize_text(obj.get("primaryImageSmall"))
        if not obj.get("isPublicDomain") or not image_url:
            return None
        tags = [
            normalize_text(tag.get("term"))
            for tag in (obj.get("tags") or [])
            if normalize_text(tag.get("term"))
        ]
        return CandidateRecord(
            source=self.name,
            source_id=str(obj.get("objectID", object_id)),
            title=normalize_text(obj.get("title")),
            artist=normalize_text(obj.get("artistDisplayName")) or normalize_text(obj.get("artistDisplayBio")),
            culture=normalize_text(obj.get("culture")),
            period=normalize_text(obj.get("period")),
            dynasty=normalize_text(obj.get("dynasty")),
            medium=normalize_text(obj.get("medium")),
            department=normalize_text(obj.get("department")),
            object_url=normalize_text(obj.get("objectURL")),
            image_url=image_url,
            license="CC0 / Public Domain",
            tags=tags,
            date_display=normalize_text(obj.get("objectDate")),
            extra={
                "repository": normalize_text(obj.get("repository")),
                "classification": normalize_text(obj.get("classification")),
                "country": normalize_text(obj.get("country")),
            },
        )

    def _collect_from_csv(self, limit: int) -> list[CandidateRecord]:
        rows = self._csv_rows()
        if not rows:
            return []
        candidate_ids: list[int] = []
        seen_ids: set[int] = set()
        for obj in rows:
            if normalize_text(obj.get("Is Public Domain")).lower() not in {"true", "1"}:
                continue
            object_id = self._parse_object_id(obj.get("Object ID"))
            if not object_id or object_id in seen_ids:
                continue
            text = " ".join(
                [
                    normalize_text(obj.get("Title")),
                    normalize_text(obj.get("Culture")),
                    normalize_text(obj.get("Period")),
                    normalize_text(obj.get("Dynasty")),
                    normalize_text(obj.get("Medium")),
                    normalize_text(obj.get("Department")),
                    normalize_text(obj.get("Object Name")),
                    normalize_text(obj.get("Tags")),
                ]
            )
            text_lower = text.lower()
            culture_ok = any(token in text_lower for token in ["china", "chinese", "song", "yuan", "ming", "qing"])
            painting_ok = any(
                token in text_lower
                for token in [
                    "painting",
                    "hanging scroll",
                    "handscroll",
                    "album leaf",
                    "ink",
                    "color on silk",
                    "ink and color",
                    "ink on silk",
                    "ink on paper",
                ]
            )
            subject_ok = any(
                token in text_lower for token in ["landscape", "mountain", "river", "waterfall", "mist", "pavilion", "tree"]
            )
            if culture_ok and painting_ok and subject_ok:
                seen_ids.add(object_id)
                candidate_ids.append(object_id)
                if len(candidate_ids) >= limit * 3:
                    break
        records: list[CandidateRecord] = []
        for object_id in candidate_ids:
            record = self._fetch_object_record(object_id)
            if record is not None and looks_like_chinese_landscape(record):
                records.append(record)
                if len(records) >= limit:
                    break
        return records

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        csv_records = self._collect_from_csv(limit=limit)
        if csv_records:
            return csv_records[:limit]
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
        object_ids: list[int] = []
        seen_ids: set[int] = set()
        for spec in search_specs:
            params = {
                "hasImages": "true",
                "dateBegin": 900,
                "dateEnd": 1912,
                **spec,
            }
            result = self.client.get_json(
                "https://collectionapi.metmuseum.org/public/collection/v1/search",
                params=params,
            )
            for object_id in result.get("objectIDs") or []:
                if object_id not in seen_ids:
                    seen_ids.add(object_id)
                    object_ids.append(object_id)
                    if len(object_ids) >= limit:
                        break
            if len(object_ids) >= limit:
                break
        records: list[CandidateRecord] = []
        for object_id in object_ids:
            record = self._fetch_object_record(object_id)
            if record is not None and looks_like_chinese_landscape(record):
                records.append(record)
        return records


class ClevelandCollector:
    name = "cma"
    CSV_CANDIDATE_PATHS = [
        Path("data/external/cma_openaccess/data.full.csv"),
        Path("data/external/cma_openaccess/data.csv"),
    ]

    def __init__(self, client: HTTPClient) -> None:
        self.client = client

    def _csv_rows(self) -> list[dict[str, str]]:
        for path in self.CSV_CANDIDATE_PATHS:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                first_line = f.readline()
                if "share_license_status" not in first_line:
                    continue
                f.seek(0)
                return list(csv.DictReader(f))
        return []

    def _parse_list_string(self, value: str) -> list[str]:
        text = normalize_text(value)
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [item.strip() for item in text.split(";") if item.strip()]
        if isinstance(parsed, list):
            return [normalize_text(item) for item in parsed if normalize_text(item)]
        return [normalize_text(parsed)] if normalize_text(parsed) else []

    def _collect_from_csv(self, limit: int) -> list[CandidateRecord]:
        rows = self._csv_rows()
        if not rows:
            return []
        records: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        for row in rows:
            if normalize_text(row.get("share_license_status")).lower() != "cc0":
                continue
            source_id = normalize_text(row.get("id"))
            if not source_id or source_id in seen_ids:
                continue
            image_url = (
                normalize_text(row.get("image_print"))
                or normalize_text(row.get("image_web"))
                or normalize_text(row.get("image_full"))
            )
            if not image_url:
                continue
            culture = normalize_text(row.get("culture"))
            title = normalize_text(row.get("title"))
            technique = normalize_text(row.get("technique"))
            support_materials = normalize_text(row.get("support_materials"))
            department = normalize_text(row.get("department"))
            collection = normalize_text(row.get("collection"))
            object_type = normalize_text(row.get("type"))
            description = normalize_text(row.get("description"))
            tombstone = normalize_text(row.get("tombstone"))
            creator_text = normalize_text(row.get("creators"))
            text = " ".join(
                [
                    title,
                    creator_text,
                    culture,
                    technique,
                    support_materials,
                    department,
                    collection,
                    object_type,
                    description,
                    tombstone,
                ]
            ).lower()
            culture_ok = "china" in culture.lower() or "chinese" in culture.lower()
            painting_ok = "painting" in object_type.lower() and any(
                token in text
                for token in [
                    "hanging scroll",
                    "handscroll",
                    "album leaf",
                    "fan mounted as an album leaf",
                    "folding fan",
                    "ink on paper",
                    "ink on silk",
                    "ink and color",
                    "ink and slight color",
                    "ink and light color",
                    "color on silk",
                    "color on paper",
                ]
            )
            subject_ok = any(
                token in text
                for token in [
                    "landscape",
                    "mountain",
                    "river",
                    "waterfall",
                    "mist",
                    "pavilion",
                    "cloud",
                    "hill",
                    "stream",
                    "valley",
                    "spring mountains",
                    "autumn landscape",
                ]
            )
            if not (culture_ok and painting_ok and subject_ok):
                continue
            tags = self._parse_list_string(row.get("artists_tags", ""))
            record = CandidateRecord(
                source=self.name,
                source_id=source_id,
                title=title,
                artist=creator_text or title,
                culture=culture,
                period=normalize_text(row.get("creation_date")),
                dynasty="",
                medium=" ".join(part for part in [technique, support_materials] if part),
                department=" ".join(part for part in [department, collection, object_type] if part),
                object_url=normalize_text(row.get("url")),
                image_url=image_url,
                license="CC0",
                tags=tags,
                date_display=normalize_text(row.get("creation_date")),
                extra={
                    "open_access": True,
                    "type": object_type,
                    "creditline": normalize_text(row.get("creditline")),
                },
            )
            if looks_like_chinese_landscape(record):
                records.append(record)
                seen_ids.add(source_id)
                if len(records) >= limit:
                    break
        return records

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        csv_records = self._collect_from_csv(limit=limit)
        if csv_records:
            return csv_records[:limit]
        queries = [
            "Chinese painting",
            "Chinese landscape painting",
            "Chinese ink painting",
            "shan shui",
            "landscape",
            "Chinese mountain river painting",
            "Chinese ink landscape",
            "China",
            "Guo Xi",
            "Fan Kuan",
            "Huang Gongwang",
            "Ni Zan",
            "Wang Meng",
        ]
        data: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for query in queries:
            params = {
                "q": query,
                "has_image": 1,
                "cc0": 1,
                "limit": min(limit, 1000),
            }
            result = self.client.get_json("https://openaccess-api.clevelandart.org/api/artworks/", params=params)
            for item in result.get("data", []):
                item_id = str(item.get("id", ""))
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    data.append(item)
        records: list[CandidateRecord] = []
        for item in data:
            image_url = normalize_text(item.get("images", {}).get("print", {}).get("url")) or normalize_text(
                item.get("images", {}).get("web", {}).get("url")
            )
            if not image_url:
                continue
            tags = [normalize_text(tag) for tag in (item.get("technique") or "").split(";") if normalize_text(tag)]
            record = CandidateRecord(
                source=self.name,
                source_id=str(item.get("id", "")),
                title=normalize_text(item.get("title")),
                artist=normalize_text(item.get("creators", [{}])[0].get("description"))
                if item.get("creators")
                else normalize_text(item.get("title")),
                culture=normalize_text(item.get("culture")),
                period=normalize_text(item.get("dated")),
                dynasty=normalize_text(item.get("dynasty")),
                medium=normalize_text(item.get("technique")),
                department=normalize_text(item.get("department")),
                object_url=normalize_text(item.get("url")),
                image_url=image_url,
                license="CC0",
                tags=tags,
                width=int(item.get("images", {}).get("print", {}).get("width") or 0),
                height=int(item.get("images", {}).get("print", {}).get("height") or 0),
                date_display=normalize_text(item.get("dated")),
                extra={"open_access": True, "type": normalize_text(item.get("type"))},
            )
            if looks_like_chinese_landscape(record):
                records.append(record)
        return records[:limit]


class ArtICCollector:
    name = "artic"

    def __init__(self, client: HTTPClient) -> None:
        self.client = client

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        params = {
            "q": "Chinese landscape",
            "fields": ",".join(
                [
                    "id",
                    "title",
                    "artist_title",
                    "image_id",
                    "date_display",
                    "place_of_origin",
                    "medium_display",
                    "style_title",
                    "classification_title",
                    "department_title",
                    "is_public_domain",
                    "api_link",
                    "thumbnail",
                    "term_titles",
                ]
            ),
            "limit": min(limit, 100),
            "page": 1,
        }
        records: list[CandidateRecord] = []
        while len(records) < limit:
            try:
                result = self.client.get_json("https://api.artic.edu/api/v1/artworks/search", params=params)
            except requests.RequestException:
                break
            for item in result.get("data", []):
                if not item.get("is_public_domain") or not item.get("image_id"):
                    continue
                image_url = f"https://www.artic.edu/iiif/2/{item['image_id']}/full/843,/0/default.jpg"
                tags = [normalize_text(tag) for tag in item.get("term_titles") or [] if normalize_text(tag)]
                record = CandidateRecord(
                    source=self.name,
                    source_id=str(item.get("id", "")),
                    title=normalize_text(item.get("title")),
                    artist=normalize_text(item.get("artist_title")),
                    culture=normalize_text(item.get("place_of_origin")),
                    period=normalize_text(item.get("style_title")),
                    dynasty="",
                    medium=normalize_text(item.get("medium_display")),
                    department=normalize_text(item.get("department_title")),
                    object_url=normalize_text(item.get("api_link")),
                    image_url=image_url,
                    license="Public Domain",
                    tags=tags,
                    width=int((item.get("thumbnail") or {}).get("width") or 0),
                    height=int((item.get("thumbnail") or {}).get("height") or 0),
                    date_display=normalize_text(item.get("date_display")),
                    extra={"classification_title": normalize_text(item.get("classification_title"))},
                )
                if looks_like_chinese_landscape(record):
                    records.append(record)
                    if len(records) >= limit:
                        break
            pagination = result.get("pagination") or {}
            current_page = int(pagination.get("current_page") or params["page"])
            total_pages = int(pagination.get("total_pages") or current_page)
            if current_page >= total_pages or not result.get("data"):
                break
            params["page"] = current_page + 1
        return records[:limit]


class WikimediaCollector:
    name = "wikimedia"

    def __init__(self, client: HTTPClient) -> None:
        self.client = client

    def _category_members(self, category: str, file_limit: int = 400, depth: int = 1) -> list[str]:
        titles: list[str] = []
        visited_categories: set[str] = set()

        def visit(current_category: str, current_depth: int) -> None:
            if current_category in visited_categories or len(titles) >= file_limit:
                return
            visited_categories.add(current_category)
            continuation = None
            while len(titles) < file_limit:
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{current_category}",
                    "cmnamespace": "6|14",
                    "cmlimit": 500,
                }
                if continuation:
                    params["cmcontinue"] = continuation
                try:
                    result = self.client.get_json("https://commons.wikimedia.org/w/api.php", params=params)
                except requests.RequestException:
                    break
                members = (result.get("query") or {}).get("categorymembers") or []
                if not members:
                    break
                for member in members:
                    title = normalize_text(member.get("title"))
                    ns = int(member.get("ns") or 0)
                    if ns == 6 and title and title not in titles:
                        titles.append(title)
                        if len(titles) >= file_limit:
                            break
                    elif ns == 14 and current_depth < depth:
                        child = title.replace("Category:", "", 1)
                        visit(child, current_depth + 1)
                        if len(titles) >= file_limit:
                            break
                continuation = ((result.get("continue") or {}).get("cmcontinue")) or None
                if not continuation:
                    break

        visit(category, 0)
        return titles[:file_limit]

    def _page_info_to_records(self, titles: list[str]) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        if not titles:
            return records
        batch_size = 25
        for start in range(0, len(titles), batch_size):
            batch = titles[start : start + batch_size]
            try:
                info = self.client.get_json(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "prop": "imageinfo|categories",
                        "titles": "|".join(batch),
                        "iiprop": "url|size|extmetadata",
                        "iiurlwidth": 1600,
                        "cllimit": "max",
                    },
                )
            except requests.RequestException:
                continue
            pages = (info.get("query") or {}).get("pages") or {}
            for page in pages.values():
                image_info = (page.get("imageinfo") or [{}])[0]
                ext = image_info.get("extmetadata") or {}
                image_url = normalize_text(image_info.get("thumburl")) or normalize_text(image_info.get("url"))
                if not image_url:
                    continue
                suffix = Path(image_url).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
                    continue
                categories = [
                    normalize_text(cat.get("title")).replace("Category:", "")
                    for cat in page.get("categories") or []
                ]
                record = CandidateRecord(
                    source=self.name,
                    source_id=str(page.get("pageid", "")),
                    title=normalize_text(page.get("title")),
                    artist=normalize_text((ext.get("Artist") or {}).get("value")),
                    culture=normalize_text((ext.get("ObjectName") or {}).get("value")),
                    period="",
                    dynasty="",
                    medium=normalize_text((ext.get("ImageDescription") or {}).get("value")),
                    department="Wikimedia Commons",
                    object_url=normalize_text(image_info.get("descriptionurl")),
                    image_url=image_url,
                    license=normalize_text((ext.get("LicenseShortName") or {}).get("value")) or "Wikimedia Commons",
                    tags=categories,
                    width=int(image_info.get("width") or 0),
                    height=int(image_info.get("height") or 0),
                    date_display=normalize_text((ext.get("DateTimeOriginal") or {}).get("value")),
                    extra={"categories": categories},
                )
                if looks_like_chinese_landscape(record) or contains_landscape_semantics(" ".join(categories)):
                    records.append(record)
        return records

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        search_terms = [
            'intitle:"Chinese landscape painting"',
            'intitle:"shan shui"',
            'intitle:"Chinese mountain"',
            'incategory:"Chinese landscape paintings"',
            'incategory:"Shan shui"',
            'incategory:"Landscape paintings from China"',
            '"Guo Xi" landscape',
            '"Fan Kuan" landscape',
            '"Huang Gongwang" landscape',
            '"Ni Zan" landscape',
            '"Wang Meng" landscape',
            '"Wu Zhen" landscape',
            '"Shen Zhou" landscape',
            '"Wen Zhengming" landscape',
            '"Dong Qichang" landscape',
            '"Shitao" landscape',
            '"Bada Shanren" landscape',
            '"Wang Hui" landscape',
            '"Mi Fu" landscape',
            '"Mi Youren" landscape',
            '"Xia Gui" landscape',
            '"Ma Yuan" landscape',
        ]
        category_terms = [
            "Chinese landscape paintings",
            "Landscape paintings from China",
            "Shan shui",
            "Ink paintings of China",
            "Hanging scroll paintings of China",
            "Album leaf paintings of China",
            "Paintings by Guo Xi",
            "Paintings by Fan Kuan",
            "Paintings by Huang Gongwang",
            "Paintings by Ni Zan",
            "Paintings by Wang Meng",
            "Paintings by Dong Yuan",
            "Paintings by Mi Fu",
            "Paintings by Xia Gui",
            "Paintings by Ma Yuan",
        ]
        records: list[CandidateRecord] = []
        seen_titles: set[str] = set()
        seen_source_ids: set[str] = set()

        for category in category_terms:
            titles = [title for title in self._category_members(category, file_limit=500, depth=1) if title not in seen_titles]
            seen_titles.update(titles)
            for record in self._page_info_to_records(titles):
                if record.source_id and record.source_id not in seen_source_ids:
                    seen_source_ids.add(record.source_id)
                    records.append(record)
                    if len(records) >= limit:
                        return records[:limit]

        for search_term in search_terms:
            offset = 0
            while len(records) < limit:
                try:
                    result = self.client.get_json(
                        "https://commons.wikimedia.org/w/api.php",
                        params={
                            "action": "query",
                            "format": "json",
                            "list": "search",
                            "srsearch": search_term,
                            "srnamespace": 6,
                            "srlimit": 50,
                            "sroffset": offset,
                        },
                    )
                except requests.RequestException:
                    break
                search_rows = (result.get("query") or {}).get("search") or []
                if not search_rows:
                    break
                titles = [row["title"] for row in search_rows if row.get("title") and row["title"] not in seen_titles]
                seen_titles.update(titles)
                if not titles:
                    break
                try:
                    info = self.client.get_json(
                        "https://commons.wikimedia.org/w/api.php",
                        params={
                            "action": "query",
                            "format": "json",
                        "prop": "imageinfo|categories",
                        "titles": "|".join(titles),
                        "iiprop": "url|size|extmetadata",
                        "iiurlwidth": 1600,
                        "cllimit": "max",
                    },
                )
                except requests.RequestException:
                    offset += len(search_rows)
                    continue
                pages = (info.get("query") or {}).get("pages") or {}
                for page in pages.values():
                    image_info = (page.get("imageinfo") or [{}])[0]
                    ext = image_info.get("extmetadata") or {}
                    image_url = normalize_text(image_info.get("thumburl")) or normalize_text(image_info.get("url"))
                    if not image_url:
                        continue
                    suffix = Path(image_url).suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
                        continue
                    categories = [
                        normalize_text(cat.get("title")).replace("Category:", "")
                        for cat in page.get("categories") or []
                    ]
                    record = CandidateRecord(
                        source=self.name,
                        source_id=str(page.get("pageid", "")),
                        title=normalize_text(page.get("title")),
                        artist=normalize_text((ext.get("Artist") or {}).get("value")),
                        culture=normalize_text((ext.get("ObjectName") or {}).get("value")),
                        period="",
                        dynasty="",
                        medium=normalize_text((ext.get("ImageDescription") or {}).get("value")),
                        department="Wikimedia Commons",
                        object_url=normalize_text(image_info.get("descriptionurl")),
                        image_url=image_url,
                        license=normalize_text((ext.get("LicenseShortName") or {}).get("value")) or "Wikimedia Commons",
                        tags=categories,
                        width=int(image_info.get("width") or 0),
                        height=int(image_info.get("height") or 0),
                        date_display=normalize_text((ext.get("DateTimeOriginal") or {}).get("value")),
                        extra={"categories": categories},
                    )
                    if (looks_like_chinese_landscape(record) or contains_landscape_semantics(" ".join(categories))) and (
                        record.source_id not in seen_source_ids
                    ):
                        seen_source_ids.add(record.source_id)
                        records.append(record)
                        if len(records) >= limit:
                            break
                offset += len(search_rows)
                if len(search_rows) < 50 or len(records) >= limit:
                    break
        return records[:limit]


class HarvardCollector:
    name = "harvard"

    def __init__(self, client: HTTPClient) -> None:
        self.client = client
        self.api_key = os.environ.get("HARVARD_ART_MUSEUMS_API_KEY", "").strip()

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        if not self.api_key:
            return []
        queries = [
            "landscape",
            "shan shui",
            "mountain",
            "river",
            "Guo Xi",
            "Fan Kuan",
            "Huang Gongwang",
            "Ni Zan",
        ]
        records: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        for query in queries:
            page = 1
            while len(records) < limit:
                try:
                    result = self.client.get_json(
                        "https://api.harvardartmuseums.org/object",
                        params={
                            "apikey": self.api_key,
                            "q": query,
                            "classification": "Paintings",
                            "culture": "Chinese",
                            "hasimage": 1,
                            "size": 100,
                            "page": page,
                        },
                    )
                except requests.RequestException:
                    break
                for item in result.get("records") or []:
                    object_id = str(item.get("id", ""))
                    if not object_id or object_id in seen_ids:
                        continue
                    primary_image = ""
                    images = item.get("images") or []
                    if images:
                        primary_image = normalize_text(images[0].get("baseimageurl"))
                    if not primary_image:
                        continue
                    seen_ids.add(object_id)
                    record = CandidateRecord(
                        source=self.name,
                        source_id=object_id,
                        title=normalize_text(item.get("title")),
                        artist=normalize_text(item.get("people", [{}])[0].get("displayname")) if item.get("people") else "",
                        culture=normalize_text(item.get("culture")),
                        period=normalize_text(item.get("period")),
                        dynasty=normalize_text(item.get("dated")),
                        medium=normalize_text(item.get("medium")),
                        department=normalize_text(item.get("division")),
                        object_url=normalize_text(item.get("url")),
                        image_url=primary_image,
                        license=normalize_text(item.get("copyright")) or "Harvard scholarly use",
                        tags=[normalize_text(item.get("classification"))] if normalize_text(item.get("classification")) else [],
                        width=int(images[0].get("width") or 0) if images else 0,
                        height=int(images[0].get("height") or 0) if images else 0,
                        date_display=normalize_text(item.get("dated")),
                        extra={"verificationlevel": normalize_text(item.get("verificationleveldescription"))},
                    )
                    if looks_like_chinese_landscape(record):
                        records.append(record)
                        if len(records) >= limit:
                            break
                if page >= int(result.get("info", {}).get("pages") or page):
                    break
                page += 1
        return records[:limit]


class WaltersCollector:
    name = "walters"
    ART_URL = "https://raw.githubusercontent.com/WaltersArtMuseum/api-thewalters-org/main/art.csv"
    MEDIA_URL = "https://raw.githubusercontent.com/WaltersArtMuseum/api-thewalters-org/main/media.csv"

    def __init__(self, client: HTTPClient) -> None:
        self.client = client

    def _load_csv(self, url: str) -> list[dict[str, str]]:
        response = self.client.session.get(url, timeout=self.client.timeout)
        response.raise_for_status()
        return list(csv.DictReader(StringIO(response.text)))

    def collect(self, limit: int = 2000) -> list[CandidateRecord]:
        try:
            art_rows = self._load_csv(self.ART_URL)
            media_rows = self._load_csv(self.MEDIA_URL)
        except requests.RequestException:
            return []

        primary_media: dict[str, str] = {}
        for row in media_rows:
            if normalize_text(row.get("MediaType")).lower() != "image":
                continue
            object_id = normalize_text(row.get("ObjectID"))
            image_url = normalize_text(row.get("ImageURL"))
            if not object_id or not image_url:
                continue
            rank = int(normalize_text(row.get("Rank")) or "999")
            existing = primary_media.get(object_id)
            if existing is None or rank == 1 or normalize_text(row.get("IsPrimary")) == "1":
                primary_media[object_id] = image_url

        records: list[CandidateRecord] = []
        for item in art_rows:
            object_id = normalize_text(item.get("ObjectID"))
            image_url = primary_media.get(object_id, "")
            if not object_id or not image_url:
                continue
            record = CandidateRecord(
                source=self.name,
                source_id=object_id,
                title=normalize_text(item.get("Title")),
                artist=normalize_text(item.get("Creators")),
                culture=normalize_text(item.get("Culture")),
                period=normalize_text(item.get("Period")),
                dynasty=normalize_text(item.get("Dynasty")),
                medium=normalize_text(item.get("Medium")),
                department=normalize_text(item.get("Classification")),
                object_url=normalize_text(item.get("ResourceURL")),
                image_url=image_url,
                license="CC0",
                tags=[normalize_text(item.get("Keywords")), normalize_text(item.get("Style"))],
                date_display=normalize_text(item.get("DateText")),
                extra={
                    "object_name": normalize_text(item.get("ObjectName")),
                    "description": normalize_text(item.get("Description")),
                    "collection_id": normalize_text(item.get("CollectionID")),
                },
            )
            if looks_like_chinese_landscape(record):
                records.append(record)
                if len(records) >= limit:
                    break
        return records[:limit]


COLLECTOR_REGISTRY = {
    "met": MetCollector,
    "cma": ClevelandCollector,
    "artic": ArtICCollector,
    "wikimedia": WikimediaCollector,
    "harvard": HarvardCollector,
    "walters": WaltersCollector,
}
