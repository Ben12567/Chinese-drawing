"""Artist overlap between splits after normalising the artist field (reported in Methods and Limitations).

The split groups paintings by the artist field as recorded in the source metadata. That field is not
normalised across sources (HTML markup from Wikimedia, life dates, qualifiers such as "attributed to"),
so the same artist can occur in several splits. This script normalises the names and counts the test
paintings whose artist also has paintings in the training split.

Usage: python scripts/artist_overlap.py  -> outputs/revision/audit/artist_overlap.json
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/clp_v7/manifest.jsonl"
OUT = ROOT / "outputs/revision/audit/artist_overlap.json"
ANONYMOUS = {"", "unknown", "anonymous", "unidentified artist", "unidentified", "artist unknown", "unknown artist",
             "unknown author unknown author", "unknown author"}
QUALIFIERS = r"\b(attributed to|attributed|style of|after|follower of|workshop of|circle of|formerly attributed to|" \
             r"traditionally attributed to|imitator of|school of)\b"


def normalise(name: str) -> str:
    s = re.sub(r"<[^>]+>", " ", name or "").lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(QUALIFIERS, " ", s)
    s = re.sub(r"\d{3,4}\s*[-–]\s*\d{2,4}|\d{3,4}", " ", s)
    s = re.sub(r"[^a-z一-鿿 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = collections.defaultdict(set)
    for r in rows:
        splits[normalise(r["painter"])].add(r["split"])
    test = [r for r in rows if r["split"] == "test"]
    anonymous = [r for r in test if normalise(r["painter"]) in ANONYMOUS]
    shared = [r for r in test if normalise(r["painter"]) not in ANONYMOUS and "train" in splits[normalise(r["painter"])]]
    result = {
        "n_test": len(test),
        "n_test_anonymous": len(anonymous),
        "n_test_artist_also_in_train": len(shared),
        "n_normalised_names_in_several_splits": sum(1 for k, v in splits.items() if len(v) > 1 and k not in ANONYMOUS),
        "test_samples_artist_also_in_train": sorted(r["sample_id"] for r in shared),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in result.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
