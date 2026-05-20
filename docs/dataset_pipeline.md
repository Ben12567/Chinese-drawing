# Dataset Pipeline

This repository now includes a full paper-oriented dataset pipeline for building `Chinese Landscape Painting-4K`.

## 1. Collect public-domain candidates

Use [collect_landscape_dataset.py](/C:/Users/Administrator/Access%20paper/scripts/collect_landscape_dataset.py) to query multiple public or open-access sources:

- The Metropolitan Museum of Art
- Cleveland Museum of Art
- Art Institute of Chicago
- Wikimedia Commons

The script stores metadata in JSONL and can optionally download all images.

## 2. Clean and deduplicate

Use [clean_landscape_dataset.py](/C:/Users/Administrator/Access%20paper/scripts/clean_landscape_dataset.py) to:

- score image quality
- flag heavy borders, blur, low-information images, and text-heavy pages
- remove perceptual near-duplicates
- copy kept and rejected images into separate directories

## 3. Build paper-ready manifest

Use [build_landscape_paper_dataset.py](/C:/Users/Administrator/Access%20paper/scripts/build_landscape_paper_dataset.py) to:

- resize kept images
- generate structure maps
- create structured prompts
- write `manifest.jsonl`
- generate grouped train/val/test splits

## Recommended Commands

```powershell
$env:PYTHONPATH = "src"

python scripts/collect_landscape_dataset.py `
  --output-manifest data/metadata/landscape_candidates.jsonl `
  --downloads-dir data/raw/landscape_candidates `
  --download-images `
  --per-source-limit 1500 `
  --sources met cma artic wikimedia

python scripts/clean_landscape_dataset.py `
  --candidate-manifest data/metadata/landscape_candidates.jsonl `
  --cleaned-dir data/interim/landscape_cleaning `
  --report-path data/metadata/landscape_cleaned_report.jsonl `
  --copy-kept

python scripts/build_landscape_paper_dataset.py `
  --cleaned-report data/metadata/landscape_cleaned_report.jsonl `
  --output-dir data/processed/clp4k_full `
  --split-name clp4k_v2 `
  --target-size 1536
```

## Practical Notes

- Expect to over-collect. To end with about 4,000 usable images, target roughly 6,000 to 8,000 candidates before cleaning.
- The rule-based collector is intentionally high recall. Final semantic verification should include manual spot checks.
- For the final paper, keep a source breakdown table and a rejection-reason table.
