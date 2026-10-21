from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(repo_path: str, question_answer_path: str, id2img_path: str, output_path: str, vqa_model: str) -> None:
    repo_root = Path(repo_path).resolve()
    sys.path.insert(0, str(repo_root))
    from tifascore import tifa_score_benchmark

    result = tifa_score_benchmark(vqa_model, question_answer_path, id2img_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=".tmp/tifa")
    parser.add_argument("--question-answer-path", required=True)
    parser.add_argument("--id2img-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--vqa-model", default="vilt")
    args = parser.parse_args()
    main(
        repo_path=args.repo_path,
        question_answer_path=args.question_answer_path,
        id2img_path=args.id2img_path,
        output_path=args.output_path,
        vqa_model=args.vqa_model,
    )
