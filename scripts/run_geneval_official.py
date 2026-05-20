from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main(repo_path: str, image_dir: str, output_path: str, model_path: str, extra_options: list[str]) -> None:
    repo_root = Path(repo_path).resolve()
    cmd = [
        "python",
        str(repo_root / "evaluation" / "evaluate_images.py"),
        image_dir,
        "--outfile",
        output_path,
        "--model-path",
        model_path,
    ]
    if extra_options:
        cmd.extend(["--options", *extra_options])
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=".tmp/geneval_repo")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--options", nargs="*", default=[])
    args = parser.parse_args()
    main(
        repo_path=args.repo_path,
        image_dir=args.image_dir,
        output_path=args.output_path,
        model_path=args.model_path,
        extra_options=args.options,
    )
