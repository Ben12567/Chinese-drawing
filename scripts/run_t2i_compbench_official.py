from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


TASK_TO_COMMAND = {
    "color": ["BLIPvqa_eval", "BLIP_vqa.py", "--out_dir", "../examples/"],
    "shape": ["BLIPvqa_eval", "BLIP_vqa.py", "--out_dir", "../examples/"],
    "texture": ["BLIPvqa_eval", "BLIP_vqa.py", "--out_dir", "../examples/"],
    "spatial": ["UniDet_eval", "2D_spatial_eval.py"],
    "3d_spatial": ["UniDet_eval", "3D_spatial_eval.py"],
    "numeracy": ["UniDet_eval", "numeracy_eval.py"],
    "non_spatial": ["CLIPScore_eval", "CLIP_similarity.py", "--outpath", "examples/"],
    "complex": ["3_in_1_eval", "3_in_1.py", "--outpath", "../examples/"],
}


def main(repo_path: str, task: str, outpath: str | None, complex_mode: bool) -> None:
    if task not in TASK_TO_COMMAND:
        raise ValueError(f"Unsupported task: {task}")
    repo_root = Path(repo_path).resolve()
    parts = TASK_TO_COMMAND[task]
    workdir = repo_root / parts[0]
    cmd = ["python", parts[1], *parts[2:]]
    if outpath:
        outpath = str(Path(outpath).resolve())
        if task in {"non_spatial", "complex"}:
            if "--outpath" in cmd:
                idx = cmd.index("--outpath")
                cmd[idx + 1] = outpath
            else:
                cmd.extend(["--outpath", outpath])
        elif "--out_dir" in cmd:
            idx = cmd.index("--out_dir")
            cmd[idx + 1] = outpath
    if complex_mode and task == "non_spatial":
        cmd.extend(["--complex", "True"])
    subprocess.run(cmd, cwd=str(workdir), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=".tmp/t2i_compbench")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_TO_COMMAND),
        required=True,
    )
    parser.add_argument("--outpath")
    parser.add_argument("--complex", action="store_true")
    args = parser.parse_args()
    main(repo_path=args.repo_path, task=args.task, outpath=args.outpath, complex_mode=args.complex)
