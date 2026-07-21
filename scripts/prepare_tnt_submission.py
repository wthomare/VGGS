import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

INTERMEDIATE_SCENES = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]
ADVANCED_SCENES = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]


def scenes_for_group(group):
    if group == "intermediate":
        return INTERMEDIATE_SCENES
    if group == "advanced":
        return ADVANCED_SCENES
    return INTERMEDIATE_SCENES + ADVANCED_SCENES


def find_best_trial(scene_output):
    report = scene_output / "report" / "trials.json"
    if not report.exists():
        candidates = sorted(scene_output.glob("optuna_trial_*/point_cloud/iteration_3000/point_cloud.ply"))
        if not candidates:
            raise FileNotFoundError(f"No trial point cloud found under {scene_output}")
        return candidates[-1].parents[2]
    data = json.loads(report.read_text())
    records = data.get("trials", data if isinstance(data, list) else [])
    complete = [r for r in records if r.get("value") is not None]
    if not complete:
        raise RuntimeError(f"No completed trials in {report}")
    best = max(complete, key=lambda r: r["value"])
    number = int(best["number"])
    return scene_output / f"optuna_trial_{number:04d}"


def strip_gaussian_ply(src, dst):
    ply = PlyData.read(src)
    vertices = ply["vertex"].data
    names = vertices.dtype.names
    out_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    out = np.empty(len(vertices), dtype=out_dtype)
    out["x"] = vertices["x"]
    out["y"] = vertices["y"]
    out["z"] = vertices["z"]
    if {"red", "green", "blue"}.issubset(names):
        out["red"] = vertices["red"]
        out["green"] = vertices["green"]
        out["blue"] = vertices["blue"]
    else:
        out["red"] = 255
        out["green"] = 255
        out["blue"] = 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(dst)


def main():
    parser = argparse.ArgumentParser(description="Assemble Tanks and Temples upload folder from VGGS outputs.")
    parser.add_argument("--group", choices=["intermediate", "advanced", "both"], default="intermediate")
    parser.add_argument("--data-root", default="data/tnt_official")
    parser.add_argument("--output-root", default="exp/tnt_official")
    parser.add_argument("--submission-dir", default="submission_tnt")
    parser.add_argument("--iteration", type=int, default=3000)
    args = parser.parse_args()

    submission = Path(args.submission_dir)
    submission.mkdir(parents=True, exist_ok=True)
    for scene in scenes_for_group(args.group):
        group = "intermediate" if scene in INTERMEDIATE_SCENES else "advanced"
        scene_output = Path(args.output_root) / group / scene
        trial = find_best_trial(scene_output)
        src_ply = trial / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
        src_log = Path(args.data_root) / group / scene / f"{scene}.log"
        if not src_ply.exists():
            raise FileNotFoundError(src_ply)
        if not src_log.exists():
            raise FileNotFoundError(src_log)
        strip_gaussian_ply(src_ply, submission / f"{scene}.ply")
        shutil.copy2(src_log, submission / f"{scene}.log")
        print(f"Prepared {scene}.ply and {scene}.log")

    print(f"Submission folder ready: {submission}")
    print("Copy upload_t2_results.py and your credentials file into that folder, then run the official uploader there.")


if __name__ == "__main__":
    main()
