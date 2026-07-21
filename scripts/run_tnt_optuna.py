import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENES = ["Barn", "Caterpillar", "Ignatius", "Truck"]
DEFAULT_COMMON_ARGS = (
    "-r2 --ncc_scale 0.5 --data_device cuda "
    "--densify_abs_grad_threshold 0.00015 "
    "--opacity_cull_threshold 0.05 --exposure_compensation"
)


def scene_command(args, scene):
    source = Path(args.data_root) / scene
    model_root = Path(args.output_root) / scene
    storage = model_root / f"{scene.lower()}_optuna.db"
    report_dir = model_root / "report"

    cmd = [
        sys.executable,
        "train_optuna.py",
        "-s",
        str(source),
        "-m",
        str(model_root),
        "--config",
        args.config,
        "--n-trials",
        str(args.n_trials),
        "--test-iteration",
        str(args.test_iteration),
        "--metric-split",
        args.metric_split,
        "--sampler",
        args.sampler,
        "--pruner",
        args.pruner,
        "--storage",
        f"sqlite:///{storage}",
        "--study-name",
        f"vggs-tnt-{scene.lower()}",
        "--report-dir",
        str(report_dir),
        "--common-args",
        args.common_args,
    ]
    return cmd


def validate_scene(path, scene):
    required = ["images", "depth_vggt", "normal", "sparse"]
    missing = [name for name in required if not (path / scene / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path / scene} is missing: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Run VGGS Optuna tuning on preprocessed Tanks and Temples scenes.")
    parser.add_argument("--data-root", default="data/tnt_dataset/tnt_10views")
    parser.add_argument("--output-root", default="exp/optuna/tnt_10views")
    parser.add_argument("--config", default="configs/tnt_10views.yaml")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--test-iteration", type=int, default=3000)
    parser.add_argument("--metric-split", choices=["train", "test"], default="train")
    parser.add_argument("--sampler", choices=["tpe", "random", "cmaes"], default="tpe")
    parser.add_argument("--pruner", choices=["none", "median", "hyperband"], default="none")
    parser.add_argument("--common-args", default=DEFAULT_COMMON_ARGS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    for scene in args.scenes:
        validate_scene(data_root, scene)
        cmd = scene_command(args, scene)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(printable, flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
