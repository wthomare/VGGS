import argparse
import itertools
import shlex
import subprocess


def parse_values(raw):
    return [value.strip() for value in raw.split(",") if value.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run or print small VGGS quality sweeps.")
    parser.add_argument("--source", "-s", required=True, help="Scene source path passed to train.py.")
    parser.add_argument("--model-root", "-m", required=True, help="Output directory root for sweep runs.")
    parser.add_argument("--config", default="configs/dtu.yaml")
    parser.add_argument("--python", default="python")
    parser.add_argument("--name", default="quality")
    parser.add_argument("--common-args", default="--quiet -r2 --ncc_scale 0.5")
    parser.add_argument("--depth-conf-keep-ratio", default="0.25,0.35,0.45")
    parser.add_argument("--depth-edge-keep-ratio", default="0.8,0.9")
    parser.add_argument("--pseudo-depth-ema", default="0.0,0.6")
    parser.add_argument("--loss-ramp-iters", default="300,500,800")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing them.")
    args = parser.parse_args()

    grid = itertools.product(
        parse_values(args.depth_conf_keep_ratio),
        parse_values(args.depth_edge_keep_ratio),
        parse_values(args.pseudo_depth_ema),
        parse_values(args.loss_ramp_iters),
    )

    for idx, (conf_keep, edge_keep, ema, ramp) in enumerate(grid):
        run_name = (
            f"{args.name}_{idx:03d}"
            f"_conf{conf_keep}_edge{edge_keep}_ema{ema}_ramp{ramp}"
        )
        cmd = [
            args.python,
            "train.py",
            "-s",
            args.source,
            "-m",
            f"{args.model_root}/{run_name}",
            "--config",
            args.config,
            "--depth_conf_keep_ratio",
            conf_keep,
            "--depth_conf_keep_ratio_2",
            conf_keep,
            "--depth_edge_keep_ratio",
            edge_keep,
            "--pseudo_depth_ema",
            ema,
            "--loss_ramp_iters",
            ramp,
        ]
        cmd.extend(shlex.split(args.common_args))
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
