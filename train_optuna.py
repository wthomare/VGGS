import argparse
import csv
import html
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import optuna
except ImportError as exc:
    optuna = None
    OPTUNA_IMPORT_ERROR = exc
else:
    OPTUNA_IMPORT_ERROR = None


PSNR_RE = re.compile(r"Evaluating\s+(?P<split>\w+):.*?PSNR\s+(?P<psnr>[-+0-9.eE]+)")
PARAM_NAMES = [
    "depth_conf_keep_ratio",
    "depth_conf_keep_ratio_2",
    "depth_edge_keep_ratio",
    "pseudo_depth_ema",
    "loss_ramp_iters",
    "weight_depth",
    "weight_rdc",
    "weight_normal",
    "tsdf_edge_keep_ratio",
]


def ensure_output_paths(args):
    Path(args.model_root).mkdir(parents=True, exist_ok=True)
    if args.report_dir:
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    if args.storage and args.storage.startswith("sqlite:///"):
        db_path = args.storage.removeprefix("sqlite:///")
        if db_path and db_path != ":memory:":
            Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def parse_psnr(output, preferred_split):
    matches = PSNR_RE.findall(output)
    if not matches:
        raise RuntimeError("No PSNR value found in train.py output. Include test iterations that run evaluation.")

    preferred = [float(value) for split, value in matches if split == preferred_split]
    if preferred:
        return preferred[-1]
    return float(matches[-1][1])


def suggest_params(trial):
    return {
        "depth_conf_keep_ratio": trial.suggest_float("depth_conf_keep_ratio", 0.2, 0.55),
        "depth_conf_keep_ratio_2": trial.suggest_float("depth_conf_keep_ratio_2", 0.3, 0.7),
        "depth_edge_keep_ratio": trial.suggest_float("depth_edge_keep_ratio", 0.75, 0.95),
        "pseudo_depth_ema": trial.suggest_float("pseudo_depth_ema", 0.0, 0.8),
        "loss_ramp_iters": trial.suggest_int("loss_ramp_iters", 250, 900, step=50),
        "weight_depth": trial.suggest_float("weight_depth", 0.2, 1.2),
        "weight_rdc": trial.suggest_float("weight_rdc", 0.5, 3.5),
        "weight_normal": trial.suggest_float("weight_normal", 1.0, 5.0),
        "tsdf_edge_keep_ratio": trial.suggest_float("tsdf_edge_keep_ratio", 0.0, 0.95),
    }


def build_train_command(args, trial, params):
    model_path = Path(args.model_root) / f"{args.name}_trial_{trial.number:04d}"
    cmd = [
        args.python,
        "train.py",
        "-s",
        args.source,
        "-m",
        str(model_path),
        "--config",
        args.config,
        "--test_iterations",
        str(args.test_iteration),
        "--save_iterations",
        str(args.test_iteration),
    ]
    for key, value in params.items():
        if key == "tsdf_edge_keep_ratio":
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(shlex.split(args.common_args))
    return cmd


def objective(args):
    def run_trial(trial):
        params = suggest_params(trial)
        cmd = build_train_command(args, trial, params)
        command = " ".join(shlex.quote(part) for part in cmd)
        trial.set_user_attr("command", command)
        trial.set_user_attr("tsdf_edge_keep_ratio", params["tsdf_edge_keep_ratio"])

        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        trial.set_user_attr("returncode", completed.returncode)
        trial.set_user_attr("tail", completed.stdout[-4000:])
        if completed.returncode != 0:
            raise RuntimeError(f"train.py failed with return code {completed.returncode}")

        score = parse_psnr(completed.stdout, args.metric_split)
        trial.set_user_attr("psnr", score)
        return score

    return run_trial


def create_sampler(name, seed):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {name}")


def create_pruner(name):
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unsupported pruner: {name}")


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def trial_records(study):
    records = []
    for trial in study.trials:
        record = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "psnr": trial.user_attrs.get("psnr", trial.value),
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            "command": trial.user_attrs.get("command", ""),
            "returncode": trial.user_attrs.get("returncode", ""),
        }
        for name in PARAM_NAMES:
            record[name] = trial.params.get(name, trial.user_attrs.get(name, ""))
        records.append(record)
    return records


def best_trials(records, limit=5):
    complete = [record for record in records if record["value"] is not None]
    return sorted(complete, key=lambda item: item["value"], reverse=True)[:limit]


def write_csv(path, records):
    fieldnames = [
        "number",
        "state",
        "value",
        "psnr",
        "duration_seconds",
        *PARAM_NAMES,
        "returncode",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, study, records):
    best = best_trials(records, limit=1)
    payload = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_value": best[0]["value"] if best else None,
        "best_params": study.best_params if best else {},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trials": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_param_chips(record):
    chips = []
    for name in PARAM_NAMES:
        value = record.get(name)
        if value != "":
            chips.append(f"<span><b>{html.escape(name)}</b> {html.escape(format_value(value))}</span>")
    return "\n".join(chips)


def render_bars(records):
    complete = [record for record in records if record["value"] is not None]
    if not complete:
        return "<p>No completed trials yet.</p>"

    min_value = min(record["value"] for record in complete)
    max_value = max(record["value"] for record in complete)
    span = max(max_value - min_value, 1e-9)
    bars = []
    for record in sorted(complete, key=lambda item: item["number"]):
        width = 8 + 92 * ((record["value"] - min_value) / span)
        bars.append(
            f"""
            <div class="bar-row">
              <span class="bar-label">#{record['number']}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>
              <span class="bar-score">{record['value']:.4f}</span>
            </div>
            """
        )
    return "\n".join(bars)


def write_html(path, study, records, args):
    completed = [record for record in records if record["value"] is not None]
    failed = [record for record in records if record["state"] != "COMPLETE"]
    best = best_trials(records, limit=5)
    best_record = best[0] if best else None
    best_value = best_record["value"] if best_record else None
    median_value = None
    if completed:
        values = sorted(record["value"] for record in completed)
        median_value = values[len(values) // 2]

    best_cards = []
    for rank, record in enumerate(best, start=1):
        best_cards.append(
            f"""
            <article class="best-card">
              <div class="rank">#{rank}</div>
              <h3>Trial {record['number']}</h3>
              <p class="score">{record['value']:.4f} PSNR</p>
              <div class="chips">{render_param_chips(record)}</div>
            </article>
            """
        )

    complete_rank = {
        record["number"]: rank
        for rank, record in enumerate(best_trials(records, limit=len(records)), start=1)
    }
    ordered_records = sorted(
        records,
        key=lambda item: (
            item["value"] is None,
            -(item["value"] or float("-inf")),
            item["number"],
        ),
    )
    rows = []
    for record in ordered_records:
        rank = complete_rank.get(record["number"], "")
        rows.append(
            f"""
            <tr>
              <td>{rank}</td>
              <td>#{record['number']}</td>
              <td>{html.escape(record['state'])}</td>
              <td>{format_value(record['value'])}</td>
              <td>{format_value(record['duration_seconds'])}</td>
              <td><div class="chips">{render_param_chips(record)}</div></td>
              <td><code>{html.escape(record.get('command', ''))}</code></td>
            </tr>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VGGS Optuna Benchmark - {html.escape(study.study_name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101214;
      --panel: #181b1f;
      --panel-2: #20242a;
      --text: #eceff4;
      --muted: #aab2c0;
      --line: #333944;
      --accent: #78d6b5;
      --accent-2: #f0c36a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 36px 42px 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #15191d, #22272f);
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 32px; }}
    h2 {{ margin-bottom: 16px; font-size: 20px; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; max-width: 900px; }}
    main {{ padding: 28px 42px 44px; display: grid; gap: 24px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }}
    .metric {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .metric span {{ color: var(--muted); display: block; font-size: 12px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 26px; }}
    .best-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .best-card {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 16px; position: relative; }}
    .rank {{ position: absolute; top: 12px; right: 14px; color: var(--accent-2); font-weight: 700; }}
    .score {{ color: var(--accent); font-size: 22px; margin: 8px 0 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .chips span {{ background: #111418; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); padding: 4px 8px; font-size: 12px; }}
    .chips b {{ color: var(--text); font-weight: 600; }}
    .bar-row {{ display: grid; grid-template-columns: 48px 1fr 82px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-label, .bar-score {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .bar-track {{ height: 14px; border-radius: 999px; background: #0d0f12; border: 1px solid var(--line); overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1120px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    tr:hover td {{ background: #1d2228; }}
    code {{ color: #d7e0ea; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .footer {{ color: var(--muted); }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VGGS Optuna Benchmark</h1>
    <p class="subtitle">Study <b>{html.escape(study.study_name)}</b> generated {datetime.now().isoformat(timespec="seconds")}. Source: <code>{html.escape(args.source)}</code>. Config: <code>{html.escape(args.config)}</code>.</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><span>Best PSNR</span><strong>{format_value(best_value)}</strong></div>
      <div class="metric"><span>Median PSNR</span><strong>{format_value(median_value)}</strong></div>
      <div class="metric"><span>Completed Trials</span><strong>{len(completed)}</strong></div>
      <div class="metric"><span>Failed/Pruned</span><strong>{len(failed)}</strong></div>
    </section>
    <section>
      <h2>Best Trials</h2>
      <div class="best-grid">{''.join(best_cards) if best_cards else '<p>No completed trials yet.</p>'}</div>
    </section>
    <section>
      <h2>Score Timeline</h2>
      {render_bars(records)}
    </section>
    <section>
      <h2>All Trials</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Rank</th><th>Trial</th><th>State</th><th>PSNR</th><th>Seconds</th><th>Parameters</th><th>Command</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="footer">
      <p>Machine-readable exports are written next to this file as <code>trials.csv</code> and <code>trials.json</code>.</p>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def write_report(study, args):
    report_dir = Path(args.report_dir) if args.report_dir else Path(args.model_root) / f"{args.name}_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = trial_records(study)
    write_csv(report_dir / "trials.csv", records)
    write_json(report_dir / "trials.json", study, records)
    write_html(report_dir / "study_report.html", study, records, args)
    return report_dir


def main():
    parser = argparse.ArgumentParser(description="Tune VGGS train.py quality parameters with Optuna.")
    parser.add_argument("--source", "-s", required=True, help="Scene source path passed to train.py.")
    parser.add_argument("--model-root", "-m", required=True, help="Output directory root for trial runs.")
    parser.add_argument("--config", default="configs/dtu.yaml")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch train.py. Defaults to the current interpreter.")
    parser.add_argument("--name", default="optuna")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///optuna.db.")
    parser.add_argument("--study-name", default="vggs-quality")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "random", "cmaes"])
    parser.add_argument("--pruner", default="none", choices=["none", "median", "hyperband"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--test-iteration", type=int, default=3000)
    parser.add_argument("--metric-split", default="test", choices=["test", "train"])
    parser.add_argument("--common-args", default="-r2 --ncc_scale 0.5")
    parser.add_argument("--report-dir", default=None, help="Directory for HTML/CSV/JSON benchmark report.")
    parser.add_argument("--no-report", action="store_true", help="Disable benchmark report generation.")
    args = parser.parse_args()

    if optuna is None:
        raise SystemExit("Optuna is not installed. Run `uv sync --extra tuning` first.") from OPTUNA_IMPORT_ERROR

    ensure_output_paths(args)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=create_sampler(args.sampler, args.seed),
        pruner=create_pruner(args.pruner),
    )
    study.optimize(objective(args), n_trials=args.n_trials, timeout=args.timeout)

    completed = [trial for trial in study.trials if trial.value is not None]
    if completed:
        print("Best PSNR:", study.best_value)
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("Best command:")
        print(study.best_trial.user_attrs.get("command", ""))
    else:
        print("No completed trials produced a score.")

    if not args.no_report:
        report_dir = write_report(study, args)
        print("Benchmark report:")
        print(report_dir / "study_report.html")


if __name__ == "__main__":
    main()
    """
    python train_optuna.py \
    -s data/DTU/set_22_25_28/scan24/dense \
    -m exp/optuna/dtu_scan24 \
    --config configs/dtu.yaml \
    --n-trials 20 \
    --report-dir exp/optuna/dtu_scan24_report
    """
