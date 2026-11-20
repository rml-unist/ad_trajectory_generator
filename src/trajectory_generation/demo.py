"""CPU-only synthetic reference generation, available after wheel installation."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from trajectory_generation.config import TrajectoryGenerationConfig
from trajectory_generation.reporting import read_json, summarize, write_json
from trajectory_generation.synthetic import CASES, STATE_COLUMNS, run_case
from trajectory_generation.window_validation import check_windows


def plot_case(destination, numeric, config):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for key, label, color, style in (("raw", "Input", "black", "--"),
                                     ("retimed", "Retimed target", "tab:blue", ":"),
                                     ("reference", "QP reference model", "tab:red", "-")):
        values = np.asarray(numeric[key])
        seconds = (values[:, 0] - values[0, 0]) * 1e-6
        axes[0].plot(values[:, 1], values[:, 2], style, color=color, label=label)
        axes[1].plot(seconds, values[:, 4], style, color=color, label=label)
        axes[2].plot(seconds, values[:, 6], style, color=color, label=label)
    axes[0].set(xlabel="X [m]", ylabel="Y [m]", title="Rear-axle path")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(xlabel="Time [s]", ylabel="Longitudinal speed [m/s]", title="Speed profile")
    axes[2].set(xlabel="Time [s]", ylabel="Longitudinal acceleration [m/s^2]", title="Acceleration")
    axes[2].axhline(config.kinematic.max_accel_mps2, color="0.5", linewidth=1,
                   linestyle="--", label="Nominal target cap")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"Synthetic {numeric['case']} | QP refresh {config.kmpc_qp.reference_update_steps * .1:.1f} s")
    figure.savefig(destination / "trajectory.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="straight")
    parser.add_argument("--config", type=Path, help="Use a JSON configuration instead of the generator defaults")
    parser.add_argument("--output", type=Path, help="New directory; existing results are never overwritten")
    parser.add_argument("--plot", action="store_true", help="Requires the plot extra and --output")
    args = parser.parse_args(argv)
    if args.plot and args.output is None:
        parser.error("--plot requires --output")
    created = False
    try:
        config = (TrajectoryGenerationConfig.from_dict(read_json(args.config))
                  if args.config else TrajectoryGenerationConfig())
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            created = True
            write_json(args.output / "status.json", {"status": "running"})
        versions = {}
        for name in ("trajectory-generator", "nuplan-devkit", "numpy", "scipy", "osqp", "shapely"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "source-on-PYTHONPATH"
        summaries = []
        for kind in CASES if args.case == "all" else (args.case,):
            numeric, diagnostics = run_case(config, kind)
            check_windows(diagnostics["windows"], config, 0)
            summary = summarize(kind, config, diagnostics, versions,
                final_time_us=numeric["reference"][-1][0], fallback=numeric["fallback"],
                applied_mode=numeric["mode"])
            if args.output:
                destination = args.output / kind
                destination.mkdir()
                for key in ("raw", "retimed", "reference"):
                    with (destination / (key + ".csv")).open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(STATE_COLUMNS)
                        writer.writerows(numeric[key])
                write_json(destination / "config.json", config.to_dict())
                write_json(destination / "diagnostics.json", diagnostics)
                write_json(destination / "summary.json", summary)
                if args.plot:
                    plot_case(destination, numeric, config)
            summaries.append(summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
        if args.output:
            write_json(args.output / "summary.json", summaries)
            write_json(args.output / "status.json", {"status": "completed", "cases": len(summaries)})
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
        if created:
            write_json(args.output / "status.json", {"status": "failed", "error": str(exc)})
        print("ERROR | " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
