"""Read and validate saved synthetic trajectories, diagnostics and summaries."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import TrajectoryGenerationConfig
from .synthetic import CASES, STATE_COLUMNS
from .window_validation import check_windows

SCHEMA = "trajectory_generator_result_v1"
SERIES = ("raw", "retimed", "reference")


def write_json(path, value, *, exclusive=False):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(text)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON value")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    _finite(value)
    return value


def summarize(case, config, diagnostics, versions, *, final_time_us, fallback, applied_mode):
    windows = diagnostics["windows"]
    elapsed = [w["generator_elapsed_s"] for w in windows]
    return dict(case=case, synthetic=True, status="passed", windows=len(windows),
        qp_refreshes=sum(w["qp_refreshed"] for w in windows), fallback=fallback,
        final_time_us=final_time_us, applied_mode=applied_mode,
        versions=versions,
        preparation_s=diagnostics["preparation_s"], generation_mean_s=float(np.mean(elapsed)),
        generation_p95_s=float(np.percentile(elapsed, 95)), generation_max_s=float(max(elapsed)),
        max_model_residual=max(w["max_model_residual"] for w in windows),
        wall_time_limits_enforced=config.kmpc_qp.enforce_wall_time_limits,
        note="Synthetic generator reference-model rollout")


def _result_files(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Result root must be a directory, not a symlink")
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Unsupported result entry: " + str(path))
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def read_series(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != STATE_COLUMNS:
            raise ValueError("CSV header mismatch: " + str(path))
        values = np.asarray([[float(v) for v in row] for row in reader], dtype=np.float64)
    if values.shape != (81, len(STATE_COLUMNS)) or not np.isfinite(values).all():
        raise ValueError("CSV must contain 81 finite states: " + str(path))
    if not np.array_equal(values[:, 0], np.arange(81) * 100000):
        raise ValueError("CSV timestamps must cover 0..8 seconds at 0.1 s: " + str(path))
    return values


def _same_json(actual, expected, label):
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Inconsistent " + label)


def check_result(root):
    root = Path(root)
    files = _result_files(root)
    index = read_json(root / "summary.json")
    if (not isinstance(index, list) or not index
            or any(not isinstance(row, dict) or row.get("case") not in CASES for row in index)):
        raise ValueError("Invalid result cases in summary.json")
    cases = [row["case"] for row in index]
    if len(set(cases)) != len(cases):
        raise ValueError("Duplicate result case in summary.json")
    required = {"status.json", "summary.json"}
    for case in cases:
        required.update(case + "/" + name for name in (
            "raw.csv", "retimed.csv", "reference.csv", "config.json", "diagnostics.json", "summary.json"))
    optional = {case + "/trajectory.png" for case in cases}
    if not required.issubset(files) or files - required - optional:
        raise ValueError("Missing or unexpected result files")
    _same_json(read_json(root / "status.json"), dict(status="completed", cases=len(cases)), "completion status")
    summaries = []
    for row in index:
        case = row["case"]
        destination = root / case
        try:
            recorded = read_json(destination / "config.json")
            config = TrajectoryGenerationConfig.from_dict(recorded)
            active = {k: v for k, v in recorded.items() if k not in ("estimator", "kmpc")}
            _same_json(active, config.to_dict(), "fully resolved generator configuration")
            numeric = {key: read_series(destination / (key + ".csv")) for key in SERIES}
            for key in SERIES[1:]:
                if not np.array_equal(numeric[key][0], numeric["raw"][0]):
                    raise ValueError("Initial state mismatch: " + key)
            diagnostics = read_json(destination / "diagnostics.json")
            check_windows(diagnostics["windows"], config, 0)
            for value in [diagnostics["preparation_s"], *[
                    w["generator_elapsed_s"] for w in diagnostics["windows"]]]:
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid generation timing")
            summary = read_json(destination / "summary.json")
            _same_json(summary, row, "root/case summaries")
            versions = summary["versions"]
            if not isinstance(versions, dict) or not versions or any(
                    not isinstance(v, str) or not v for v in versions.values()):
                raise ValueError("Invalid recorded dependency versions")
            expected = summarize(case, config, diagnostics, versions,
                final_time_us=int(numeric["reference"][-1, 0]), fallback=0, applied_mode=config.mode)
            _same_json(summary, expected, case + " summary/diagnostics/config")
            summaries.append(summary)
        except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Malformed case result: " + case) from exc
    return dict(status="passed", schema=SCHEMA, version="1.0.0", cases=len(cases),
                files=len(files), summaries=summaries)


def separate_output(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the verified input directory")
    if destination.exists():
        raise FileExistsError("Output already exists: " + str(destination))
    return destination


def replot_result(source, output):
    from .demo import plot_case

    source = Path(source)
    report = check_result(source)
    output = separate_output(source, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", dict(status="running"))
    try:
        for summary in report["summaries"]:
            case = summary["case"]
            destination = output / case
            destination.mkdir()
            numeric = {key: read_series(source / case / (key + ".csv")) for key in SERIES}
            numeric["case"] = case
            config = TrajectoryGenerationConfig.from_dict(read_json(source / case / "config.json"))
            plot_case(destination, numeric, config)
        write_json(output / "status.json", dict(status="completed", cases=report["cases"]))
    except Exception as exc:
        write_json(output / "status.json", dict(status="failed", error=str(exc)))
        raise
    return report
