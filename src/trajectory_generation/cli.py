"""Generate reference trajectories and inspect saved results."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        from .demo import main as demo_main
        return demo_main(argv[1:])
    parser = argparse.ArgumentParser(description="Trajectory Generator")
    parser.add_argument("--version", action="version", version="1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Generate synthetic reference trajectories")
    for command in ("check", "info", "plot"):
        sub = commands.add_parser(command, help=command + " saved synthetic results")
        sub.add_argument("--input", type=Path, required=True)
        if command == "check":
            sub.add_argument("--report", type=Path, help="New JSON file outside the input directory")
        if command == "plot":
            sub.add_argument("--output", type=Path, required=True, help="New plot directory")
    args = parser.parse_args(argv)
    try:
        from .reporting import check_result, replot_result, separate_output, write_json
        if args.command == "plot":
            report = replot_result(args.input, args.output)
            print(f"PLOT PASSED | {report['cases']} cases | {args.output}")
            return 0
        report = check_result(args.input)
        if args.command == "check" and args.report:
            output = separate_output(args.input, args.report)
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, report, exclusive=True)
        print(f"CHECK PASSED | {report['cases']} cases | {report['files']} files")
        if args.command == "info":
            print("CASE | STATES | WINDOWS | QP REFRESHES | FALLBACK | GENERATION MEAN [ms]")
            for summary in report["summaries"]:
                print(f"{summary['case']} | 81 | {summary['windows']} | {summary['qp_refreshes']} | "
                      f"{summary['fallback']} | {summary['generation_mean_s'] * 1000:.3f}")
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
        print("ERROR | " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
