"""Copy a variant into the problem tree and run `okc check` or `okc submit`.

Dry-run unless `--approved` is set and the matching switch in state/config.json
is true. Identical file hashes are refused: TASK.md says the same code returns
its earlier result, and padding the file to force a new timing would game that.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import lint_kernel
import parse_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log an okc check or submit.")
    parser.add_argument("problem", choices=common.PROBLEMS)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--kind", choices=("check", "submit"), default="check")
    parser.add_argument("--approved", action="store_true", help="required for a real run")
    args = parser.parse_args(argv)

    config = common.load_config()
    if config.get("phase2_enabled"):
        print("phase 2 is not part of this loop", file=sys.stderr)
        return 1
    if args.problem != config["active_problem"]:
        print(
            f"active problem is {config['active_problem']}; refusing {args.problem}",
            file=sys.stderr,
        )
        return 1
    if common.HALT_PATH.exists():
        print(common.HALT_PATH.read_text().strip(), file=sys.stderr)
        return 2

    variant = args.variant.resolve()
    allowed = (common.VARIANTS / args.problem).resolve()
    if allowed not in variant.parents:
        print(f"{variant} is not under {allowed}", file=sys.stderr)
        return 1

    source = variant.read_text()
    header = common.parse_variant_header(source)
    digest = common.sha256_file(variant)
    prior = [
        row for row in common.load_results()
        if row.get("problem") == args.problem and row.get("hash") == digest
    ]
    report = lint_kernel.lint_source(source, filename=str(variant))
    print(lint_kernel.format_report(variant, report))

    now = common.utc_now()
    kind = "checks" if args.kind == "check" else "scored"
    allowed_budget, budget_why = common.budget_decision(args.problem, kind, now, config)
    enabled = bool(config["check_enabled"] if args.kind == "check" else config["submit_enabled"])
    destination = common.ROOT / args.problem / "submission.py"
    command = [
        str(common.okc_executable()),
        args.kind,
        args.problem,
        str(destination),
        "--json",
    ]
    print(f"would run: {' '.join(command)}")
    print(f"budget: {budget_why}")
    print(f"hash: {digest}")

    if not args.approved or not enabled:
        print("dry-run: pass --approved only after an explicit OK, and set the config switch")
        return 0
    if not report.ok:
        print("lint failed; nothing was sent", file=sys.stderr)
        return 1
    if not allowed_budget:
        print(f"budget refused the run: {budget_why}", file=sys.stderr)
        return 1
    if prior:
        print(
            "this exact file was already sent "
            f"({prior[-1].get('job')}). The judge returns that result again. "
            "Not resubmitting, and not editing the file to dodge the cache.",
            file=sys.stderr,
        )
        return 1
    if not destination.parent.is_dir():
        print(f"missing {destination.parent}; run okc init {args.problem} first", file=sys.stderr)
        return 1

    shutil.copyfile(variant, destination)
    completed = subprocess.run(
        command,
        cwd=common.ROOT,
        env=common.okc_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    transcript = common.redact(completed.stdout + "\n" + completed.stderr)
    raw_path = common.RAW / f"{args.kind}-{digest[:12]}.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(transcript)
    parsed = parse_result.parse_output(transcript, problem=args.problem)
    row = {
        "problem": args.problem,
        "kind": args.kind,
        "variant_id": header.get("variant") or variant.stem,
        "generator": header.get("generator") or "unspecified",
        "hash": digest,
        "speedup": parsed["speedup"],
        "time_us": parsed["time_us"],
        "tests_passed": parsed["tests_passed"],
        "tests_total": parsed["tests_total"],
        "status": parsed["status"],
        "job": parsed["job"],
        "confident": parsed["confident"],
        "notes": header.get("idea", ""),
        "message": parsed["message"],
        "at": now.isoformat(),
        "returncode": completed.returncode,
    }
    if row["job"] and any(item.get("job") == row["job"] for item in common.load_results()):
        print(f"job {row['job']} is already logged")
    else:
        common.append_result(row)
    if row["job"] or row["status"] in {"passed", "failed", "timeout"}:
        common.record_attempt(
            args.problem,
            kind,
            {
                "at": now.isoformat(),
                "kind": kind,
                "job": row["job"],
                "variant": row["variant_id"],
                "hash": digest,
                "status": row["status"],
            },
            now,
        )
    if row["status"] == "passed" and args.kind == "submit" and row["speedup"] is not None:
        _maybe_update_best(args.problem, variant, row)
    print(
        f"logged {row['status']} speedup={row['speedup']} "
        f"time_us={row['time_us']} tests={row['tests_passed']}/{row['tests_total']} "
        f"confident={row['confident']}"
    )
    if parsed["halt"]:
        common.HALT_PATH.write_text(parsed["message"] + "\n")
        print(parsed["message"], file=sys.stderr)
        return 2 if row["status"] == "logged_out" else 3
    if not parsed["confident"]:
        print("parser is not confident. Read state/raw and update tools/parse_result.py before another run.")
        return 1
    return 0 if row["status"] == "passed" else 1


def _maybe_update_best(problem: str, variant: Path, row: dict) -> None:
    current = common.best_speedup(problem)
    if current is not None and float(row["speedup"]) <= current:
        return
    dest_dir = common.BEST / problem
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(variant, dest_dir / "submission.py")
    common.save_json(
        dest_dir / "meta.json",
        {
            "variant_id": row["variant_id"],
            "generator": row["generator"],
            "hash": row["hash"],
            "speedup": row["speedup"],
            "job": row["job"],
            "at": row["at"],
        },
    )


if __name__ == "__main__":
    sys.exit(main())
