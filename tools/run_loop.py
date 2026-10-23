"""Resumable planner / gatekeeper / status loop.

Generators are not spawned here. A cycle reads notes and results, assigns up
to three open hypotheses, lints any variant files already on disk, and prints
the status table. It never calls `okc submit` or `okc check`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import lint_kernel
import parse_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Status and planning for the okc loop.")
    parser.add_argument(
        "command",
        choices=("status", "plan", "cycle", "router", "refresh-limits", "analyze", "lint"),
    )
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    if args.command == "lint":
        if args.path is None:
            print("usage: python tools/run_loop.py lint <variant.py>", file=sys.stderr)
            return 2
        report = lint_kernel.lint_path(args.path)
        print(lint_kernel.format_report(args.path, report))
        return 0 if report.ok else 1
    if args.command == "refresh-limits":
        return refresh_limits()
    if args.command == "router":
        print(format_router(common.load_results()))
        return 0
    if args.command == "analyze":
        return analyze()
    if args.command == "status":
        print(format_status())
        return 0
    plan = build_plan()
    common.save_json(common.PLAN_PATH, plan)
    loop = common.load_loop()
    loop["last_cycle_at"] = common.utc_now().isoformat()
    loop["stage"] = plan["stage"]
    loop["next_action"] = plan["next_action"]
    common.save_json(common.LOOP_PATH, loop)
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
    for item in plan["lint"]:
        if not item["ok"]:
            print(f"lint failed {item['file']}: {'; '.join(item['errors'])}")
    print(format_status())
    return 0


def refresh_limits() -> int:
    import subprocess

    command = [str(common.okc_executable()), "problems", "--json"]
    completed = subprocess.run(
        command,
        cwd=common.ROOT,
        env=common.okc_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    transcript = common.redact(completed.stdout + "\n" + completed.stderr)
    if completed.returncode != 0:
        print(transcript, file=sys.stderr)
        parsed = parse_result.parse_output(transcript)
        if parsed["halt"]:
            common.HALT_PATH.write_text(parsed["message"] + "\n")
            print(parsed["message"], file=sys.stderr)
            return 2 if parsed["status"] == "logged_out" else 3
        return completed.returncode or 1
    payload = json.loads(completed.stdout)
    parsed = common.parse_problems_payload(payload)
    attempts = common.load_attempts()
    now = common.utc_now()
    attempts["limits_fetched_at"] = now.isoformat()
    attempts["source"] = "okc problems --json"
    for name, blocks in parsed.items():
        if name not in attempts["problems"]:
            continue
        for kind in ("scored", "checks"):
            remote = blocks[kind]
            local = attempts["problems"][name][kind]
            if remote["per_day"] is not None:
                local["per_day"] = remote["per_day"]
            local["remaining_server"] = remote["remaining_server"]
            if remote["used_server"] is not None:
                local["used_server"] = remote["used_server"]
    common.save_json(common.ATTEMPTS_PATH, attempts)
    print(f"refreshed limits for {', '.join(sorted(parsed))}")
    return 0


def build_plan() -> dict:
    config = common.load_config()
    problem = config["active_problem"]
    now = common.utc_now()
    open_window, window_why = common.window_open(now, config)
    notes_path = common.NOTES / f"{problem}.md"
    hypotheses = common.parse_hypotheses(notes_path.read_text()) if notes_path.exists() else []
    open_items = [item for item in hypotheses if item.get("status", "open") == "open"]
    assignments = []
    for item in open_items[: int(config["max_parallel_generators"])]:
        assignments.append({
            "title": item["title"],
            "generator": item.get("generator", "unspecified"),
            "expected_gain": item.get("expected_gain", ""),
            "risk": item.get("risk", ""),
            "file": _variant_for(problem, item["title"]),
        })
    variants = sorted((common.VARIANTS / problem).glob("*.py"))
    lint = []
    for path in variants:
        report = lint_kernel.lint_path(path)
        lint.append({"file": str(path.relative_to(common.ROOT)), "ok": report.ok, "errors": report.errors})
    results = [row for row in common.load_results() if row.get("problem") == problem]
    misses = _consecutive_misses(results, float(config["min_improvement_to_continue"]))
    stage, action = _next_action(
        config, problem, open_window, window_why, variants, misses, results,
    )
    return {
        "problem": problem,
        "generated_at": now.isoformat(),
        "stage": stage,
        "next_action": action,
        "window": window_why,
        "assignments": assignments,
        "lint": lint,
        "consecutive_misses": misses,
        "best_speedup": common.best_speedup(problem, results),
        "phase2_enabled": bool(config.get("phase2_enabled")),
    }


def _variant_for(problem: str, title: str) -> str | None:
    parts = title.split()
    slug = parts[1].lower() if len(parts) > 1 else parts[0].lower()
    folder = common.VARIANTS / problem
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob("*.py")):
        if slug in path.name.lower():
            return str(path.relative_to(common.ROOT))
    return None


def _consecutive_misses(results: list[dict], margin: float) -> int:
    """Tail of passing scored runs that missed the then-current best by `margin`.

    A 1% bump still becomes the best we keep, and it still counts as a miss
    against the 2% bar. The counter resets only when a run clears that bar.
    """
    passed = [
        row for row in results
        if row.get("status") == "passed"
        and row.get("kind", "scored") == "scored"
        and row.get("speedup") is not None
    ]
    if not passed:
        return 0
    best = float(passed[0]["speedup"])
    misses = 0
    for row in passed[1:]:
        speed = float(row["speedup"])
        if speed > best * (1.0 + margin):
            best = speed
            misses = 0
            continue
        misses += 1
        if speed > best:
            best = speed
    return misses


def _next_action(config, problem, open_window, window_why, variants, misses, results) -> tuple[str, str]:
    if config.get("phase2_enabled"):
        return "blocked", "phase 2 is enabled in config; this loop will not start it"
    if not open_window:
        return "stopped", window_why
    if common.HALT_PATH.exists():
        return "halted", common.HALT_PATH.read_text().strip()
    loop = common.load_loop()
    if loop.get("stage") == "awaiting_login" or "not signed in" in str(loop.get("login", "")):
        return "awaiting_login", loop.get("next_action") or "Ready for okc login?"
    if not (common.ROOT / problem).is_dir():
        return "need_init", f"run okc init {problem} and re-read every file before writing a kernel"
    if not variants:
        return "need_baseline", (
            f"copy the init submission.py to variants/{problem}/v001_baseline_copy.py unchanged, "
            "lint it, then wait for OK before okc check"
        )
    if not results:
        return "need_measurement", "no logged run yet. Wait for OK before the first okc check"
    if misses >= int(config["stop_after_consecutive_misses"]):
        return "plateau", (
            f"{misses} passing runs did not beat the best by more than "
            f"{config['min_improvement_to_continue']:.0%}. Stop this problem and ask before continuing."
        )
    best = common.best_speedup(problem, results)
    shown = f"{best:.3f}x" if best is not None else "none"
    return "iterate", f"best scored speedup {shown}. Pick the next open hypothesis. Do not submit without OK."


def format_status() -> str:
    config = common.load_config()
    attempts = common.load_attempts()
    results = common.load_results()
    now = common.utc_now()
    headers = ("problem", "best speedup", "attempts used/limit", "last result", "next action")
    rows = []
    for problem in common.PROBLEMS:
        best = common.best_speedup(problem, results)
        last = common.last_result(problem, results)
        block = attempts["problems"][problem]
        scored_limit = block["scored"]["per_day"]
        check_limit = block["checks"]["per_day"]
        scored_used = common.used_on(problem, "scored", now, attempts)
        check_used = common.used_on(problem, "checks", now, attempts)
        reserve = config["reserve_fraction"]
        scored_cap, scored_spend = "?", "?"
        check_cap, check_spend = "?", "?"
        if scored_limit is not None:
            scored_cap = str(scored_limit)
            scored_spend = str(common.spendable(int(scored_limit), reserve))
        if check_limit is not None:
            check_cap = str(check_limit)
            check_spend = str(common.spendable(int(check_limit), reserve))
        last_text = "none" if last is None else f"{last.get('kind', 'submit')} {last.get('status')}"
        if problem == config["active_problem"]:
            action = common.load_loop().get("next_action") or "see plan"
        elif block.get("status") == "not_started":
            action = "not started"
        else:
            action = block.get("status", "")
        rows.append((
            problem,
            "—" if best is None else f"{best:.3f}x",
            f"scored {scored_used}/{scored_cap} (spend {scored_spend}), checks {check_used}/{check_cap} (spend {check_spend})",
            last_text,
            action,
        ))
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    milestone = datetime.fromisoformat(config["milestone_pt"])
    if now >= milestone and common.best_speedup(config["active_problem"], results) is None:
        lines.append(f"past milestone {milestone.isoformat()} with no measured speedup")
    if not common.window_open(now, config)[0]:
        lines.append(common.window_open(now, config)[1])
    return "\n".join(lines)


def format_router(results: list[dict]) -> str:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        key = (row.get("problem") or "?", row.get("generator") or "unspecified")
        groups.setdefault(key, []).append(row)
    if not groups:
        return "problem | generator | runs | passed | best speedup\n(no results yet)"
    lines = ["problem | generator | runs | passed | best speedup"]
    for (problem, generator), rows in sorted(groups):
        passed = [row for row in rows if row.get("status") == "passed" and row.get("speedup") is not None]
        best = max((float(row["speedup"]) for row in passed), default=None)
        best_text = "—" if best is None else f"{best:.3f}x"
        lines.append(f"{problem} | {generator} | {len(rows)} | {len(passed)} | {best_text}")
    return "\n".join(lines)


def analyze() -> int:
    config = common.load_config()
    problem = config["active_problem"]
    last = common.last_result(problem)
    if last is None:
        print(f"no results for {problem}")
        return 0
    job = last.get("job") or last.get("hash", "")[:12]
    notes_path = common.NOTES / f"{problem}.md"
    existing = notes_path.read_text() if notes_path.exists() else f"# {problem}\n"
    marker = f"<!-- result:{job} -->"
    if marker in existing:
        print(f"{marker} already in {notes_path}")
        return 0
    best = common.best_speedup(problem)
    best_text = "none" if best is None else f"{best:.3f}x"
    speed = "none" if last.get("speedup") is None else f"{last['speedup']:.3f}x"
    entry = (
        f"\n{marker}\n"
        f"- {last.get('variant_id')} ({last.get('generator')}) {last.get('kind')} {last.get('status')}, "
        f"speedup {speed}, tests {last.get('tests_passed')}/{last.get('tests_total')}.\n"
        f"- Best scored speedup is {best_text}. {last.get('message') or ''}\n"
        f"- Next: read the transcript before writing another variant. Do not resubmit this hash.\n"
    )
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(existing.rstrip() + "\n" + entry)
    print(f"appended post-mortem to {notes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
