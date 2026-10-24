# SPDX-License-Identifier: Apache-2.0

"""Print `okc leaderboard --json` and keep a redacted copy under state/raw/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show an okc leaderboard.")
    parser.add_argument("problem", nargs="?", choices=common.PROBLEMS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if common.HALT_PATH.exists():
        print(common.HALT_PATH.read_text().strip(), file=sys.stderr)
        return 2
    command = [str(common.okc_executable()), "leaderboard"]
    if args.problem:
        command.append(args.problem)
    if args.all:
        command.append("--all")
    command.append("--json")
    completed = subprocess.run(
        command,
        cwd=common.ROOT,
        env=common.okc_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    transcript = common.redact(completed.stdout + "\n" + completed.stderr)
    stamp = common.utc_now().strftime("%Y%m%dT%H%M%SZ")
    name = args.problem or "overall"
    raw_path = common.RAW / f"leaderboard-{name}-{stamp}.log"
    raw_path.write_text(transcript)
    text = completed.stdout.strip()
    if not text:
        print(transcript, file=sys.stderr)
        return completed.returncode or 1
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        print(common.redact(text))
        return completed.returncode or 1
    print(json.dumps(payload, indent=2))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
