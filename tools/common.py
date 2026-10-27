# SPDX-License-Identifier: Apache-2.0

"""Paths, budgets, and records shared by the okc loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
NOTES = ROOT / "notes"
STATE = ROOT / "state"
VARIANTS = ROOT / "variants"
BEST = STATE / "best"
RAW = STATE / "raw"

CONFIG_PATH = STATE / "config.json"
ATTEMPTS_PATH = STATE / "attempts.json"
RESULTS_PATH = STATE / "results.jsonl"
LOOP_PATH = STATE / "loop.json"
PLAN_PATH = STATE / "plan.json"
HALT_PATH = STATE / "HALT"

PROBLEMS = (
    "muse_bf16_proj",
    "muse_attention_decode",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
)

GENERATORS = ("triton", "cutlass_flashinfer", "raw_cuda", "pytorch")
HEADER_KEYS = ("variant", "problem", "generator", "idea", "expected_gain", "risks")
PT = ZoneInfo("America/Los_Angeles")

# Commands that print or revoke credentials. The wrappers never call these.
FORBIDDEN_OKC_COMMANDS = frozenset({"api", "logout"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_config() -> dict:
    return load_json(CONFIG_PATH)


def load_attempts() -> dict:
    return load_json(ATTEMPTS_PATH)


def load_loop() -> dict:
    return load_json(LOOP_PATH)


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    rows = []
    for line in RESULTS_PATH.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_result(row: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def redact(text: str) -> str:
    text = re.sub(r"ghp_[A-Za-z0-9]+", "[redacted]", text)
    text = re.sub(r"github_pat_[A-Za-z0-9_]+", "[redacted]", text)
    text = re.sub(r"OKC_TOKEN=\S+", "OKC_TOKEN=[redacted]", text)
    return text


def okc_executable() -> Path:
    found = shutil.which("okc")
    if found:
        return Path(found)
    fallback = Path.home() / ".okc" / "bin" / "okc"
    if fallback.is_file():
        return fallback
    raise SystemExit("okc is not on PATH and ~/.okc/bin/okc is missing")


def okc_env() -> dict[str, str]:
    env = dict(os.environ)
    bindir = str(Path.home() / ".okc" / "bin")
    if bindir not in env.get("PATH", "").split(":"):
        env["PATH"] = bindir + ":" + env.get("PATH", "")
    return env


def reserve_count(limit: int, fraction: float) -> int:
    """Attempts that must stay unused. ceil, so a 10-cap at 20% holds 2 back."""
    if limit < 0:
        raise ValueError(f"negative limit {limit}")
    return math.ceil(limit * fraction - 1e-9)


def spendable(limit: int, fraction: float) -> int:
    return max(0, limit - reserve_count(limit, fraction))


def deadline(config: dict | None = None) -> datetime:
    config = config if config is not None else load_config()
    return datetime.fromisoformat(config["deadline"])


def stop_at(config: dict | None = None) -> datetime:
    config = config if config is not None else load_config()
    return deadline(config) - timedelta(hours=float(config["stop_hours_before_deadline"]))


def window_open(now: datetime, config: dict | None = None) -> tuple[bool, str]:
    config = config if config is not None else load_config()
    stop = stop_at(config)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if now >= stop:
        return False, f"budget guard: stopped at {stop.isoformat()} (deadline minus {config['stop_hours_before_deadline']}h)"
    return True, f"open until {stop.isoformat()}"


def _bucket(entry: dict, kind: str, day) -> bool:
    if entry.get("kind") != kind:
        return False
    stamp = datetime.fromisoformat(entry["at"])
    return stamp.astimezone(timezone.utc).date() == day


def used_on(problem: str, kind: str, now: datetime, attempts: dict | None = None) -> int:
    attempts = attempts if attempts is not None else load_attempts()
    history = attempts["problems"][problem][kind]["history"]
    day = now.astimezone(timezone.utc).date()
    return sum(1 for entry in history if _bucket(entry, kind, day))


def budget_decision(
    problem: str,
    kind: str,
    now: datetime,
    config: dict | None = None,
    attempts: dict | None = None,
) -> tuple[bool, str]:
    """kind is 'scored' or 'checks'. Unknown limits and the reserve both block."""
    config = config if config is not None else load_config()
    attempts = attempts if attempts is not None else load_attempts()
    if problem not in PROBLEMS:
        return False, f"{problem} is not a kernel problem"
    if kind not in ("scored", "checks"):
        return False, f"unknown budget kind {kind}"
    open_, why = window_open(now, config)
    if not open_:
        return False, why
    block = attempts["problems"][problem][kind]
    limit = block.get("per_day")
    if limit is None:
        return False, f"{problem} {kind} daily limit is unknown; refresh okc problems before spending"
    reserve = reserve_count(int(limit), float(config["reserve_fraction"]))
    local_used = used_on(problem, kind, now, attempts)
    if local_used >= limit - reserve:
        return False, (
            f"{problem} {kind}: local used {local_used} of {limit}/day, "
            f"reserve {reserve} (reset {attempts.get('resets', '00:00 UTC')})"
        )
    remaining = _fresh_remaining(attempts, block, now)
    if remaining is not None and int(remaining) <= reserve:
        return False, (
            f"{problem} {kind}: server remaining {remaining}, reserve {reserve}"
        )
    return True, (
        f"{problem} {kind}: local used {local_used}, limit {limit}/day, reserve {reserve}"
    )


def _fresh_remaining(attempts: dict, block: dict, now: datetime) -> int | None:
    """Ignore a server remainder fetched on an earlier UTC day."""
    fetched = attempts.get("limits_fetched_at")
    if not fetched or block.get("remaining_server") is None:
        return None
    stamp = datetime.fromisoformat(fetched)
    if stamp.astimezone(timezone.utc).date() != now.astimezone(timezone.utc).date():
        return None
    return int(block["remaining_server"])


def record_attempt(problem: str, kind: str, entry: dict, now: datetime) -> None:
    attempts = load_attempts()
    block = attempts["problems"][problem][kind]
    block["history"].append(entry)
    day = now.astimezone(timezone.utc).date()
    block["used_local"] = sum(1 for item in block["history"] if _bucket(item, kind, day))
    save_json(ATTEMPTS_PATH, attempts)


def parse_variant_header(text: str) -> dict[str, str]:
    """Leading comment keys. Later code comments are not headers."""
    header: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        match = re.match(r"#\s*([A-Za-z_]+):\s*(.*)$", stripped)
        if match and match.group(1) in HEADER_KEYS:
            header[match.group(1)] = match.group(2).strip()
    return header


def parse_hypotheses(markdown: str) -> list[dict[str, str]]:
    """Entries under '## Hypotheses' until the next level-2 heading."""
    match = re.search(r"^## Hypotheses\s*$", markdown, flags=re.M)
    if not match:
        return []
    body = markdown[match.end() :]
    end = re.search(r"^## ", body, flags=re.M)
    if end:
        body = body[: end.start()]
    found = []
    for chunk in re.split(r"^### ", body, flags=re.M)[1:]:
        title, _, rest = chunk.partition("\n")
        title = title.strip()
        if not title.startswith("H"):
            continue
        fields: dict[str, str] = {"title": title, "status": "open"}
        for line in rest.splitlines():
            item = re.match(r"-\s+([a-z_]+):\s*(.*)$", line.strip())
            if item:
                fields[item.group(1)] = item.group(2).strip()
        found.append(fields)
    return found


def best_speedup(problem: str, results: list[dict] | None = None) -> float | None:
    results = results if results is not None else load_results()
    speeds = [
        float(row["speedup"])
        for row in results
        if row.get("problem") == problem
        and row.get("status") == "passed"
        and row.get("kind", "scored") == "scored"
        and row.get("speedup") is not None
    ]
    return max(speeds) if speeds else None


def last_result(problem: str, results: list[dict] | None = None) -> dict | None:
    results = results if results is not None else load_results()
    matches = [row for row in results if row.get("problem") == problem]
    return matches[-1] if matches else None


def parse_problems_payload(payload: dict) -> dict[str, dict]:
    """Normalize `okc problems --json`. Remaining fields are optional."""
    parsed: dict[str, dict] = {}
    for item in payload.get("problems", []):
        name = item["name"]
        limits = item.get("limits") or {}
        parsed[name] = {
            "scored": _limit_block(limits.get("scored") or {}),
            "checks": _limit_block(limits.get("checks") or {}),
        }
    return parsed


def _limit_block(block: dict) -> dict:
    remaining = block.get("remaining", block.get("left"))
    return {
        "per_day": block.get("per_day"),
        "remaining_server": remaining,
        "used_server": block.get("used"),
    }
