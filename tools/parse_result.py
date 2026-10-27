# SPDX-License-Identifier: Apache-2.0

"""Turn an `okc submit` / `okc check` transcript into one result record.

The text patterns follow the sample on https://openkernelchallenge.dev/.
JSON is whatever `okc submit --json` prints. The first real transcript still
has to confirm the field names; until then `parser_locked` stays false and a
JSON object is stored with the keys we could read, plus `json_keys`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_JOB = re.compile(r"\bjob\s+([0-9a-fA-F]{8,})\b")
_SPEEDUP = re.compile(r"(\d+(?:\.\d+)?)\s*[×x](?!\w)")
_TIME = re.compile(
    r"\btime\s+([0-9][0-9,]*(?:\.\d+)?)\s*(µs|us|μs|ms|s)\b",
    re.IGNORECASE,
)
_TESTS = re.compile(r"\btests\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
_ERROR = re.compile(r"relative L2 error\s+([0-9.eE+-]+)", re.IGNORECASE)

_LOGGED_OUT = (
    "not signed in",
    "not logged in",
    "logged out",
    "unauthorized",
    "accept the rules",
)
_RATE_LIMITED = (
    "rate limit",
    "daily limit",
    "no attempts left",
    "none left",
    "limit reached",
    "too many requests",
)


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def time_to_us(amount: str, unit: str) -> float:
    value = float(amount.replace(",", ""))
    unit = unit.lower().replace("μ", "µ")
    if unit == "µs" or unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    raise ValueError(f"unknown time unit {unit}")


def parse_output(text: str, problem: str | None = None) -> dict:
    raw = common.redact(strip_ansi(text or ""))
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = parse_json_output(json.loads(stripped), problem)
            parsed["raw_excerpt"] = stripped[:500]
            return parsed
        except json.JSONDecodeError:
            pass
    parsed = parse_text_output(raw, problem)
    parsed["raw_excerpt"] = raw[:500]
    return parsed


def parse_text_output(text: str, problem: str | None = None) -> dict:
    lowered = text.lower()
    job = _first(_JOB, text)
    speedup = _first(_SPEEDUP, text)
    time_match = _TIME.search(text)
    tests = _TESTS.search(text)
    error = _first(_ERROR, text)
    status, halt = _status_from_text(lowered, tests)
    confident = status in {"passed", "failed", "logged_out", "rate_limited"} and (
        status in {"logged_out", "rate_limited"} or tests is not None or speedup is not None
    )
    return {
        "source": "text",
        "status": status,
        "problem": problem,
        "job": job,
        "speedup": float(speedup) if speedup is not None else None,
        "time_us": time_to_us(time_match.group(1), time_match.group(2)) if time_match else None,
        "tests_passed": int(tests.group(1)) if tests else None,
        "tests_total": int(tests.group(2)) if tests else None,
        "error": error,
        "halt": halt,
        "confident": confident,
        "message": _message(status, error),
    }


def parse_json_output(payload: object, problem: str | None = None) -> dict:
    structured = _parse_evaluator_job(payload, problem)
    if structured is not None:
        return structured
    flat = _flatten(payload)
    status, halt = _status_from_json(flat)
    tests_passed, tests_total = _tests_from_json(flat)
    speedup = _number(flat, ("speedup", "score"))
    time_us = _time_from_json(flat)
    job = _string(flat, ("job", "job_id", "id"))
    error = _string(flat, ("error", "message", "reason"))
    confident = status != "unknown" and (speedup is not None or tests_passed is not None or halt)
    return {
        "source": "json",
        "status": status,
        "problem": _string(flat, ("problem", "name")) or problem,
        "job": job,
        "speedup": speedup,
        "time_us": time_us,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "error": error,
        "halt": halt,
        "confident": confident,
        "message": _message(status, error),
        "json_keys": sorted(flat),
    }


def _parse_evaluator_job(payload: object, problem: str | None) -> dict | None:
    """Parse the completed job JSON returned by the current OKC CLI."""
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    job = payload.get("id")
    if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
        return None
    if not isinstance(job, str) or not job:
        return None

    status = "passed" if result["passed"] else "failed"
    records = result.get("records")
    tests = [
        record for record in records
        if isinstance(record, dict) and record.get("event") == "test"
    ] if isinstance(records, list) else []
    tests_passed = sum(record.get("status") == "pass" for record in tests) if tests else None
    tests_total = len(tests) if tests else None

    error = result.get("error")
    if not isinstance(error, str) or not error.strip():
        failed = [record.get("message") for record in tests if record.get("status") != "pass"]
        error = next((message for message in failed if isinstance(message, str) and message.strip()), None)
    if not error:
        log_tail = result.get("log_tail")
        error = log_tail[:1200] if isinstance(log_tail, str) and log_tail else None
    elif len(error) > 1200:
        error = error[:1200]

    score_us = result.get("score_us")
    return {
        "source": "json",
        "status": status,
        "problem": payload.get("board") or problem,
        "job": job,
        "speedup": _number({"speedup": result.get("speedup")}, ("speedup",)),
        "time_us": float(score_us) if isinstance(score_us, (int, float)) else None,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "error": error,
        "halt": False,
        "confident": True,
        "message": _message(status, error),
        "json_keys": sorted(_flatten(payload)),
    }


def _status_from_text(lowered: str, tests: re.Match | None) -> tuple[str, bool]:
    if any(phrase in lowered for phrase in _LOGGED_OUT):
        return "logged_out", True
    if any(phrase in lowered for phrase in _RATE_LIMITED):
        return "rate_limited", True
    if "timeout" in lowered:
        return "timeout", False
    tests_ok = tests is not None and tests.group(1) == tests.group(2) and int(tests.group(2)) > 0
    if re.search(r"\bfailed\b", lowered) or (tests is not None and not tests_ok):
        return "failed", False
    if re.search(r"\bpassed\b", lowered) and (tests is None or tests_ok):
        return "passed", False
    return "unknown", False


def _status_from_json(flat: dict) -> tuple[str, bool]:
    interesting = [
        str(flat[key]).lower()
        for key in ("status", "passed", "error", "message", "reason", "detail")
        if key in flat and flat[key] is not None
    ]
    blob = " ".join(interesting)
    if any(phrase in blob for phrase in _LOGGED_OUT):
        return "logged_out", True
    if any(phrase in blob for phrase in _RATE_LIMITED):
        return "rate_limited", True
    status = str(flat.get("status", "")).lower()
    passed = flat.get("passed")
    if status in {"passed", "pass", "ok"} or passed is True:
        return "passed", False
    if status in {"failed", "fail", "error"} or passed is False:
        return "failed", False
    if "timeout" in status:
        return "timeout", False
    return "unknown", False


def _tests_from_json(flat: dict) -> tuple[int | None, int | None]:
    if "tests_passed" in flat and "tests_total" in flat:
        return int(flat["tests_passed"]), int(flat["tests_total"])
    tests = flat.get("tests")
    if isinstance(tests, str):
        match = _TESTS.search("tests " + tests)
        if match:
            return int(match.group(1)), int(match.group(2))
    if isinstance(tests, dict):
        passed = tests.get("passed", tests.get("ok"))
        total = tests.get("total", tests.get("count"))
        if passed is not None and total is not None:
            return int(passed), int(total)
    return None, None


def _time_from_json(flat: dict) -> float | None:
    if "time_us" in flat and isinstance(flat["time_us"], (int, float)):
        return float(flat["time_us"])
    if "time_ms" in flat and isinstance(flat["time_ms"], (int, float)):
        return float(flat["time_ms"]) * 1000.0
    raw = flat.get("time")
    if isinstance(raw, str):
        match = _TIME.search("time " + raw)
        if match:
            return time_to_us(match.group(1), match.group(2))
    # A bare number has no unit until a real transcript locks the schema.
    return None


def _flatten(payload: object, prefix: str = "") -> dict:
    flat: dict = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                flat.update(_flatten(value, name))
            else:
                flat[str(key)] = value
                flat[name] = value
    elif isinstance(payload, list) and payload:
        flat.update(_flatten(payload[0], prefix))
    return flat


def _number(flat: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in flat and flat[key] is not None:
            try:
                return float(flat[key])
            except (TypeError, ValueError):
                continue
    return None


def _string(flat: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = flat.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _message(status: str, error: str | None) -> str:
    if status == "logged_out":
        return "okc is logged out. Stop. Do not retry until a person runs okc login."
    if status == "rate_limited":
        return "okc reported a rate or daily limit. Stop. Do not retry until the budget resets."
    if error:
        return error
    return status


def main() -> None:
    text = sys.stdin.read() if len(sys.argv) == 1 else Path(sys.argv[1]).read_text()
    json.dump(parse_output(text), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
