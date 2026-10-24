# SPDX-License-Identifier: Apache-2.0

"""Local tests for the loop. No okc login and no GPU."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import lint_kernel
import parse_result
import run_loop
import submit

SAMPLE = """\
✓ submitted muse_moe_w4a16/submission.py to muse_moe_w4a16 job 9c1e04b7a2d35f86
✓ muse_moe_w4a16 passed job 9c1e04b7a2d35f86
1.384× ━━━━━━━━━╋━━━━────────────── faster than SGLang
time 9,978.6 µs
tests 12/12 passed
"""

NOTES = """\
# example

## Hypotheses

### H1 triton-splitk-small-m
- generator: triton
- expected_gain: decode
- risk: medium
- cost: high
- status: open
- note: split-K

### H2 parked-idea
- generator: pytorch
- expected_gain: none
- risk: low
- cost: low
- status: dead
- note: do not retry

## Dead ends
"""

GOOD = """\
# okc: problem=muse_bf16_proj
# variant: v001_baseline_copy
# problem: muse_bf16_proj
# generator: pytorch
# idea: starter
# expected_gain: pipeline check
# risks: router matmul may be TF32

import torch


def prepare(weights):
    return weights


def kernel_fn(data, state):
    w = state[data["config"]["layer"]][: data["config"]["n"]]
    x = data["x"]
    if w.dtype == torch.float32:
        return x.float() @ w.T
    return x @ w.T
"""

BAD = """\
# okc: problem=muse_bf16_proj
import requests

CACHE = {}

def kernel_fn(data, state):
    value = data["x"].item()
    CACHE[id(data)] = value
    open("/tmp/out", "w")
    return data["x"]
"""


class ParseTests(unittest.TestCase):
    def test_public_sample(self):
        parsed = parse_result.parse_output(SAMPLE, problem="muse_moe_w4a16")
        self.assertEqual(parsed["status"], "passed")
        self.assertEqual(parsed["job"], "9c1e04b7a2d35f86")
        self.assertAlmostEqual(parsed["speedup"], 1.384)
        self.assertAlmostEqual(parsed["time_us"], 9978.6)
        self.assertEqual((parsed["tests_passed"], parsed["tests_total"]), (12, 12))
        self.assertTrue(parsed["confident"])
        self.assertFalse(parsed["halt"])

    def test_milliseconds(self):
        self.assertAlmostEqual(parse_result.time_to_us("10.53", "ms"), 10530.0)

    def test_logged_out_halts(self):
        parsed = parse_result.parse_output("✗ not signed in\nnext  okc login\n")
        self.assertEqual(parsed["status"], "logged_out")
        self.assertTrue(parsed["halt"])

    def test_json_does_not_trust_a_unitless_time(self):
        parsed = parse_result.parse_output(
            '{"status": "passed", "speedup": 1.1, "time": 12, "tests_passed": 9, "tests_total": 9}'
        )
        self.assertEqual(parsed["status"], "passed")
        self.assertIsNone(parsed["time_us"])
        self.assertTrue(parsed["confident"])


class BudgetTests(unittest.TestCase):
    def test_reserve(self):
        self.assertEqual(common.reserve_count(10, 0.2), 2)
        self.assertEqual(common.spendable(10, 0.2), 8)
        self.assertEqual(common.reserve_count(20, 0.2), 4)
        self.assertEqual(common.spendable(1, 0.2), 0)

    def test_unknown_limit_blocks(self):
        now = datetime.fromisoformat("2026-10-23T12:00:00+00:00")
        config = {"deadline": "2026-11-05T00:00:00-08:00", "stop_hours_before_deadline": 3, "reserve_fraction": 0.2}
        attempts = {
            "problems": {"muse_bf16_proj": {"scored": {"per_day": None, "history": [], "remaining_server": None}}},
            "limits_fetched_at": now.isoformat(),
        }
        ok, why = common.budget_decision("muse_bf16_proj", "scored", now, config, attempts)
        self.assertFalse(ok)
        self.assertIn("unknown", why)

    def test_reserve_blocks_the_last_fifth(self):
        now = datetime.fromisoformat("2026-10-23T12:00:00+00:00")
        config = {"deadline": "2026-11-05T00:00:00-08:00", "stop_hours_before_deadline": 3, "reserve_fraction": 0.2}
        history = [
            {"at": now.isoformat(), "kind": "scored"}
            for _ in range(8)
        ]
        attempts = {
            "problems": {"muse_bf16_proj": {"scored": {"per_day": 10, "history": history, "remaining_server": None}}},
            "limits_fetched_at": now.isoformat(),
        }
        ok, _ = common.budget_decision("muse_bf16_proj", "scored", now, config, attempts)
        self.assertFalse(ok)

    def test_stop_three_hours_before_close(self):
        config = {"deadline": "2026-11-05T00:00:00-08:00", "stop_hours_before_deadline": 3}
        early = datetime.fromisoformat("2026-11-04T20:00:00-08:00")
        late = datetime.fromisoformat("2026-11-04T21:30:00-08:00")
        self.assertTrue(common.window_open(early, config)[0])
        self.assertFalse(common.window_open(late, config)[0])


class NotesAndLintTests(unittest.TestCase):
    def test_hypotheses_skip_dead(self):
        found = common.parse_hypotheses(NOTES)
        self.assertEqual([item["title"].split()[0] for item in found], ["H1", "H2"])
        self.assertEqual(found[1]["status"], "dead")

    def test_real_notes_have_seven_open(self):
        text = (common.NOTES / "muse_bf16_proj.md").read_text()
        found = [item for item in common.parse_hypotheses(text) if item.get("status") == "open"]
        self.assertEqual(len(found), 7)
        self.assertEqual(found[0]["generator"], "cutlass_flashinfer")


class SubmitPromotionTests(unittest.TestCase):
    def test_exact_hash_promotion_requires_all_check_gates(self):
        digest = "a" * 64
        row = {
            "problem": "muse_bf16_proj",
            "hash": digest,
            "kind": "check",
            "status": "passed",
            "job": "check-job",
        }
        result = {
            "passed": True,
            "checks": {"ok": True},
            "speedup": 1.022,
            "speedup_ci": [1.005, 1.04],
            "records": [{"event": "test", "status": "pass"} for _ in range(9)],
        }
        payload = {
            "board": "muse_bf16_proj",
            "mode": "check",
            "id": "check-job",
            "result": result,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            (raw / f"check-{digest[:12]}.log").write_text(json.dumps(payload))
            with patch.object(submit.common, "RAW", raw):
                self.assertTrue(
                    submit._has_eligible_check(
                        "muse_bf16_proj", digest, [row], {"min_improvement_to_continue": 0.02}
                    )
                )
                result["speedup_ci"] = [0.99, 1.04]
                (raw / f"check-{digest[:12]}.log").write_text(json.dumps(payload))
                self.assertFalse(
                    submit._has_eligible_check(
                        "muse_bf16_proj", digest, [row], {"min_improvement_to_continue": 0.02}
                    )
                )

    def test_lint(self):
        good = lint_kernel.lint_source(GOOD)
        bad = lint_kernel.lint_source(BAD)
        self.assertTrue(good.ok, good.errors)
        self.assertFalse(bad.ok)
        blob = " ".join(bad.errors)
        self.assertIn("requests", blob)
        self.assertIn("item", blob)
        self.assertIn("open", blob)

    def test_header(self):
        header = common.parse_variant_header(GOOD)
        self.assertEqual(header["generator"], "pytorch")
        self.assertEqual(header["variant"], "v001_baseline_copy")

    def test_consecutive_small_gains_count_as_misses(self):
        rows = [
            {"problem": "muse_bf16_proj", "status": "passed", "kind": "scored", "speedup": speed}
            for speed in (1.0, 1.10, 1.111, 1.12, 1.13)
        ]
        self.assertEqual(run_loop._consecutive_misses(rows, 0.02), 3)


class StatusTests(unittest.TestCase):
    def test_status_prints_the_active_problem(self):
        table = run_loop.format_status()
        self.assertIn("muse_bf16_proj", table)
        self.assertIn("scored 0/10", table)
        self.assertIn("checks 0/20", table)
        self.assertIn("not started", table)


if __name__ == "__main__":
    unittest.main()
