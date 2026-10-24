# okc-muse-kernels

Kernel experiments for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). Official checks run on an H200. This machine has no GPU, so local lint and tests verify code and workflow only; performance claims come from recorded OKC results.

## Current state

- Active problem: `muse_bf16_proj` only. Other problems and Phase 2 are out of scope.
- Candidate `v006_hopper_gemv_router_addmm.py` passed all nine correctness cases. Its off-board check measured **1.0225×** (95% CI **1.0072–1.0378×**).
- Scored job `00e22ad9869b4bb7` remains the only scored result: **1.0139×** (95% CI **0.9987–1.0291×**), with correctness passed. The live leaderboard readback at **2026-10-24 16:40 UTC (19:40 Europe/Istanbul)** shows rank **751/755**.
- Four follow-up checks (v007–v010) did not meet the stricter 1.04× / CI lower-bound 1.02× gate. No additional score was submitted.
- `state/config.json` keeps `check_enabled` and `submit_enabled` false. Do not resend the v006 hash; the recorded results are in `state/results.jsonl`.

See [the problem note](notes/muse_bf16_proj.md) for the specification, candidate history, and per-shape timings. `state/results.jsonl` records aggregate check and submission results. Detailed evaluator logs stay local under the ignored `state/raw/` directory.

## Quick start

Run from the repository directory:

```bash
export PATH="$HOME/.okc/bin:$PATH"
okc --version
okc whoami
python3 tools/run_loop.py status
```

If the CLI reports that you are signed out, stop and complete `okc login` yourself. Do not read or copy authentication tokens into this repository.

## Kernel workflow

1. Read the active problem's `TASK.md` and `AGENTS.md`, its note under `notes/`, and the state files before editing.
2. Refresh daily limits with `python3 tools/run_loop.py refresh-limits`. Preserve the 20% reserve and stop if OKC reports a login or rate-limit problem.
3. Keep generated starter, reference, and checks files unchanged. Copy the starter to a new file under `variants/<problem>/`, then lint it:

   ```bash
   python3 tools/lint_kernel.py variants/<problem>/<candidate>.py
   ```

4. Run an off-board check only with explicit authorization. Use `tools/submit.py` so the exact source hash and job are logged:

   ```bash
   python3 tools/submit.py <problem> variants/<problem>/<candidate>.py --kind check --approved
   ```

5. Submit a score only after the exact checked hash passes correctness and the required speed and confidence gates. Refresh limits and verify the hash first. Use the same helper with `--kind submit --approved`.

The helper stays in dry-run mode unless the matching switch in `state/config.json` is enabled and `--approved` is supplied. Restore both switches to `false` after each run. Never edit whitespace or comments to get another timing for an already checked hash.

## Local tools

Run the offline workflow tests with:

```bash
python3 tools/test_tools.py
```

The loop utility can also refresh limits and report the project state:

```bash
python3 tools/run_loop.py refresh-limits
python3 tools/run_loop.py status
```

It does not run a GPU check or submission by itself.

## Challenge rules

- Correctness comes before speed; a tolerance miss scores zero.
- Do not detect the benchmark, hardcode outputs, cache results across calls, move timed work into `prepare()`, alter the harness or timing, or use network access on the judge.
- Keep activation and weight precision within the active problem's rules.
- Submissions must use the Apache-2.0 license.

Read the [official challenge rules](https://openkernelchallenge.dev/rules) and the active problem files before implementing a kernel.

## Dates

The kernel track closes at 00:00 on 5 November 2026, America/Los_Angeles. Stop three hours before close. The project target is a working `muse_bf16_proj` by 1 November.

## Repository map

```text
notes/<problem>.md          specification notes, hypotheses, and findings
state/attempts.json         daily check and submit counters
state/results.jsonl         one aggregate record per run
state/raw/                  local evaluator logs; ignored by Git
state/best/<problem>/       fastest passing scored kernel
state/config.json           safety switches and project limits
variants/<problem>/         versioned candidate kernels
tools/                      lint, submit, parsing, and loop utilities
```
