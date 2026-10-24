# okc-muse-kernels

This repository tracks kernel experiments for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). The organizers run checks and submissions on an H200. This workspace has no GPU, so local lint is not performance evidence.

## Current project

- Active problem: `muse_bf16_proj` only. The other four problems and Phase 2 are out of scope.
- Latest candidate: `v002_triton_splitk_small_m.py` passed all 9 correctness cases, but measured **0.8145×** (5,931.3 µs vs. SGLang's 4,830.8 µs; 95% interval 0.8022–0.8267). It is slower; do not submit it for a score. Per-shape timings are in the problem note.
- No scored submission has been sent. Both `check_enabled` and `submit_enabled` are false in `state/config.json`.

See [the problem note](notes/muse_bf16_proj.md), [the loop state](state/loop.json), and [the attempt counters](state/attempts.json) for the current plan and run history. Raw OKC logs remain local under the ignored `state/raw/` directory.

## Start or resume

From the repository directory:

```bash
export PATH="$HOME/.okc/bin:$PATH"
okc --version
okc whoami
python3 tools/run_loop.py status
```

If `okc` is not installed, install the official CLI:

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
```

Sign in with `okc login` and complete the device approval yourself. Never expose or copy authentication tokens into this repository; do not run `okc api token` or inspect `~/.okc`.

Before changing a kernel, read its `notes/<problem>.md` and the state files above. After `okc init <problem>`, read every generated file. The generated `TASK.md` and `AGENTS.md` take priority when they differ from the note.

## Kernel workflow

1. Run `okc problems --json` to refresh the current daily limits.
2. Initialize only the active problem: `okc init muse_bf16_proj`.
3. Keep the generated starter unchanged. Copy it into a new, never-overwritten file under `variants/muse_bf16_proj/`, then lint that copy:

   ```bash
   python3 tools/lint_kernel.py variants/muse_bf16_proj/<variant>.py
   ```

4. Get explicit approval before an off-board GPU check. A check does not count as a scored submission, but it uses the daily check allowance.
5. Only after a candidate passes a check, get separate approval before any scored submission. Use `tools/submit.py` for both checks and submissions so every run is logged. The helper requires `--approved` and the matching config switch; it defaults to a dry run.

Example commands, to use only after approval and after enabling the matching switch in `state/config.json`:

```bash
python3 tools/submit.py muse_bf16_proj variants/muse_bf16_proj/my_variant.py --kind check --approved
python3 tools/submit.py muse_bf16_proj variants/muse_bf16_proj/my_variant.py --kind submit --approved
```

Do not edit generated `baseline.py`, `reference.py`, or `checks.py`. The helper copies a variant into the generated `submission.py` only when an approved run starts.

## Loop commands

Run from the repository root with `python3 tools/run_loop.py <command>`:

| Command | Purpose |
| --- | --- |
| `status` | Show each problem's best result, attempt usage, and next action. |
| `plan` | Build `state/plan.json` from the notes and recorded results; lint variants. |
| `cycle` | Run the planning step and print the current loop summary. It does not submit. |
| `refresh-limits` | Fetch `okc problems --json` and update `state/attempts.json`. |
| `analyze` | Add a short post-mortem to the active problem note. |
| `router` | Summarize results by generator style. |

`state/results.jsonl` records each check or submit. Raw OKC logs contain detailed correctness and timing data. Do not claim a speedup unless a result is recorded there.

## Limits and dates

The CLI reports per-problem daily limits, which reset at 00:00 UTC and may change. Keep 20% unused: at the current documented limits, that means at most 8 scored submissions and 16 checks per day. Refresh the counters before future GPU runs.

The kernel track closes at 00:00 on 5 November 2026, America/Los_Angeles. Stop work three hours before closing. The target is a working `muse_bf16_proj` by 1 November.

## Repository map

```text
notes/<problem>.md          problem rules, hypotheses, and findings
state/attempts.json         daily check and submit counters
state/results.jsonl         one aggregate record per run
state/raw/                  local OKC run logs; ignored by Git
state/best/<problem>/       fastest passing scored kernel
state/config.json           safety switches and project limits
variants/<problem>/         versioned candidate kernels
tools/                      lint, submit, parse, and loop utilities
```

`okc init` creates gitignored `muse_*/` directories at the repository root. Keep candidate source in `variants/`; the submit helper copies it into the generated `submission.py` when a run is approved.

## Challenge rules

- Correctness comes before speed. A kernel that misses tolerance scores zero.
- No benchmark detection, hardcoded outputs, caching across calls, timing or harness tampering, or judge-time network access.
- Keep activations and weights at their allowed precision. Repack weights losslessly in `prepare`; do not move timed work there.
- If OKC reports a logged-out or rate-limited state, stop.
- Submissions must use the Apache-2.0 license.

Read the official [challenge rules](https://openkernelchallenge.dev/rules) and the active problem's `TASK.md` before implementing or submitting a kernel.
