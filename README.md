# okc-muse-kernels

My entry for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/), limited to `muse_bf16_proj`: dense BF16/FP32 projections on one H200, timed against SGLang. Generated challenge files are authoritative. See [the problem note](notes/muse_bf16_proj.md) and the [challenge rules](https://openkernelchallenge.dev/rules).

## Results and current goal

| Candidate | Result |
| --- | --- |
| v001–v006 | Correctness passed. v006 was best: 1.0225× check, 1.0139× scored. |
| v007 | 9/9 correctness; 0.7964×. |
| v008–v009 | Gluon compilation failures during correctness. |
| v010 | 9/9 correctness; 0.8034×. |

The v006 scored run remains the only scored result: 1.0139× (95% CI 0.9987–1.0291). The current user-reported live board readback is rank **1147/1151**, with a tenth-place threshold of **1.5687×** (3079.5657 µs) and a leader at **1.6367×**. Top ten is an evidence goal, not a guarantee; this requires roughly 35% lower latency than the scored result.

The next local candidate is v011, targeting the router across token shapes while preserving the proven v006 dense paths. No GPU check or scored submission is authorized for this follow-up yet. Both switches in `state/config.json` remain false.

## Local commands

Python 3.10+ and the official `okc` CLI are used for challenge work. This machine has no GPU; local lint and workflow tests do not establish GPU performance.

```bash
python3 tools/test_tools.py
python3 tools/lint_kernel.py variants/muse_bf16_proj/<candidate>.py
python3 tools/run_loop.py status
```

Refresh limits before any challenge run, preserve the 20% reserve, and stop on a login or rate-limit error. `tools/submit.py` logs exact hashes and requires the matching switch plus `--approved`. The current root [AGENTS.md](AGENTS.md) also requires separate approval before every GPU check or scored submission.

## Security and license

Authentication stays in `~/.okc`; never read or copy tokens into the repository. Do not detect the benchmark, hardcode outputs, cache results across calls, move timed work into `prepare()`, alter the harness or timing, or use network access from the timed kernel. Leave generated `baseline.py`, `reference.py`, and `checks.py` unchanged. Correctness comes before speed.

Submissions use Apache-2.0; see [LICENSE](LICENSE). The kernel track closes at 00:00 on 5 November 2026, America/Los_Angeles. Stop three hours before close. The project target is a working `muse_bf16_proj` by 1 November.

## Repository map

```text
notes/<problem>.md          specification notes, hypotheses, and results
state/attempts.json         daily check and submit counters
state/results.jsonl         aggregate record per challenge run
state/raw/                  local evaluator logs; ignored by Git
state/best/<problem>/       fastest passing scored kernel
state/config.json           safety switches and project limits
variants/<problem>/         versioned candidate kernels
tools/                      lint, submission, parsing, and loop utilities
```
