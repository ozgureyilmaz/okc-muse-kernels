# okc-muse-kernels

Kernel tuning for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). This project works only on `muse_bf16_proj`: BF16 dense projections and an FP32 router on one H200, measured against SGLang.

Generated challenge files define the task and rules. After local initialization, read `muse_bf16_proj/TASK.md` and `muse_bf16_proj/AGENTS.md`. Public references: [official problem](https://openkernelchallenge.dev/problems/muse_bf16_proj), [challenge rules](https://openkernelchallenge.dev/rules), and the [problem notes](notes/muse_bf16_proj.md).

## Results

| Candidate | Run | Correctness | Speedup (95% CI) | Normalized time |
| --- | --- | --- | ---: | ---: |
| v006 | Scored | 9/9 | 1.0139× (0.9987–1.0291) | 4764.5 µs |
| v027 | Check | 9/9 | **1.1804× (1.1626–1.1981)** | **4092.7 µs** |

v027 is the best off-board check. It uses FP32 register accumulation for small routers, BF16 hi/lo Tensor Core split-K for medium routers, hi/lo concatenation for larger routers, selected FlashInfer TinyGEMM paths, and native DeepGEMM for medium-size pooler calls. Other projections retain the measured starter/Hopper paths. Its exact source hash and all 30 timings are in the [round results](notes/muse_bf16_proj.md#top-ten-check-round-2026-10-27) and [`state/results.jsonl`](state/results.jsonl).

**Top ten has not been reached.** The live read at 2026-10-27T16:53:30.411355+00:00 shows the account at rank 1153/1158, with the v006 scored value of 1.0139×. Rank 10 is 1.5696×. v027 needs about 24.8% more normalized latency reduction to reach that point. A check does not update the leaderboard, and no new scored submission was sent.

## Scope and status

- Only `muse_bf16_proj`; the other problems and Phase 2 remain untouched.
- All 16 checks authorized for 27 October completed with 9/9 correctness. Several candidates were slower; see the recorded results.
- Refreshed counters: 16/20 checks and 0/10 scored submissions today. Four checks remain reserved.
- Check, submit, and Phase 2 switches are false; there are no pending GPU jobs.
- The approved scored run required the exact checked hash, 9/9 correctness, and a 95% CI lower bound above live rank 10. v027 does not meet that gate.
- That daily authorization is exhausted; a quota reset does not authorize further runs.

## Local workflow

```bash
python3 tools/test_tools.py
python3 tools/lint_kernel.py variants/muse_bf16_proj/<candidate>.py
python3 tools/run_loop.py status
```

This checkout has no GPU. Local lint and workflow tests do not prove GPU correctness or speed. Challenge runs go through `tools/submit.py` after explicit approval and a fresh limit read, preserving the 20% reserve. The helper logs the exact source hash and requires `--approved` plus the matching config switch. Stop on login or rate-limit errors. Daily server limits reset at 00:00 UTC.

The [technical plan](notes/top10-next-round.md) identifies the remaining dense-layer bottlenecks and the next discriminating experiments. It is a plan, not a measured performance claim.

## Challenge rules

Prioritize correctness. Do not detect the benchmark, hardcode outputs, cache results across calls, move timed GEMM work into `prepare()`, alter the harness or timing, or use network access from the timed kernel. Leave generated `baseline.py`, `reference.py`, and `checks.py` unchanged. Authentication stays in `~/.okc`; do not read or copy its token.

The kernel track closes at 00:00 on 5 November 2026 in America/Los_Angeles. Stop at least three hours before close. The project target is a working `muse_bf16_proj` by 1 November.

Submissions use Apache-2.0; see [`LICENSE`](LICENSE).
