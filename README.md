# okc-muse-kernels

This repository is my entry for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/), problem `muse_bf16_proj`: dense BF16/FP32 projections on one H200, timed against SGLang. See the [challenge rules](https://openkernelchallenge.dev/rules).

## Results

| Variant | Idea | Check speedup |
| --- | --- | ---: |
| v001 | Copy of the starter baseline. | 0.9104× |
| v002 | Split K across Triton CTAs for small batches. | 0.8145× |
| v003 | One tiled Triton GEMM with the router hi/lo path. | 0.5677× |
| v004 | Vector-reduce low-token GEMVs; use `F.linear` for larger batches. | 0.7145× |
| v005 | Use SGLang's Hopper BF16 GEMV for eligible single-token projections. | 1.0174× |
| v006 | Keep Hopper GEMV and fold the router hi/lo sum into FP32 `addmm`. | 1.0225× check; 1.0139× scored |

The best approach uses SGLang's Hopper BF16 GEMV for `m=1` and folds the router hi/lo path into an FP32 `addmm`. The scored run is recorded in `state/results.jsonl`.

## Requirements and local commands

Python 3.10+ and the `okc` CLI from the [challenge site](https://openkernelchallenge.dev/) are required. Challenge runs use the challenge's H200 image; local lint and workflow tests do not need a GPU.

```bash
python3 tools/test_tools.py
python3 tools/lint_kernel.py variants/muse_bf16_proj/<file>.py
python3 tools/run_loop.py status
```

`tools/submit.py` is a dry-run unless the matching `check_enabled` or `submit_enabled` switch in `state/config.json` and the `--approved` flag are both set.

Agent operating instructions are in [AGENTS.md](AGENTS.md).

## Security

Authentication lives in `~/.okc` via `okc login`. Never commit tokens; `.env` files are ignored.

## License

Apache-2.0

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
