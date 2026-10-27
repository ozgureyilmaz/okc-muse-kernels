# Agent operating instructions

- Work only on `muse_bf16_proj`; do not initialize another problem or enable Phase 2.
- Treat generated challenge files as authoritative and leave them unchanged. Put candidate kernels under `variants/muse_bf16_proj/`.
- Complete `okc login` yourself. Agents must not read `~/.okc` or copy authentication data into the repository.
- Never resend a source hash already recorded in `state/results.jsonl`; the check and scored result for the current best candidate are already recorded there.
- Before checks or submissions, refresh the daily limits, preserve the 20% reserve, and stop if login or rate-limit errors occur. Do not run a check or scored submission without explicit approval; `tools/submit.py` also requires the matching switch in `state/config.json` and `--approved`.
- The challenge closes at 00:00 on 5 November 2026 in America/Los_Angeles. Stop work at least three hours before close.
- Follow the challenge rules: prioritize correctness, do not detect the benchmark or hardcode outputs, do not alter the harness or timing, and do not use network access from the timed kernel.
