# okc-muse-kernels

Autonomous kernel loop for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). Evaluation is `okc submit` on the organizers' H200. This machine is not a source of speedups.

The default branch is `master`. There is no `main`.

## Resume

From a clean checkout:

```bash
git clone https://github.com/ozgureyilmaz/okc-muse-kernels
cd okc-muse-kernels
export PATH="$HOME/.okc/bin:$PATH"
okc --version          # 0.7.1 when this tree was written
okc whoami             # "not signed in" until login is approved
python tools/run_loop.py status
```

If `okc` is missing, the only install is:

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
```

Then read `notes/muse_bf16_proj.md` and `state/loop.json`.

Current stage: **waiting for login.** `okc init` has not been run. No variant exists. No check and no submit has been sent. `submit_enabled` and `check_enabled` in `state/config.json` are false, so `python tools/submit.py ...` prints the command and does not run it.

Next, in order:

1. Ask "Ready for okc login?" and wait for a yes. Then `okc login`, post the device code and https://github.com/login/device, and wait. Do not automate login and do not read `~/.okc`.
2. `okc problems --json`, then `okc init muse_bf16_proj` only.
3. Re-read every file init wrote. If it disagrees with `notes/muse_bf16_proj.md`, the files win. Update the note.
4. Copy the untouched starter to `variants/muse_bf16_proj/v001_baseline_copy.py` and lint it.
5. Stop again. The first GPU run is `okc check` of that starter, and only after a separate OK. The first `okc submit` is another OK, after a check passes.

Do not start the other four problems. Do not start Phase 2. Do not claim a speedup that is not in `state/results.jsonl`.

## Loop

`python tools/run_loop.py cycle` plans from `notes/` and `state/results.jsonl`, lints variants, and prints:

```text
problem | best speedup | attempts used/limit | last result | next action
```

It does not submit. `tools/submit.py` is the only submit path. It logs one JSON line per run (problem, variant, generator, hash, speedup, time, tests, status). That generator column is the record of which style worked. `python tools/run_loop.py router` prints it.

Stages, each resumable from `state/`:

| command | what it does |
| --- | --- |
| `status` | print the table, change nothing |
| `plan` or `cycle` | rewrite `state/plan.json` from the open hypotheses, lint variants |
| `refresh-limits` | `okc problems --json` into `state/attempts.json` |
| `analyze` | append a three-line post-mortem to the problem note |
| `lint <file>` | static gate, no GPU |

Hypotheses live under `## Hypotheses` in the problem note, as `### Hn slug` plus `- generator:`, `- expected_gain:`, `- risk:`, `- cost:`, `- status:`, `- note:`. A variant file starts with `# okc: problem=<name>` and the header keys `variant`, `problem`, `generator`, `idea`, `expected_gain`, `risks`. Files in `variants/` are never overwritten.

## Budget

Per problem, from `okc problems` on 23 Oct 2026: 10 scored submits and 20 checks a day, reset 00:00 UTC. The rules page says these can change. Keep 20% unused, so the tools spend at most 8 submits and 16 checks. The kernel track closes 2026-11-05 00:00 America/Los_Angeles (Nov 4, 11:59 pm PT). The loop stops three hours earlier. Target a working `muse_bf16_proj` by Nov 1.

`okc check` is a real run kept off the board. It does not spend a scored attempt. It does spend the check cap.

## Layout

```text
notes/<problem>.md          rules, roofline, hypotheses, dead ends
state/attempts.json         daily counters
state/results.jsonl         one line per run, including generator style
state/best/<problem>/       fastest passing kernel
state/config.json           switches, all defaulting to off
variants/<problem>/         every candidate, kept
tools/                      submit, parse, lint, leaderboard, run_loop
```

`okc init` writes `muse_*/` at the repo root. Those trees are gitignored. Edit only `submission.py` inside them, and only by copying a file from `variants/`.

## Hard rules

- No benchmark detection, no caching across calls, no hardcoded outputs, no harness or timing tampering, no network on the judge. A loophole gets reported and not used.
- Correctness before speed. A faster kernel that misses tolerance scores zero.
- Never print or store `okc` tokens, device codes, or auth files. Never run `okc api token`.
- If okc says logged out or rate limited, stop.
- Local CPU checks are syntax, imports, shapes, and dtypes only.

Kernels we submit are Apache-2.0, which is the challenge rule.
