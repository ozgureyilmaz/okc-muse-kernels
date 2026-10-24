# muse_bf16_proj

Dense projections for Muse Spark Open, one H200, timed against SGLang. The unchanged starter passed its first off-board check (9/9 correctness cases), with a recorded ratio of 0.9104×. Candidate `v002_triton_splitk_small_m.py` passed all nine correctness cases, but its off-board ratio was 0.8145× (5931.3 µs vs. SGLang's 4830.8 µs; 95% interval 0.8022–0.8267). It is slower and has not been submitted for a score.

The initial problem details below came from `okc show muse_bf16_proj --json` on 23 Oct 2026 with okc 0.7.1, while logged out. On 2026-10-23, after the user completed login, `okc problems --json` succeeded and `okc init muse_bf16_proj` ran. All six generated files (`TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py`, `checks.py`, and `submission.py`) were read. They are authoritative if they differ from this note. The starter remains untouched; `variants/muse_bf16_proj/v001_baseline_copy.py` is an exact copy and passed lint. Its first check passed as job `0879b1ef893c49d6`; v002's check passed as job `52ed320192494632`. Raw evaluator records remain local under ignored `state/raw/`; aggregate rows are in `state/results.jsonl` and per-shape v002 timings are in this note.

Challenge: https://openkernelchallenge.dev/problems/muse_bf16_proj
Rules: https://openkernelchallenge.dev/rules

## Operation

`prepare(weights)` runs once, untimed, up to 300 s. Repack is allowed. `kernel_fn(data, state)` is the timed call.

```text
out = x @ W[:n].T
```

`x` is BF16 `[m, k]`. `W` is `weights[config["layer"]]`. Return a GPU tensor of shape `[m, n]` and the layer's dtype. The submission's first line must stay `# okc: problem=muse_bf16_proj` (init adds it; the show template did not include it).

| layer | W shape | dtype | n |
| --- | --- | --- | --- |
| qkv_gate | 2304 × 8192 | BF16 | 2304 |
| o | 8192 × 1024 | BF16 | 8192 |
| pooler | 4096 × 8192 | BF16 | 4096 |
| unpooler | 8192 × 4096 | BF16 | 1024 when m ≤ 8, else 8192 |
| router | 256 × 8192 | FP32 in, FP32 out | 256 |

`x` stays BF16. qkv/o/pooler/unpooler return BF16. The router returns FP32.

## Scored shapes

Six loads, five layers each, 30 shapes. Every shape's weight is 62: a load's time is the sum of its five kernel times, times 62. The problem speedup is the geometric mean, over the six loads, of SGLang's load time over ours. The published SGLang figure is 4830.8 µs per decode step (4.83 ms on the site).

| load | m |
| --- | --- |
| decode 1, 16, 32 | 1, 16, 32 |
| verify 1, 16, 32 | 8, 128, 256 |

Verify is an 8-token speculative draft check. Decode is spec off.

Correctness also runs nine cases that are not the scored set, including outliers (1% of channels ×30) and m outside the scored list: m=1 qkv_gate, m=7 pooler, m=33 o, m=130 unpooler, m=5 unpooler, m=8 router, m=9 router outliers, m=512 qkv_gate outliers, m=1000 unpooler outliers. A kernel that only handles the thirty scored values of m fails. Final review also reruns nearby shapes. Dispatch on ranges of m, n, k, and layer, not on the exact scored list.

## Tolerance

Against an FP64 reference (`reference.py`):

- Relative L2 at most 1e-2, and 1e-4 for the router.
- Each row at most 5× that bound. A near-zero row is measured against 0.1× the average row norm.
- No NaN or Inf.
- About one row in eight of `x` is a single power-of-two spike. That output row is a scaled weight column. It must be within 1e-3. Exact for the original BF16 weights in any accumulation order. The router's BF16 hi+lo split is about 3e-6 off and passes. BF16 router weights alone fail. INT8 or FP8 weights miss the probe by about 5–10×.

The router may run as a BF16 hi + lo pair with FP32 accumulation at any m. SGLang does that above 8 tokens. That is the one allowed precision exception.

## Baseline

SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, file `baseline.py`:

- qkv_gate, o, pooler, unpooler: `_bf16_gemm_dispatch_impl` (cuBLAS on H200). Unpooler uses `W[:n]`, so the m≤8 slice is already there. The sliced unpooler's all-gather is not part of this problem.
- Router: FP32 `F.linear` for m≤8, then two BF16 GEMMs on a hi/lo split of the weight with FP32 accumulation.

The PyTorch starter is `x @ W[:n].T`, FP32 matmul for the router at every m. On Hopper that FP32 matmul can be TF32. TF32 may fail the router tolerance. Do not assume the starter passes until a check says so.

## Roofline

H200 figures used by the judge: 4.8 TB/s, 989 TFLOP/s BF16. Bytes are the weight, `x`, and the output. FLOPs are `2·m·n·k`. The floor in `TASK.md` matches that model within 0.1 µs.

Decode is bandwidth-bound on all five layers. At verify-32, pooler, unpooler, and qkv_gate touch the 989 TFLOP/s roof (floors 17.4, 17.4, and 9.8 µs). `o` and the router stay bandwidth-bound.

Per-load floors, already ×62:

| load | floor |
| --- | --- |
| decode 1 | 1792 µs |
| decode 16 | 2573 µs |
| decode 32 | 2585 µs |
| verify 1 | 1798 µs |
| verify 16 | 2716 µs |
| verify 32 | 3212 µs |

Geometric mean of those floors is about 2391 µs. If 4830.8 µs is the same average of SGLang's load times, the ceiling is about 2.02×. That is an estimate, not a measurement. Pooler dominates decode-1 bytes (14.0 of 28.9 µs). Pooler plus the full unpooler dominate once m>8.

The judge subtracts about 4 µs of CUDA-graph replay overhead from both sides, takes the median of paired ratios, and reports a 95% interval of at least ±1.5%. A gain inside that band is noise. Each replay starts with a cold L2.

## Attempt limits

From `okc problems --json` and the rules page, per problem, reset 00:00 UTC. The rules say the caps can change with load. Refreshed after v002's check on 2026-10-23: `muse_bf16_proj` used 0/10 submits and 2/20 checks, with 18 checks remaining. Keep the 20% reserve (at most 16 checks and 8 submits per day).

| kind | per day | 20% reserve | may spend |
| --- | --- | --- | --- |
| scored `okc submit` | 10 | 2 | 8 |
| `okc check` (off the board, not a scored attempt) | 20 | 4 | 16 |

No contest-long cap showed up, only the daily cap. A run has 10 minutes including compilation. A timeout is charged. The same source returns its earlier result.

`state/config.json` has `submit_enabled` and `check_enabled` set false. `tools/submit.py` stays a dry-run until both the switch and `--approved` are set, and it refuses a hash that was already sent.

## Rules that change the kernel

- CUDA-graph replay: no `.item()`, `.cpu()`, `synchronize()`, printing a tensor, or data-dependent allocation or branches inside `kernel_fn`. Branching on shapes and `config` is allowed.
- Do not cache anything derived from `data`.
- BF16 weights stay BF16, or smaller only if lossless. Activations stay BF16.
- The image is `lmsysorg/sglang:dev-muse-spark-open`. Triton, CUTLASS, CuTe, FlashInfer, DeepGEMM, and TileLang are available. Nothing else installs. No network on the judge.
- `TASK.md` adds a pre-run guard: submissions must not patch timing, read harness frames or harness I/O/secrets, edit loaded modules, import the reference, start processes, or use the network.
- Submissions are Apache-2.0.

## Hypotheses

Ranked. H1 has a first implementation and check result; other hypotheses remain unimplemented.

### H1 triton-splitk-small-m
- generator: triton
- expected_gain: the decode loads, if cuBLAS misses bandwidth at m=1..32. Pooler and qkv_gate are the bytes that matter.
- risk: medium
- cost: high
- status: checked-no-gain
- note: `variants/muse_bf16_proj/v002_triton_splitk_small_m.py` splits K for non-router m<=32, writes FP32 partials, then reduces and stores BF16 in a second kernel. All nine correctness cases passed, but the aggregate off-board ratio was 0.8145×. The implementation adds a second launch and FP32 scratch; this run does not isolate their individual cost. Router and m>32 use the starter path. Do not submit this candidate for a score.

## v002 per-shape check (off-board)

Job `52ed320192494632` passed all nine correctness cases. The table records OKC's paired adjusted medians in microseconds. Ratio is SGLang time divided by v002 time; values below 1 mean v002 was slower. The aggregate ratio was 0.8145× (95% interval 0.8022–0.8267). The raw log is local and ignored by Git; this table preserves its per-shape measurements. Router and m>32 use the starter path, so their rows do not exercise the new split-K kernels.

| Load | m | Layer | SGLang µs | v002 µs | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| decode 1 | 1 | qkv_gate | 14.416 | 18.355 | 0.7854 |
| decode 1 | 1 | o | 6.448 | 7.938 | 0.8123 |
| decode 1 | 1 | pooler | 21.440 | 22.169 | 0.9671 |
| decode 1 | 1 | unpooler | 7.264 | 9.155 | 0.7935 |
| decode 1 | 1 | router | 10.432 | 10.231 | 1.0196 |
| decode 16 | 16 | qkv_gate | 13.904 | 18.075 | 0.7692 |
| decode 16 | 16 | o | 7.024 | 8.845 | 0.7941 |
| decode 16 | 16 | pooler | 21.456 | 22.642 | 0.9476 |
| decode 16 | 16 | unpooler | 18.640 | 21.115 | 0.8828 |
| decode 16 | 16 | router | 14.960 | 26.481 | 0.5649 |
| decode 32 | 32 | qkv_gate | 14.528 | 20.195 | 0.7194 |
| decode 32 | 32 | o | 7.072 | 11.385 | 0.6212 |
| decode 32 | 32 | pooler | 21.584 | 28.665 | 0.7530 |
| decode 32 | 32 | unpooler | 18.704 | 29.852 | 0.6265 |
| decode 32 | 32 | router | 15.392 | 20.012 | 0.7691 |
| verify 1 | 8 | qkv_gate | 14.384 | 18.497 | 0.7776 |
| verify 1 | 8 | o | 6.944 | 8.949 | 0.7759 |
| verify 1 | 8 | pooler | 21.088 | 22.610 | 0.9327 |
| verify 1 | 8 | unpooler | 6.976 | 9.071 | 0.7690 |
| verify 1 | 8 | router | 26.464 | 26.752 | 0.9892 |
| verify 16 | 128 | qkv_gate | 17.248 | 17.442 | 0.9889 |
| verify 16 | 128 | o | 7.488 | 7.528 | 0.9947 |
| verify 16 | 128 | pooler | 25.696 | 25.836 | 0.9946 |
| verify 16 | 128 | unpooler | 19.776 | 19.639 | 1.0070 |
| verify 16 | 128 | router | 16.224 | 28.766 | 0.5640 |
| verify 32 | 256 | qkv_gate | 19.808 | 19.830 | 0.9989 |
| verify 32 | 256 | o | 9.472 | 9.607 | 0.9860 |
| verify 32 | 256 | pooler | 30.768 | 30.704 | 1.0021 |
| verify 32 | 256 | unpooler | 24.336 | 24.322 | 1.0006 |
| verify 32 | 256 | router | 17.488 | 45.909 | 0.3809 |

### H2 cute-wgmma-verify32
- generator: cutlass_flashinfer
- expected_gain: verify-32 pooler, unpooler, and qkv_gate, which are at the 989 TFLOP/s roof. Bandwidth-only tuning cannot reach those floors.
- risk: high
- cost: high
- status: open
- note: CuTe or CUTLASS SM90 wgmma with TMA. Build it only after H1, so a long compile is not the first thing that can time out.

### H3 prepare-repack
- generator: triton
- expected_gain: none by itself. It is the layout H1 and H2 need.
- risk: low
- cost: medium
- status: open
- note: Pre-transpose, pad, and swizzle in prepare(). Lossless. Untimed.

### H4 shape-buckets
- generator: triton
- expected_gain: keeps the scored shapes fast without falling over on m=7, 33, 130, 512, 1000 or the nearby rerun shapes.
- risk: low
- cost: medium
- status: open
- note: Dispatch on ranges of m (1-8, 9-32, 33-256, 257+) and on layer. Not on the exact scored list.

### H5 router-hilo
- generator: triton
- expected_gain: small. The router is about 1.8-2.7 µs of a 29-52 µs floor sum.
- risk: medium
- cost: medium
- status: open
- note: FP32 for m<=8 with TF32 off, or the allowed hi+lo split at any m. Fuse the two GEMMs so x is read once. Tolerance is 1e-4.

### H6 one-launch
- generator: triton
- expected_gain: the in-graph launches that the 4 µs replay subtraction does not remove.
- risk: medium
- cost: medium
- status: open
- note: Persistent grid, reduction in the same launch. No host sync between split-K partials.

### H7 unpooler-slice
- generator: triton
- expected_gain: small. At m<=8 the unpooler floor is 1.7 µs because n is 1024, not 8192.
- risk: low
- cost: low
- status: open
- note: Honor config["n"]. Reading the other 7168 rows is wasted traffic and, if those rows are returned, a wrong shape.

### H8 wide-epilogue
- generator: raw_cuda
- expected_gain: a refinement of whichever of H1 or H2 wins, not a first kernel.
- risk: medium
- cost: medium
- status: open
- note: 128-bit loads and stores, swizzled shared memory. Only after a passing faster kernel exists.

## Dead ends

- Fusing the five layers. Each `kernel_fn` call is one layer.
- Leaving weights hot in L2. The judge cold-flushes L2 before every replay.
- Quantizing BF16 weights to FP8 or INT8. Probe rows fail.
- BF16-only router weights. They fail 1e-4.
- Doing the GEMM in `prepare` and replaying an output from `kernel_fn`. That is work outside the timed call.
- A second submit of the identical file to sample variance. The judge returns the earlier result. Do not add whitespace or comments to bust that cache. The organizers rerun the top entries themselves.
- Phase 2 (SGLang serving). Do not start it.

## What happens next

1. Login, `okc problems --json`, `okc init muse_bf16_proj`, baseline copy, lint, and the first check are complete. The check passed all nine correctness cases.
2. The off-board check recorded 0.9104× (5306.0 µs against SGLang's 4830.8 µs); this is slower than SGLang and is not a scored submission.
3. Candidate `v002_triton_splitk_small_m.py` passed all nine correctness cases, but scored-shape timing was worse overall (0.8145×). It is not a scored result and must not be submitted.
4. The per-shape timings for job `52ed320192494632` are in the table above; aggregate check rows are in `state/results.jsonl`. The raw OKC log remains local under ignored `state/raw/`. Two of 20 checks are used; no scored attempts are used.
5. Any further GPU check requires separate approval. Ask separately before any scored submit; do not submit v002 or the unchanged starter.

## Loopholes that will not be used

The judge says the same source returns its earlier result, `prepare` is untimed, and the case list is public. Using any of that to cache outputs, move the GEMM out of `kernel_fn`, or special-case known inputs is a disqualification. `okc check` is the supported way to experiment. It is not a loophole. It still spends the check budget.
