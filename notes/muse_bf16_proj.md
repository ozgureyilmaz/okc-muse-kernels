# muse_bf16_proj

Dense projections for Muse Spark Open, one H200, timed against SGLang. The unchanged starter measured 0.9104×. The current scored candidate is v027: 1.1718× (95% CI 1.1542–1.1893), 9/9 correctness, normalized 4122.6 µs, job `20251c5f78804f1f`. The same source checked at 1.1804× (95% CI 1.1626–1.1981). The live board at 2026-10-27T17:23:35.293809+00:00 shows rank 820/1160, with rank 10 at 1.5696×. Top ten remains unmet.

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

Limits reset at 00:00 UTC and can change with load. The latest refresh at 2026-10-27T17:23:34.407835+00:00 shows 16/20 checks and 1/10 scored submissions today. The 16-check authorization is exhausted and the four-check reserve is untouched. The user separately authorized one v027 scored promotion and merge into master after the earlier top-ten-only gate was not met. That promotion completed; no additional GPU runs are authorized, including after reset. All switches are false. Refresh limits before every future approved run.

| kind | per day | 20% reserve | may spend |
| --- | --- | --- | --- |
| scored `okc submit` | 10 | 2 | 8 |
| `okc check` (off the board, not a scored attempt) | 20 | 4 | 16 |

No contest-long cap showed up, only the daily cap. A run has 10 minutes including compilation. A timeout is charged. The same source returns its earlier result.

`state/config.json` has `submit_enabled` and `check_enabled` set false. `tools/submit.py` stays a dry-run until the matching switch and `--approved` are set. It blocks duplicate checks and scored submissions; a scored run may promote the exact hash of a passing check only when the correctness, aggregate, and confidence gates pass.

## Rules that change the kernel

- CUDA-graph replay: no `.item()`, `.cpu()`, `synchronize()`, printing a tensor, or data-dependent allocation or branches inside `kernel_fn`. Branching on shapes and `config` is allowed.
- Do not cache anything derived from `data`.
- BF16 weights stay BF16, or smaller only if lossless. Activations stay BF16.
- The image is `lmsysorg/sglang:dev-muse-spark-open`. Triton, CUTLASS, CuTe, FlashInfer, DeepGEMM, and TileLang are available. Nothing else installs. No network on the judge.
- `TASK.md` adds a pre-run guard: submissions must not patch timing, read harness frames or harness I/O/secrets, edit loaded modules, import the reference, start processes, or use the network.
- Submissions are Apache-2.0.

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

## Hypotheses

H1 and H9 were checked and rejected on performance. H10 passed its check gate and was submitted once after the exact-hash and limit refresh.

### H1 triton-splitk-small-m
- generator: triton
- expected_gain: the decode loads, if cuBLAS misses bandwidth at m=1..32. Pooler and qkv_gate are the bytes that matter.
- risk: medium
- cost: high
- status: checked-no-gain
- note: `variants/muse_bf16_proj/v002_triton_splitk_small_m.py` splits K for non-router m<=32, writes FP32 partials, then reduces and stores BF16 in a second kernel. All nine correctness cases passed, but the aggregate off-board ratio was 0.8145×. The implementation adds a second launch and FP32 scratch; this run does not isolate their individual cost. Router and m>32 use the starter path. Do not submit this candidate for a score.

### H11 deepgemm-sm90-dense
- generator: deep_gemm
- expected_gain: improve BF16 dense projections across decode-16/32 and verify-16/32, including qkv_gate, o, pooler, and unpooler.
- risk: high
- cost: high
- status: checked-passing
- note: Candidate `variants/muse_bf16_proj/v012_deepgemm_dense_sm90.py` uses the DeepGEMM BF16 NT API with `compiled_dims="nk"`, keeps v006 m=1 Hopper GEMVs, and retains a PyTorch fallback if the package/API is unavailable. Check job `72a41d512f964de7` passed 9/9 at 1.0301× (95% CI 1.0147–1.0456). Its graph-capture path worked in this run. Shape-level gains are uneven; see the recorded check and raw log.

### H12 router-hilo-kcat
- generator: pytorch_bf16
- expected_gain: replace the router FP32 m=1/m=8 path and split hi/lo path at larger m with one BF16 hi+lo matrix product; router is 26.4 µs at m=8 in the scored v006 run.
- risk: medium
- cost: low
- status: checked-passing
- note: Candidate `variants/muse_bf16_proj/v011_router_concat_hilo.py` prepares BF16 hi/lo weights concatenated along K, then forms a fresh `[x, x]` activation inside `kernel_fn`. The allowed router hi+lo exception applies at every m. Check job `88d7246abe254117` passed 9/9 at 1.1095× (95% CI 1.0929–1.1262), including the strict router tolerance. The router-only path cannot reach top ten.

### H13 router-fp32-gemv-small-m
- generator: triton
- expected_gain: improve the m=1 and m=8 router loads while using the original FP32 weights and exact FP32 accumulation.
- risk: medium
- cost: medium
- status: checked-passing
- note: Candidate `variants/muse_bf16_proj/v013_router_fp32_gemv.py` uses one warp-reduced FP32 GEMV kernel for m<=8 and leaves other v006 paths unchanged. Check job `cdb5ab4185ce47d6` passed 9/9 at 1.0735× (95% CI 1.0574–1.0896). The m=1 and m=8 router GEMVs improved, but the aggregate is below v011.

### H14 quack-fixed-sm90-dense
- generator: quack_cute_sm90_fixed_dense
- expected_gain: improve dense BF16 paths using fixed QuACK tiles after compile-cache preparation.
- risk: high
- cost: high
- status: checked-no-gain
- note: Corrected candidate `variants/muse_bf16_proj/v014_quack_dense_sm90.py` matches QuACK v0.6.5's positional compile-cache key. Check job `184ffdecad084e0c` passed 9/9 but measured 0.5976× (95% CI 0.5886–0.6066); dense m>8 timings regressed substantially.

### H15 integrate-router-and-deepgemm
- generator: pytorch_bf16_router_concat_deepgemm_sm90
- expected_gain: combine the measured router and dense-layer paths in one candidate.
- risk: medium
- cost: low
- status: checked-below-live-threshold
- note: Candidate `variants/muse_bf16_proj/v015_router_concat_deepgemm_sm90.py` combines v011's router hi/lo concatenation with v012's DeepGEMM dense dispatch and preserves v006 m=1 GEMVs. Check job `01128730b46b4d96` passed 9/9 at 1.1265× (95% CI 1.1096–1.1434), below the live rank-10 threshold.

### H16 router-fp32-registeraccum
- generator: triton_fp32_router_register_accum
- expected_gain: keep the small-m router's per-K-lane FP32 accumulators in registers and reduce once.
- risk: medium
- cost: medium
- status: checked-passing
- note: v016 passed 9/9 as job `60c2730723084074` at 1.1283× (95% CI 1.1114–1.1453). Its exact FP32 small-router branch is retained in the final v027 composition, which also passed 9/9.

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

### H9 low-m-gemv-and-router
- generator: triton
- expected_gain: improve the bandwidth-bound decode and verify-1 projections; retain SGLang's router precision path.
- risk: high
- cost: medium
- status: checked-no-gain
- note: v003 used one tiled dot kernel for m<=32 plus the router hi/lo path; v004 used a vector reduction for m<=8 and F.linear otherwise. Both passed the nine correctness cases but had aggregate ratios below 1.0. See the raw logs and result rows; do not submit either candidate.

## New candidate check outcomes

All runs below were logged through `tools/submit.py`. Each check passed all nine correctness cases. v003–v005 missed the 1.02× point gate; v006 met both the point and confidence gates.

| Candidate | Hash | Job | Aggregate | 95% interval | Score time |
| --- | --- | --- | ---: | --- | ---: |
| v003 `v003_triton_onepass_hilo.py` | `6053432cc080365b52a9f25853b68be2fb74567fed0f3df30d4a7bec2c729eab` | `7228123952934a9a` | 0.5677× | 0.5592–0.5762 | 8509.1 µs |
| v004 `v004_bf16_gemv_hilo.py` | `1f656f39743fcacbc01a8082859a323c165c8c6d91c9cff6ddde08b16eae878b` | `3c076f2a83cd4347` | 0.7145× | 0.7037–0.7252 | 6761.4 µs |
| v005 `v005_hopper_gemv_hilo.py` | `0ea01bcb7dc410ca65fd87644b8cad4c6f7edec820d76ed6ae3e11ad600af68d` | `a981192f76b64e01` | 1.0174× | 1.0021–1.0326 | 4748.3 µs |
| v006 `v006_hopper_gemv_router_addmm.py` | `6df4c93a5da945667530c39f5439e6332a1d2240fb1d573158a684c6384444b1` | `c5deac6f18bb456e` | 1.0225× | 1.0072–1.0378 | 4724.4 µs |

v006 passed 9/9 correctness cases. It keeps the v005 Hopper GEMV path for eligible m=1 projections and folds the router hi/lo addition into `torch.addmm`. Its scored job `00e22ad9869b4bb7` also passed all correctness cases and measured 1.0139× (95% CI 0.9987–1.0291; score time 4764.5 µs). The scored-run interval includes 1.0, so treat the point improvement as provisional. The exact hash and full 30-shape check medians are below. Raw evaluator records are in ignored `state/raw/`; `state/results.jsonl` contains aggregate rows.

| Load | m | Layer | SGLang µs | v006 µs | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| decode 1 | 1 | qkv_gate | 14.304 | 13.378 | 1.0692× |
| decode 1 | 1 | o | 6.752 | 6.036 | 1.1187× |
| decode 1 | 1 | pooler | 21.504 | 18.889 | 1.1384× |
| decode 1 | 1 | unpooler | 6.784 | 5.997 | 1.1313× |
| decode 1 | 1 | router | 10.528 | 10.462 | 1.0063× |
| decode 16 | 16 | qkv_gate | 14.224 | 13.995 | 1.0163× |
| decode 16 | 16 | o | 6.960 | 6.602 | 1.0543× |
| decode 16 | 16 | pooler | 21.776 | 21.512 | 1.0123× |
| decode 16 | 16 | unpooler | 18.800 | 18.429 | 1.0201× |
| decode 16 | 16 | router | 14.800 | 14.513 | 1.0198× |
| decode 32 | 32 | qkv_gate | 14.288 | 14.345 | 0.9960× |
| decode 32 | 32 | o | 6.944 | 6.897 | 1.0069× |
| decode 32 | 32 | pooler | 21.920 | 21.857 | 1.0029× |
| decode 32 | 32 | unpooler | 18.992 | 18.679 | 1.0167× |
| decode 32 | 32 | router | 15.424 | 15.224 | 1.0132× |
| verify 1 | 8 | qkv_gate | 14.480 | 14.440 | 1.0028× |
| verify 1 | 8 | o | 7.008 | 6.927 | 1.0116× |
| verify 1 | 8 | pooler | 21.536 | 21.379 | 1.0074× |
| verify 1 | 8 | unpooler | 7.136 | 7.093 | 1.0060× |
| verify 1 | 8 | router | 26.400 | 26.312 | 1.0033× |
| verify 16 | 128 | qkv_gate | 17.120 | 16.937 | 1.0108× |
| verify 16 | 128 | o | 7.568 | 7.663 | 0.9875× |
| verify 16 | 128 | pooler | 25.104 | 24.793 | 1.0125× |
| verify 16 | 128 | unpooler | 19.776 | 19.600 | 1.0090× |
| verify 16 | 128 | router | 16.496 | 16.384 | 1.0068× |
| verify 32 | 256 | qkv_gate | 19.712 | 19.658 | 1.0028× |
| verify 32 | 256 | o | 9.376 | 9.296 | 1.0086× |
| verify 32 | 256 | pooler | 30.800 | 30.658 | 1.0046× |
| verify 32 | 256 | unpooler | 24.256 | 24.245 | 1.0004× |
| verify 32 | 256 | router | 17.568 | 17.472 | 1.0055× |

### H10 hopper-gemv-router-addmm
- generator: sglang_hopper_gemv_torch_addmm
- expected_gain: preserve the v005 decode-1 projection gains and fold a separate router accumulation into the second matrix operation.
- risk: medium
- cost: low
- status: checked-passing
- note: v006 passed 9/9 cases; check aggregate 1.0225× (95% CI 1.0072–1.0378), then scored job `00e22ad9869b4bb7` passed at 1.0139× (95% CI 0.9987–1.0291). The 16:29 UTC readback showed rank 749/753; the latest 16:40 UTC readback shows 751/755.

## Rank optimization follow-up (2026-10-24, historical)

The follow-up required a new check at least 1.04× aggregate with the 95% CI lower bound above 1.02× before another scored run. Four unique checks were used; none met that gate. No additional scored submission was sent. The v006 scored result remains the account's recorded score at 1.0139×. The 2026-10-24 16:40 UTC leaderboard readback (19:40 Europe/Istanbul) showed rank 751/755.

| Candidate | Hash | Job | Correctness | Aggregate | 95% interval | Score time |
| --- | --- | --- | --- | ---: | --- | ---: |
| v007 `v007_wgmma_verify.py` | `08c81cf4250dcbc0560db621e8da1ed9dd45a5e95044c60bd8e70a0ab434bae7` | `b717529acb4048e8` | 9/9 pass | 0.7964× | 0.7845–0.8084 | 6065.5 µs |
| v008 `v008_wgmma_pipeline_verify.py` | `01d426e45e8bb1c6f9be45a9a729de6a6ac44ea461f92de01841e1f157d47ed8` | `b723b3c6cffb4bfe` | 8/9 pass; compile failure | no score | — | — |
| v009 `v009_wgmma_rank3_buffers.py` | `131125b48e2de3345f28f0b98dc9b8d6ceaed1308506bf49577763da2104c0` | `542953ee793149f0` | 8/9 pass; compile failure | no score | — | — |
| v010 `v010_wgmma_official_pipeline.py` | `6f773a4f5b5b100964f8389a84b3e284162b5f55dc4b70191eb72a121764285e` | `1e6ada23626e415a` | 9/9 pass | 0.8034× | 0.7914–0.8155 | 6012.8 µs |

v008 failed because Gluon could not merge an FP32 accumulator with the asynchronous WGMMA accumulator through a conditional. v009 failed because its rank-3 shared layout did not slice to the expected rank-2 tile. v010 used the official two-stage TMA/WGMMA buffering pattern and compiled, but the verify projections remained much slower than SGLang. The raw logs for v008–v010 are in ignored `state/raw/`. The exact v007 source is restored in `variants/muse_bf16_proj/v007_wgmma_verify.py` from this task's generation transcript and its SHA-256 was verified. Its raw log remains in the shared checkout used by the public-cleanup task; job `b717529acb4048e8` is readable through `okc status`.

Per-load ratios from the ratio of summed paired medians:

| Load | v007 | v010 |
| --- | ---: | ---: |
| decode 1 | 1.0673× | 1.0853× |
| decode 16 | 0.9977× | 1.0026× |
| decode 32 | 1.0029× | 0.9938× |
| verify 1 | 1.0080× | 1.0014× |
| verify 16 | 0.4569× | 0.5977× |
| verify 32 | 0.5189× | 0.4155× |

### v007 per-shape check timings

| Load | m | Layer | SGLang µs | Candidate µs | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| decode 1 | 1 | qkv_gate | 14.064 | 13.364 | 1.0524× |
| decode 1 | 1 | o | 6.816 | 6.260 | 1.0888× |
| decode 1 | 1 | pooler | 21.440 | 19.335 | 1.1089× |
| decode 1 | 1 | unpooler | 7.072 | 6.593 | 1.0726× |
| decode 1 | 1 | router | 10.336 | 10.408 | 0.9931× |
| decode 16 | 16 | qkv_gate | 13.728 | 13.767 | 0.9972× |
| decode 16 | 16 | o | 6.896 | 6.977 | 0.9884× |
| decode 16 | 16 | pooler | 21.232 | 21.448 | 0.9899× |
| decode 16 | 16 | unpooler | 18.560 | 18.574 | 0.9992× |
| decode 16 | 16 | router | 15.216 | 15.042 | 1.0116× |
| decode 32 | 32 | qkv_gate | 14.640 | 14.600 | 1.0028× |
| decode 32 | 32 | o | 7.088 | 7.105 | 0.9976× |
| decode 32 | 32 | pooler | 21.936 | 21.937 | 1.0000× |
| decode 32 | 32 | unpooler | 19.040 | 19.080 | 0.9979× |
| decode 32 | 32 | router | 15.424 | 15.178 | 1.0162× |
| verify 1 | 8 | qkv_gate | 14.288 | 13.993 | 1.0211× |
| verify 1 | 8 | o | 6.768 | 6.680 | 1.0132× |
| verify 1 | 8 | pooler | 21.536 | 21.443 | 1.0043× |
| verify 1 | 8 | unpooler | 6.944 | 6.870 | 1.0107× |
| verify 1 | 8 | router | 26.096 | 26.048 | 1.0019× |
| verify 16 | 128 | qkv_gate | 17.008 | 59.872 | 0.2841× |
| verify 16 | 128 | o | 7.232 | 7.152 | 1.0112× |
| verify 16 | 128 | pooler | 25.184 | 67.220 | 0.3747× |
| verify 16 | 128 | unpooler | 20.000 | 37.035 | 0.5400× |
| verify 16 | 128 | router | 16.256 | 16.248 | 1.0005× |
| verify 32 | 256 | qkv_gate | 19.648 | 60.981 | 0.3222× |
| verify 32 | 256 | o | 9.376 | 9.321 | 1.0059× |
| verify 32 | 256 | pooler | 30.928 | 67.639 | 0.4572× |
| verify 32 | 256 | unpooler | 23.968 | 40.139 | 0.5971× |
| verify 32 | 256 | router | 17.584 | 17.534 | 1.0029× |

### v010 per-shape check timings

| Load | m | Layer | SGLang µs | Candidate µs | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| decode 1 | 1 | qkv_gate | 14.112 | 13.292 | 1.0617× |
| decode 1 | 1 | o | 6.640 | 5.938 | 1.1182× |
| decode 1 | 1 | pooler | 21.408 | 18.975 | 1.1282× |
| decode 1 | 1 | unpooler | 6.880 | 6.174 | 1.1143× |
| decode 1 | 1 | router | 10.304 | 10.300 | 1.0004× |
| decode 16 | 16 | qkv_gate | 14.128 | 14.088 | 1.0028× |
| decode 16 | 16 | o | 6.704 | 6.572 | 1.0200× |
| decode 16 | 16 | pooler | 21.184 | 21.353 | 0.9921× |
| decode 16 | 16 | unpooler | 18.656 | 18.633 | 1.0013× |
| decode 16 | 16 | router | 14.784 | 14.614 | 1.0116× |
| decode 32 | 32 | qkv_gate | 14.240 | 14.523 | 0.9805× |
| decode 32 | 32 | o | 6.816 | 6.824 | 0.9988× |
| decode 32 | 32 | pooler | 21.808 | 22.029 | 0.9900× |
| decode 32 | 32 | unpooler | 18.752 | 18.968 | 0.9886× |
| decode 32 | 32 | router | 15.520 | 15.270 | 1.0163× |
| verify 1 | 8 | qkv_gate | 14.144 | 14.022 | 1.0087× |
| verify 1 | 8 | o | 6.704 | 6.687 | 1.0026× |
| verify 1 | 8 | pooler | 21.472 | 21.517 | 0.9979× |
| verify 1 | 8 | unpooler | 6.816 | 6.827 | 0.9983× |
| verify 1 | 8 | router | 26.112 | 26.088 | 1.0009× |
| verify 16 | 128 | qkv_gate | 16.880 | 35.021 | 0.4820× |
| verify 16 | 128 | o | 7.136 | 7.091 | 1.0064× |
| verify 16 | 128 | pooler | 24.560 | 41.647 | 0.5897× |
| verify 16 | 128 | unpooler | 19.888 | 41.692 | 0.4770× |
| verify 16 | 128 | router | 16.320 | 16.401 | 0.9951× |
| verify 32 | 256 | qkv_gate | 20.016 | 67.813 | 0.2952× |
| verify 32 | 256 | o | 9.136 | 8.886 | 1.0282× |
| verify 32 | 256 | pooler | 31.008 | 75.614 | 0.4101× |
| verify 32 | 256 | unpooler | 24.352 | 75.466 | 0.3227× |
| verify 32 | 256 | router | 17.408 | 17.512 | 0.9941× |

### Optimization sources

- [Pinned SGLang Hopper BF16 GEMV](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py): keeps the proven m=1 decode path.
- [Pinned SGLang unquantized BF16 dispatcher](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/unquant.py): the H200 baseline falls through to `F.linear` unless optional backends apply.
- [Triton Gluon WGMMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html) and [TMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html): asynchronous MMA, barriers, and double buffering informed v007–v010.
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) and [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): tile reuse, grid size, and occupancy informed shape selection.

## Top-ten check round (2026-10-27)

The managing chat received direct human approval for at most 16 total checks today, preserving four of the daily 20 checks, plus one conditional scored submission. The condition was 9/9 correctness on the exact checked hash and a check 95% CI lower bound above the live rank-10 speedup at promotion. No further checks are authorized after these 16, including after a daily reset.

All 16 unique candidates below ran once through `tools/submit.py` after limit refreshes. All passed 9/9; a passing test does not imply a performance gain. Every result is tied to its logged source hash; the checked file bytes were verified at this checkpoint. No scored submission was sent during this check round. Its checkpoint refresh showed 16/20 checks and 0/10 scored submissions, all switches false, and no pending jobs. The final check finished at 16:53:12 UTC (19:53:12 Europe/Istanbul).

| Candidate | SHA-256 prefix | Job | Correctness | Speedup | 95% CI |
| --- | --- | --- | --- | ---: | --- |
| [v011](../variants/muse_bf16_proj/v011_router_concat_hilo.py) | `ef6ef68b0e2d` | `88d7246abe254117` | 9/9 | 1.1095× | 1.0929–1.1262 |
| [v012](../variants/muse_bf16_proj/v012_deepgemm_dense_sm90.py) | `8c5aa696e25d` | `72a41d512f964de7` | 9/9 | 1.0301× | 1.0147–1.0456 |
| [v013](../variants/muse_bf16_proj/v013_router_fp32_gemv.py) | `794f45e9f145` | `cdb5ab4185ce47d6` | 9/9 | 1.0735× | 1.0574–1.0896 |
| [v015](../variants/muse_bf16_proj/v015_router_concat_deepgemm_sm90.py) | `48c40a8941e9` | `01128730b46b4d96` | 9/9 | 1.1265× | 1.1096–1.1434 |
| [v014](../variants/muse_bf16_proj/v014_quack_dense_sm90.py) | `b9df97da9edb` | `184ffdecad084e0c` | 9/9 | 0.5976× | 0.5886–0.6066 |
| [v016](../variants/muse_bf16_proj/v016_router_fp32_registeraccum.py) | `a4dde2408cdc` | `60c2730723084074` | 9/9 | 1.1283× | 1.1114–1.1453 |
| [v018](../variants/muse_bf16_proj/v018_n_tiled_bmm_dense.py) | `2d826fc411fd` | `99ad7372ac854123` | 9/9 | 0.8696× | 0.8565–0.8826 |
| [v017](../variants/muse_bf16_proj/v017_router_registeraccum_deepgemm_sm90.py) | `d2691974adca` | `519bb14bda964807` | 9/9 | 1.1477× | 1.1305–1.1649 |
| [v020](../variants/muse_bf16_proj/v020_lowm_dense_registeraccum.py) | `d59ba9f7a7ec` | `bbc03a5175eb4732` | 9/9 | 0.9171× | 0.9033–0.9308 |
| [v019](../variants/muse_bf16_proj/v019_quack_wide_splitk.py) | `e430de8271d3` | `1385339c0f9944c4` | 9/9 | 0.8999× | 0.8864–0.9134 |
| [v023](../variants/muse_bf16_proj/v023_flashinfer_tinygemm_bf16.py) | `e59741c210a6` | `bd296596f2934bb8` | 9/9 | 1.1276× | 1.1106–1.1445 |
| [v025](../variants/muse_bf16_proj/v025_flashinfer_tinygemm_all_small_m.py) | `588a311f48a3` | `277a6257ded54938` | 9/9 | 1.1110× | 1.0943–1.1276 |
| [v024](../variants/muse_bf16_proj/v024_router_hilo_splitk_mma.py) | `66f65784a064` | `96b8a4c3c0674044` | 9/9 | 1.1046× | 1.0880–1.1211 |
| [v026](../variants/muse_bf16_proj/v026_transposed_dense.py) | `ddbb87de3c23` | `e040c762228c4812` | 9/9 | 1.1146× | 1.0978–1.1313 |
| [v028](../variants/muse_bf16_proj/v028_lowm_tensorcore_splitk.py) | `5cfa2d84d401` | `aaeaf188253f4a4a` | 9/9 | 1.0882× | 1.0718–1.1045 |
| [v027](../variants/muse_bf16_proj/v027_selective_measured_paths.py) | `e696a2e8f05d` | `9c28d3d304da4276` | 9/9 | 1.1804× | 1.1626–1.1981 |

Best check: v027, SHA-256 `e696a2e8f05daf5f3244f89371dfc32579486c32c53717ecee96421a5614fe30`, job `9c28d3d304da4276`, 1.180353×, 95% CI 1.162647–1.198058, normalized 4092.675 µs. That is 15.28% less normalized time than SGLang's 4830.8 µs. The live board read at 2026-10-27T16:53:30.411355+00:00 reports public rank 1153/1158, score 1.0139×, rank 10 1.5696× / 3077.632 µs, and leader 1.6367×. v027's lower CI was below rank 10, so it did not meet the check round's conditional promotion gate. A later direct user instruction separately authorized its scored promotion, recorded below. It needs approximately 24.80% additional normalized latency reduction. The top-ten objective remains unmet.

### Final dispatch and evidence

| Layer/range | v027 path | Measured evidence |
| --- | --- | --- |
| Router m≤8 | Original FP32 weights, per-K-lane FP32 register accumulation, one reduction | v016; v027 m=1 5.500 µs and m=8 6.826 µs |
| Router 8<m≤64 | BF16 hi/lo Tensor Core split-K, FP32 partials and output | v024; v027 m=16 7.016 µs and m=32 8.076 µs |
| Router m>64 | Fresh `[x,x]` times prepared concatenated hi/lo weights, FP32 output | v011; v027 m=128 11.471 µs and m=256 13.280 µs |
| Unpooler 1≤m≤8; pooler 2≤m≤8 | FlashInfer TinyGEMM BF16, FP32 accumulation, PDL disabled | v023/v025; v027 unpooler m=1 4.640 µs and m=8 4.803 µs |
| Other eligible dense m=1 | Existing SGLang Hopper GEMV | v006 and component checks |
| Pooler 8<m≤192 | Native DeepGEMM SM90 BF16 | v012/v015/v017; v027 m=16/32/128 18.639/19.306/21.437 µs |
| Remaining dense calls | `F.linear` | Preserves baseline paths where alternatives regressed |

These are broad shape/layer ranges, including unscored sizes, rather than exact scored-shape matching. Weight hi/lo preparation is the router's allowed precision exception. Dense weights remain BF16, activations remain BF16, and no input-derived state persists across calls. Preparation compiles/loads modules and transforms weights without a dummy GEMM.

### Rejected mechanisms

- QuACK narrow tiles (v014) and wider parallel split-K (v019) passed but lost on aggregate. Do not repeat them without a new mechanism and resource argument.
- N-tiled batched GEMM (v018) was graph-compatible and numerically correct, but qkv m=16/32 took about 66–68 µs. Its borrowed output layout and zero-stride input were supported; the problem was measured speed.
- Dense SIMT register accumulation (v020) did not improve aggregate performance.
- Applying TinyGEMM to every small-M layer (v023/v025) severely regressed O and did not improve qkv. Retain only the measured pooler/unpooler ranges.
- Router split-K (v024) improved m=16/32 but regressed m=128/256 against concatenation; retain it only for 8<m≤64.
- Transposing dense GEMM orientation/output (v026) did not produce a useful new range beyond selected native DeepGEMM.
- Wide-N Triton Tensor Core dense split-K (v028) still paid for a second launch and FP32 partial traffic. O m=8/16/32 took about 9.7–10.0 µs versus about 7 µs; full unpooler m=16/32 took about 21–22 µs versus about 18.5 µs. Its pooler changes were around 1%, inside noise.

### v027 per-shape timings

Judge-adjusted medians in microseconds. The normalized aggregate uses the geometric mean of per-load ratios, not a geometric mean of the 30 individual layer ratios. Each layer has weight 62.

| Load | m | Layer | SGLang µs | v027 µs | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| decode 1 | 1 | qkv_gate | 14.080 | 13.379 | 1.0524× |
| decode 1 | 1 | o | 6.592 | 5.997 | 1.0992× |
| decode 1 | 1 | pooler | 21.424 | 18.995 | 1.1279× |
| decode 1 | 1 | unpooler | 6.624 | 4.640 | 1.4275× |
| decode 1 | 1 | router | 10.304 | 5.500 | 1.8735× |
| decode 16 | 16 | qkv_gate | 13.920 | 14.082 | 0.9885× |
| decode 16 | 16 | o | 6.112 | 6.088 | 1.0039× |
| decode 16 | 16 | pooler | 21.472 | 18.639 | 1.1520× |
| decode 16 | 16 | unpooler | 18.464 | 18.592 | 0.9931× |
| decode 16 | 16 | router | 14.848 | 7.016 | 2.1162× |
| decode 32 | 32 | qkv_gate | 14.208 | 14.581 | 0.9744× |
| decode 32 | 32 | o | 6.784 | 6.929 | 0.9791× |
| decode 32 | 32 | pooler | 22.112 | 19.306 | 1.1453× |
| decode 32 | 32 | unpooler | 18.816 | 18.986 | 0.9911× |
| decode 32 | 32 | router | 15.136 | 8.076 | 1.8743× |
| verify 1 | 8 | qkv_gate | 14.400 | 14.596 | 0.9866× |
| verify 1 | 8 | o | 6.720 | 6.647 | 1.0110× |
| verify 1 | 8 | pooler | 20.992 | 18.706 | 1.1222× |
| verify 1 | 8 | unpooler | 7.088 | 4.803 | 1.4758× |
| verify 1 | 8 | router | 26.032 | 6.826 | 3.8139× |
| verify 16 | 128 | qkv_gate | 17.472 | 17.472 | 1.0000× |
| verify 16 | 128 | o | 7.200 | 7.121 | 1.0112× |
| verify 16 | 128 | pooler | 24.640 | 21.437 | 1.1494× |
| verify 16 | 128 | unpooler | 19.760 | 19.694 | 1.0033× |
| verify 16 | 128 | router | 16.496 | 11.471 | 1.4380× |
| verify 32 | 256 | qkv_gate | 19.360 | 19.497 | 0.9930× |
| verify 32 | 256 | o | 9.152 | 9.088 | 1.0070× |
| verify 32 | 256 | pooler | 30.496 | 30.582 | 0.9972× |
| verify 32 | 256 | unpooler | 24.384 | 24.138 | 1.0102× |
| verify 32 | 256 | router | 17.312 | 13.280 | 1.3036× |

| Load | Ratio of summed times | v027 total ×62 |
| --- | ---: | ---: |
| decode 1 | 1.2167× | 3007.66 µs |
| decode 16 | 1.1614× | 3993.86 µs |
| decode 32 | 1.1352× | 4208.40 µs |
| verify 1 | 1.4586× | 3197.74 µs |
| verify 16 | 1.1085× | 4786.10 µs |
| verify 32 | 1.0426× | 5988.32 µs |

### Primary source compatibility

- [Pinned SGLang dependency versions](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/pyproject.toml): FlashInfer 0.7.0.post1, sgl-deep-gemm 0.2.1.post1, QuACK 0.6.5, and CUTLASS DSL 4.8.0 informed exact API research.
- [FlashInfer TinyGEMM at the pinned tag](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/gemm/routergemm.py): BF16 API and compile/load entry point. Preparation preloads its module; the timed call supplies fresh input/output tensors and `use_pdl=False`.
- [DeepGEMM public API](https://github.com/sgl-project/DeepGEMM/blob/v0.2.1.post1/deep_gemm/__init__.py) and [SM90 implementation](https://github.com/sgl-project/DeepGEMM/blob/v0.2.1.post1/csrc/apis/gemm.hpp): `bf16_gemm_nt` with native SM90 dispatch selected by the library's deterministic setting. This setting is global to DeepGEMM; it does not change Torch's global numerical mode. Actual check profiles confirmed native SM90 kernels.
- [QuACK GEMM](https://github.com/Dao-AILab/quack/blob/v0.6.5/quack/gemm.py) and [cache wrapper](https://github.com/Dao-AILab/quack/blob/v0.6.5/quack/cache/jit.py): fixed-config compilation used identical positional keys in preparation and execution. Correcting compatibility did not make these candidates faster.
- [PyTorch BMM tests](https://github.com/pytorch/pytorch/blob/v2.14.1/test/test_linalg.py) and [CUDA BLAS implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp): supported expanded input and borrowed output layout were checked before rejecting v018 on performance.

The remaining target and proposed experiments are in [the next-round technical plan](top10-next-round.md). They require new GPU-run authorization and must preserve the daily reserve. None is a top-ten performance claim.

## Scored promotion (2026-10-27)

The user explicitly requested submitting v027 and then merging the work into GitHub master. This superseded the earlier top-ten-only condition for one exact-hash scored promotion. Limits were refreshed, the prior 9/9 check and unchanged source hash were verified, and `tools/submit.py --kind submit --approved` ran once. Check and submit switches were restored false afterward.

Scored job `20251c5f78804f1f` passed 9/9 and measured **1.171773×** (95% CI **1.154196–1.189349**), normalized **4122.643 µs**. The source is `e696a2e8f05daf5f3244f89371dfc32579486c32c53717ecee96421a5614fe30`, identical to the check. It finished at 2026-10-27T17:22:42.165847+00:00; the live read at 2026-10-27T17:23:35.293809+00:00 confirmed rank **820/1160**, public speedup **1.1718×**. This is 14.66% less normalized time than SGLang; the current rank-10 point requires another 25.35% reduction. The check/scored point difference is within their reported confidence intervals.

The ledger now contains two historical scored jobs in total: v006 and v027. Today's counters are 16/20 checks and 1/10 scored submissions. The saved best kernel under `state/best/muse_bf16_proj/submission.py` is byte-identical to v027. No second scored run or new check was sent.

Local workflow repair: scored result rows use `kind="submit"`, while summaries previously recognized only legacy `"scored"`. Both labels now count as scored results, and a best-source update compares against earlier ledger entries so an already-appended new result can replace the saved best. Regression tests cover actual submit labels, check exclusion, promotion after append, and retention of the better kernel after a lower score. The local workflow suite passes 18 tests.

| Load | Ratio of scored load sums | Scored v027 total ×62 |
| --- | ---: | ---: |
| decode 1 | 1.1994× | 3058.57 µs |
| decode 16 | 1.1594× | 4022.18 µs |
| decode 32 | 1.1425× | 4152.90 µs |
| verify 1 | 1.4180× | 3292.14 µs |
| verify 16 | 1.1034× | 4790.15 µs |
| verify 32 | 1.0413× | 6009.18 µs |

Full scored output is local and ignored at `state/raw/submit-e696a2e8f05d.log`; the committed ledger records the exact source, job, aggregate, and correctness.

## Dead ends

- Fusing the five layers. Each `kernel_fn` call is one layer.
- Leaving weights hot in L2. The judge cold-flushes L2 before every replay.
- Quantizing BF16 weights to FP8 or INT8. Probe rows fail.
- BF16-only router weights. They fail 1e-4.
- Doing the GEMM in `prepare` and replaying an output from `kernel_fn`. That is work outside the timed call.
- A second submit of the identical file to sample variance. The judge returns the earlier result. Do not add whitespace or comments to bust that cache. The organizers rerun the top entries themselves.
- Phase 2 (SGLang serving). Do not start it.

## What happens next

1. Preserve the current scored v027 source `e696a2e8f05daf5f3244f89371dfc32579486c32c53717ecee96421a5614fe30` and job `20251c5f78804f1f`: 1.1718×, 9/9. Its best-source copy and metadata are in `state/best/muse_bf16_proj/`.
2. v006 is historical; do not resubmit either recorded source. The public leaderboard read-back is 820/1160 at 1.1718×. Top ten remains an unmet objective.
3. All 16 authorized checks and the separately approved single scored promotion are used; four checks remain reserved. Further GPU work needs fresh explicit approval. No job is pending and check, submit, and Phase 2 switches are false.
4. The [dense-layer technical plan](top10-next-round.md) defines future experiments. Preserve the challenge rules and stop deadline; no new check or scored run is implied by that plan.

## Loopholes that will not be used

The judge says the same source returns its earlier result, `prepare` is untimed, and the case list is public. Using any of that to cache outputs, move the GEMM out of `kernel_fn`, or special-case known inputs is a disqualification. `okc check` is the supported way to experiment. It is not a loophole. It still spends the check budget.
