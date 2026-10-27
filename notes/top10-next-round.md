# Muse BF16 top-ten gap: next-round checkpoint

## Where the measured work stands

The best completed check is v027 at **1.18035×** (95% CI 1.16265–1.19806; normalized time 4092.7 µs). The live board read at 2026-10-27 16:53:30 UTC put the rank-10 threshold at **1.5696× / 3077.632 µs**. Closing that time gap requires another **24.8% latency reduction** from v027, or about a 33% increase in score. v027 is a useful composition result, but it remains below the top-ten target.

After this analysis, the user separately authorized one scored promotion. Identical v027 source passed 9/9 at **1.17177×** (95% CI 1.15420–1.18935; 4122.6 µs). The 2026-10-27T17:23:35.293809+00:00 read-back confirmed rank **820/1160**. From that scored point, the rank-10 gap is **25.3%** and the 1.60× target needs **26.8%** further normalized latency reduction. The check-based table below remains a planning reference, not the public scored measurement.

The geometric mean weights each of the six loads equally. The table shows weighted load sums from v027’s own benchmark records, an even per-load allocation for 1.5696×, and the task’s theoretical floors. The even allocation is a planning reference; the geometric-mean score permits uneven gains.

| Load | v027 total | Even 1.5696× budget | TASK floor | v027 ratio |
|---|---:|---:|---:|---:|
| Decode 1 | 3008 µs | 2331 µs | 1792 µs | 1.217× |
| Decode 16 | 3994 µs | 2955 µs | 2573 µs | 1.161× |
| Decode 32 | 4208 µs | 3044 µs | 2585 µs | 1.135× |
| Verify 1 | 3198 µs | 2972 µs | 1798 µs | 1.459× |
| Verify 16 | 4786 µs | 3380 µs | 2716 µs | 1.108× |
| Verify 32 | 5988 µs | 3978 µs | 3212 µs | 1.043× |

This allocation would remove about 56%, 73%, 72%, 16%, 68%, and 72% of each load’s measured time above its floor, respectively. Verify 32 is the weakest v027 load. Its pooler is 31.7% of the candidate load, qkv_gate 20.2%, and unpooler 25.0%; the dense layers together dominate. Across the six loads, router is only about 11–15% of v027 time. An optimistic replacement of every router call by its TASK floor gives a score ceiling near **1.306×**, so router work alone cannot close the gap.

## What the checks say about the hardware paths

- v017’s DeepGEMM-routed dense shapes (m>8) each show one native SM90 GEMM kernel. At verify-32 m=256, qkv_gate is 20.06 µs versus a 19.87 µs baseline, pooler 31.63 versus 30.80, and unpooler 24.03 versus 24.46. These are near baseline and still above the compute floors (9.8/17.4/17.4 µs). The profile identifies the GEMM as the whole kernel time, but has no occupancy or stall counters; the exact limiter is unknown.
- v027 improves the shared router and the selected low-M pooler/unpooler paths. It also loses v017’s m=128 qkv_gate DeepGEMM win: v017 measured 15.97 µs versus 16.98 µs baseline, while v027 falls back to 17.47 µs against 17.47 µs. That is a small, evidence-backed dispatch refinement, not a top-ten solution.
- The earlier #15 v028 check finished at 1.08815×. Its explicit K-partial path did not improve the dense shapes: O measured 9.93/9.99 µs at m=16/32 versus 7.20/6.64 µs baseline; unpooler was 20.95/22.02 versus 18.53/18.43. Profiles show the partial-GEMM kernel uses roughly 76–94% of GPU time, with a 1.3–1.8 µs reduction. The short-K O splits have only two BK tiles each. Do not rerun this split-K design or tune only its reducer.
- v025’s all-small-M TinyGEMM path regressed O at m=1 and m=8 (about 10.7/10.8 µs versus 7.1/6.8 baseline); v027 correctly selects TinyGEMM only for measured pooler/unpooler wins. v026’s transposed `torch.mm` changes layout, not the work or traffic, and measured 1.11457×. Do not repeat all-shape TinyGEMM or the same layout flip.

Evidence: the committed [run ledger](../state/results.jsonl), [all 30 final timings and dispatch](muse_bf16_proj.md#top-ten-check-round-2026-10-27), and each candidate's source under `variants/muse_bf16_proj/`. Raw evaluator records and profiler traces stay local and ignored: `state/raw/check-<first-12-source-hash>.log` and `muse_bf16_proj/profiles/submission.<job>.txt`. For v027 these identify job `9c28d3d304da4276`. The generated local `muse_bf16_proj/TASK.md` is authoritative for floors and the contract. Profiler kernel times differ from the cold-L2 adjusted benchmark medians; do not substitute them into the score.

The TASK floors assume uncompressed weights. The lossless-packing experiment below changes the weight-byte term, so these floors are planning references for the existing paths, not universal limits for every allowed implementation.

## At most three future experiments

All 16 currently authorized checks are used, with the four-check reserve untouched. Do not run another check today. Treat these as candidates for a newly authorized round, not as a promise of top-ten placement.

1. **Specialize the M=128/256 native dense GEMM.** The [pinned SM90 implementation](https://github.com/sgl-project/DeepGEMM/blob/v0.2.1.post1/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) was checked: it forwards M through `get_compiled_dim`, so `compiled_dims="mnk"` can make M static. v017 used `"nk"`. The heuristic already sees runtime M, however; compile-time M is a small control experiment, not a new tiling algorithm. Compare it against v027 on broad medium-M ranges, including verify-16 qkv_gate and verify-32 qkv_gate/pooler. If it is flat, stop this API-only change and require a resource-led tile/pipeline redesign before another dense experiment; do not repeat v010's two-stage BK=256 implementation or the QuACK settings. Proposed target: at least 15% lower medians on verify-32 qkv_gate and pooler (from v027 19.50/30.58 µs) before expanding. Existing generic DeepGEMM, QuACK, and WGMMA measurements do not support a speed claim.
2. **Test lossless BF16 weight packing for bandwidth-bound low-M projections.** A 32-value block can encode sign+mantissa at 8 bits/value, four-bit exponent deltas, and one shared exponent byte: nominally 49 bytes versus 64 bytes, or 12.25 versus 16 bits/value. Use exact raw-BF16 fallback for special values or blocks whose exponent span does not fit; decode within the same GEMM kernel. This stores the original BF16 bit patterns without rounding; it is not FP8/INT8 arithmetic. The generated TASK.md weight rule explicitly allows smaller lossless storage; [public challenge rules](https://openkernelchallenge.dev/rules) remain binding. Include signed zero, subnormal, NaN/Inf, and all fallback paths in bit-level validation. Coalescing matters: padding every 49-byte block back to 64 bytes would erase the nominal storage saving. Proposed feasibility gate: account for fallback blocks, tags, and alignment, then measure at least 16% fewer weight bytes and at least 10% lower medians on a targeted low-M load before expanding. Still requires bit-exact reconstruction, including probe weights, and may lose to decode instructions.
3. **Fuse hi/lo router reads for m>64.** v027’s m=128/256 router profile still contains a matrix kernel, `CatArrayBatchedCopy`, and split-K reduction. A single fused hi/lo kernel that reads the original activation once could remove activation concatenation; removing the reduction also requires owning the complete K reduction within the same launch. Avoid unsafe global-spin reductions between oversubscribed CTAs. This is a different algorithm from v024's measured two-launch split-K. Retain the measured v027 paths for m≤64. Proposed target: at least 25% lower m=128/256 router medians than v027 (11.47/13.28 µs). This is cross-load leverage, but even an ideal router-to-floor change is capped around 1.306×, so pursue only alongside dense gains.

The strongest overall path needs a material dense-kernel improvement on both decode-16/32 and verify-16/32, with the router and selective low-M paths retained. The current measurements do not establish that any one proposed experiment will reach the rank-10 threshold.

## Execution and acceptance

Keep v027 immutable as both the comparison candidate and the current public scored result; v006 is historical. For a future authorized round, use Luna/xhigh workers for disjoint source work: one dense-kernel worker, one lossless-codec worker, and one independent reviewer. The managing agent owns the check queue and merges only independently reviewed bytes; workers do not touch limits, config, ledgers, or submit tools. Freeze and record the source hash before each check, and retain the file unchanged afterward.

Before spending a check, validate API availability, syntax/lint, bounded shared-memory/register use, all scratch writers and readers, and broad shape ranges. Lossless packing needs an offline exact-reconstruction and actual-byte-accounting gate before a GPU candidate is worth checking.

On a newly authorized check, require all nine correctness cases before evaluating the measured 30-shape result. Derive the aggregate from per-load sums and verify gains against noise; profiling identifies kernels and launches but does not prove occupancy or memory-stall causes. Preserve the 20% daily reserve, refresh counters before each run, stop on login/rate-limit errors, and never resend a recorded hash or add a cosmetic change to evade deduplication. A later scored promotion needs fresh authorization with an explicit gate; the completed 27 October approval does not carry forward. Stop all challenge work by 4 November 21:00 America/Los_Angeles.
