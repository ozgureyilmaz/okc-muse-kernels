# okc: problem=muse_bf16_proj
# variant: v028_lowm_tensorcore_splitk
# problem: muse_bf16_proj
# generator: triton_bf16_tiled_splitk_lowm
# idea: Preserve v016 router and M=1 Hopper paths; use wide-N BF16 Tensor Core partial GEMMs plus a deterministic FP32 reduction for eligible low-M dense projections.
# expected_gain: unmeasured; raise low-M output CTA parallelism with 128-wide N tiles and grid-derived K partitions.
# risks: extra kernel launch and scratch traffic may erase Tensor Core gains; split reduction order and graph capture need correctness checks.

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
    _jit_hopper_bf16_gemv_module,
    hopper_bf16_gemv,
    use_hopper_bf16_gemv,
)


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Inputs(TypedDict):
    x: torch.Tensor
    config: dict[str, int | str]


UNPOOLER_DECODE_N = 1024
ROUTER_GEMV_BLOCK_K = 1024
ROUTER_GEMV_BLOCK_M = 4

LOWM_TILE_N = 128
LOWM_TILE_K = 128
LOWM_TARGET_CTAS = 256
LOWM_MAX_SPLIT_K = 16
LOWM_MIN_K_TILES_PER_SPLIT = 2
LOWM_REDUCE_BLOCK = 256


@triton.jit
def _router_fp32_small_gemv(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SX_M: tl.constexpr,
    SX_K: tl.constexpr,
    SW_N: tl.constexpr,
    SW_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for k_offset in range(0, K, BLOCK_K):
        k_idx = k_offset + ks
        x_tile = tl.load(
            X + rows[:, None] * SX_M + k_idx[None, :] * SX_K,
            mask=(rows[:, None] < M) & (k_idx[None, :] < K),
            other=0,
        ).to(tl.float32)
        w_tile = tl.load(
            W + pid_n * SW_N + k_idx * SW_K,
            mask=k_idx < K,
            other=0.0,
        )
        acc = tl.fma(x_tile, w_tile[None, :], acc)

    out = tl.sum(acc, axis=1)
    tl.store(Y + rows * N + pid_n, out, mask=rows < M)


@triton.jit
def _bf16_splitk_partial_gemm(
    X,
    W,
    PARTIALS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX_M: tl.constexpr,
    SX_K: tl.constexpr,
    SW_N: tl.constexpr,
    SW_K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split = tl.program_id(2)

    rows = pid_m * TILE_M + tl.arange(0, TILE_M)
    cols = pid_n * TILE_N + tl.arange(0, TILE_N)
    k_lanes = tl.arange(0, TILE_K)
    k_start = pid_split * K_PER_SPLIT
    k_end = k_start + K_PER_SPLIT

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k_block in range(0, K_PER_SPLIT, TILE_K):
        k_idx = k_start + k_block + k_lanes
        a_tile = tl.load(
            X + rows[:, None] * SX_M + k_idx[None, :] * SX_K,
            mask=(rows[:, None] < M) & (k_idx[None, :] < K) & (k_idx[None, :] < k_end),
            other=0,
        )
        b_tile = tl.load(
            W + cols[:, None] * SW_N + k_idx[None, :] * SW_K,
            mask=(cols[:, None] < N) & (k_idx[None, :] < K) & (k_idx[None, :] < k_end),
            other=0,
        )
        acc = tl.dot(a_tile, tl.trans(b_tile), acc)

    partial_offsets = (pid_split * M * N) + rows[:, None] * N + cols[None, :]
    tl.store(
        PARTIALS + partial_offsets,
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _reduce_bf16_splitk_partials(
    PARTIALS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split_idx in range(SPLIT_K):
        partial = tl.load(
            PARTIALS + split_idx * M * N + offsets,
            mask=mask,
            other=0.0,
        )
        acc = acc + partial
    tl.store(Y + offsets, acc.to(tl.bfloat16), mask=mask)


def _lowm_split_k_for_shape(m: int, n: int, k: int, tile_m: int) -> int | None:
    if k % LOWM_TILE_K != 0:
        return None

    output_tiles = triton.cdiv(m, tile_m) * triton.cdiv(n, LOWM_TILE_N)
    for split_k in (1, 2, 4, 8, 16):
        if split_k > LOWM_MAX_SPLIT_K:
            continue
        if k % split_k != 0:
            continue
        k_per_split = k // split_k
        if k_per_split % LOWM_TILE_K != 0:
            continue
        if k_per_split < LOWM_MIN_K_TILES_PER_SPLIT * LOWM_TILE_K:
            continue
        if output_tiles * split_k >= LOWM_TARGET_CTAS:
            return split_k
    return None


def _lowm_bf16_tensorcore_splitk(
    x: torch.Tensor, w: torch.Tensor, m: int, n: int, k: int, split_k: int
) -> torch.Tensor:
    tile_m = 16 if m <= 16 else 32
    k_per_split = k // split_k
    partials = torch.empty(
        (split_k, m, n), device=x.device, dtype=torch.float32
    )
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    _bf16_splitk_partial_gemm[
        (triton.cdiv(m, tile_m), triton.cdiv(n, LOWM_TILE_N), split_k)
    ](
        x,
        w,
        partials,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        tile_m,
        LOWM_TILE_N,
        LOWM_TILE_K,
        k_per_split,
        num_warps=4,
        num_stages=2,
    )
    _reduce_bf16_splitk_partials[
        (triton.cdiv(m * n, LOWM_REDUCE_BLOCK),)
    ](
        partials,
        out,
        m,
        n,
        split_k,
        LOWM_REDUCE_BLOCK,
        num_warps=4,
    )
    return out

def prepare(weights: Weights) -> dict[str, torch.Tensor]:
    state = {name: w.contiguous() for name, w in weights.items()}

    # Compile only; do not execute GEMV or derive any output from input data here.
    for layer in ("qkv_gate", "o", "pooler"):
        w = state[layer]
        n, k = w.shape
        if use_hopper_bf16_gemv(1, n, k):
            _jit_hopper_bf16_gemv_module(n, k)

    w = state["unpooler"]
    if use_hopper_bf16_gemv(1, UNPOOLER_DECODE_N, w.shape[1]):
        _jit_hopper_bf16_gemv_module(UNPOOLER_DECODE_N, w.shape[1])

    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hilo_t"] = torch.cat((hi, lo), dim=1).t().contiguous()
    return state


def kernel_fn(data: Inputs, state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]

    if layer == "router":
        if m <= 8:
            w = state["router"]
            block_m = 1 if m <= 2 else ROUTER_GEMV_BLOCK_M
            y = torch.empty((m, n), device=x.device, dtype=torch.float32)
            _router_fp32_small_gemv[(triton.cdiv(m, block_m), n)](
                x,
                w,
                y,
                m,
                n,
                k,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                block_m,
                ROUTER_GEMV_BLOCK_K,
                num_warps=4,
                num_stages=2,
            )
            return y

        # Construct a fresh activation pair for this call; do not cache input-derived data.
        x_hilo = torch.cat((x, x), dim=1)
        return torch.mm(x_hilo, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer][:n]
    if m == 1 and use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)

    if 1 < m <= 64 and n >= 2048:
        split_k = _lowm_split_k_for_shape(m, n, k, 16 if m <= 16 else 32)
        if split_k is not None:
            return _lowm_bf16_tensorcore_splitk(x, w, m, n, k, split_k)

    return F.linear(x, w)
