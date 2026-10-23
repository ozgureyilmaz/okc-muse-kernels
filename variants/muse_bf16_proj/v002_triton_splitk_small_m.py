# okc: problem=muse_bf16_proj
# variant: v002_triton_splitk_small_m
# problem: muse_bf16_proj
# generator: triton
# idea: Split K across Triton CTAs for low-token BF16 projections, then reduce FP32 partials once.
# expected_gain: unmeasured; aims to raise small-m GPU utilization for the large BF16 weights.
# risks: Two launches and scratch traffic may outweigh any bandwidth gain; router remains unchanged.

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


@triton.jit
def _bf16_splitk_partial(
    X,
    W,
    Partial,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX_M: tl.constexpr,
    SX_K: tl.constexpr,
    SW_N: tl.constexpr,
    SW_K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split = K // SPLITS
    split_k_start = pid_split * k_per_split
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, k_per_split, BLOCK_K):
        ks = split_k_start + k_offset + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + rows[:, None] * SX_M + ks[None, :] * SX_K,
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[:, None] * SW_N + ks[None, :] * SW_K,
            mask=(cols[:, None] < N) & (ks[None, :] < K),
            other=0,
        )
        acc = tl.dot(x, tl.trans(w), acc)

    partial_offsets = pid_split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(
        Partial + partial_offsets,
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _bf16_splitk_reduce(
    Partial,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in range(SPLITS):
        acc += tl.load(Partial + split * M * N + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, acc.to(tl.bfloat16), mask=mask)


def prepare(weights: Weights) -> Weights:
    return weights


def kernel_fn(data: Inputs, state: Weights) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    w = state[layer][:n]

    # Keep the checked FP32 router path and the baseline for larger token blocks.
    if layer == "router" or m > 32:
        if w.dtype == torch.float32:
            return x.float() @ w.T
        return x @ w.T

    # The supported K dimensions are multiples of these split sizes.
    splits = 4 if k <= 1024 else 8
    block_m, block_n, block_k = 16, 32, 64
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    _bf16_splitk_partial[(triton.cdiv(m, block_m), triton.cdiv(n, block_n), splits)](
        x,
        w,
        partial,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        splits,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=2,
    )
    _bf16_splitk_reduce[(triton.cdiv(m * n, 256),)](
        partial,
        y,
        m,
        n,
        splits,
        256,
        num_warps=4,
    )
    return y
