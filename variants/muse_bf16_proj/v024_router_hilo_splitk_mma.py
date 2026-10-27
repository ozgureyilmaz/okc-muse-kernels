# okc: problem=muse_bf16_proj
# variant: v024_router_hilo_splitk_mma
# problem: muse_bf16_proj
# generator: triton_bf16_hilo_splitk_mma
# idea: Keep exact FP32 GEMV for router m<=2; use split-K BF16 hi+lo tensor-core dots for larger m.
# expected_gain: unmeasured; expose more independent router CTAs and avoid the two-GEMM path.
# risks: split-K scratch/reduction overhead, FP32 accumulation order, and CUDA-graph capture behavior.

from typing import Any, TypedDict

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
ROUTER_MMA_BLOCK_M_SMALL = 16
ROUTER_MMA_BLOCK_M_LARGE = 32
ROUTER_MMA_BLOCK_N = 32
ROUTER_MMA_BLOCK_K = 128
ROUTER_REDUCE_BLOCK = 128
ROUTER_TARGET_CTAS = 256
ROUTER_MAX_SPLIT_K = 32


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
def _router_hilo_splitk_mma(
    X,
    W_HI,
    W_LO,
    PARTIALS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    SX_M: tl.constexpr,
    SX_K: tl.constexpr,
    SHI_N: tl.constexpr,
    SHI_K: tl.constexpr,
    SLO_N: tl.constexpr,
    SLO_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    k_begin = pid_split * K_PER_SPLIT
    k_end = k_begin + K_PER_SPLIT

    acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, K_PER_SPLIT, BLOCK_K):
        k_idx = k_begin + k_offset + ks
        a = tl.load(
            X + rows[:, None] * SX_M + k_idx[None, :] * SX_K,
            mask=(rows[:, None] < M) & (k_idx[None, :] < K) & (k_idx[None, :] < k_end),
            other=0,
        )
        b_hi_nk = tl.load(
            W_HI + cols[:, None] * SHI_N + k_idx[None, :] * SHI_K,
            mask=(cols[:, None] < N) & (k_idx[None, :] < K) & (k_idx[None, :] < k_end),
            other=0,
        )
        b_lo_nk = tl.load(
            W_LO + cols[:, None] * SLO_N + k_idx[None, :] * SLO_K,
            mask=(cols[:, None] < N) & (k_idx[None, :] < K) & (k_idx[None, :] < k_end),
            other=0,
        )
        b_hi = tl.trans(b_hi_nk)
        b_lo = tl.trans(b_lo_nk)
        acc_hi = tl.dot(a, b_hi, acc_hi)
        acc_lo = tl.dot(a, b_lo, acc_lo)

    partial = acc_hi + acc_lo
    offsets = pid_split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(
        PARTIALS + offsets,
        partial,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _reduce_router_splitk(
    PARTIALS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    split_offsets = tl.arange(0, SPLIT_K)
    out_offsets = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    values = tl.load(
        PARTIALS + split_offsets[:, None] * (M * N) + out_offsets[None, :],
        mask=out_offsets[None, :] < M * N,
        other=0.0,
    )
    result = tl.sum(values, axis=0)
    tl.store(Y + out_offsets, result, mask=out_offsets < M * N)


def _router_split_k(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> int:
    output_ctas = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    split_k = 1
    while output_ctas * split_k < ROUTER_TARGET_CTAS and split_k < ROUTER_MAX_SPLIT_K:
        candidate = split_k * 2
        if k % (candidate * block_k) != 0 or k // candidate < 2 * block_k:
            break
        split_k = candidate
    return split_k


def _router_hilo_concat_fallback(
    x: torch.Tensor,
    state: dict[str, Any],
    n: int,
    k: int,
) -> torch.Tensor:
    hi = state["router_hi"][:n, :k]
    lo = state["router_lo"][:n, :k]
    if n == state["router"].shape[0] and k == state["router"].shape[1]:
        weights_t = state["router_hilo_t"]
    else:
        weights_t = torch.cat((hi, lo), dim=1).t().contiguous()
    x_hilo = torch.cat((x, x), dim=1)
    return torch.mm(x_hilo, weights_t, out_dtype=torch.float32)


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
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat((hi, lo), dim=1).t().contiguous()
    return state


def kernel_fn(data: Inputs, state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]

    if layer == "router":
        if m <= 2:
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

        w_hi = state["router_hi"]
        w_lo = state["router_lo"]
        block_m = ROUTER_MMA_BLOCK_M_SMALL if m <= 16 else ROUTER_MMA_BLOCK_M_LARGE
        block_n = ROUTER_MMA_BLOCK_N
        block_k = ROUTER_MMA_BLOCK_K
        if (
            n > w_hi.shape[0]
            or k > w_hi.shape[1]
            or n % block_n != 0
            or k < 2 * block_k
            or k % (2 * block_k) != 0
        ):
            return _router_hilo_concat_fallback(x, state, n, k)

        split_k = _router_split_k(m, n, k, block_m, block_n, block_k)
        k_per_split = k // split_k
        partials = torch.empty(
            (split_k, m, n), device=x.device, dtype=torch.float32
        )
        y = torch.empty((m, n), device=x.device, dtype=torch.float32)
        _router_hilo_splitk_mma[
            (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
        ](
            x,
            w_hi,
            w_lo,
            partials,
            m,
            n,
            k,
            k_per_split,
            x.stride(0),
            x.stride(1),
            w_hi.stride(0),
            w_hi.stride(1),
            w_lo.stride(0),
            w_lo.stride(1),
            block_m,
            block_n,
            block_k,
            num_warps=4,
            num_stages=2,
        )
        _reduce_router_splitk[(triton.cdiv(m * n, ROUTER_REDUCE_BLOCK),)](
            partials,
            y,
            m,
            n,
            split_k,
            ROUTER_REDUCE_BLOCK,
            num_warps=4,
        )
        return y

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)
    return F.linear(x, w)
