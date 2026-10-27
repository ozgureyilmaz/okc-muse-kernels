# okc: problem=muse_bf16_proj
# variant: v013_router_fp32_gemv
# problem: muse_bf16_proj
# generator: triton_fp32_router_gemv
# idea: Use a one-launch exact-FP32 GEMV for router inputs with at most eight rows; retain v006 paths elsewhere.
# expected_gain: unmeasured; avoid the FP32 input-conversion and GEMM sequence on small router batches.
# risks: per-channel CTAs may not saturate memory bandwidth; verify FP32 reduction error and graph capture.

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
ROUTER_GEMV_BLOCK_M = 4
ROUTER_GEMV_BLOCK_K = 1024


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
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

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
        acc += tl.sum(x_tile * w_tile[None, :], axis=1)

    tl.store(Y + rows * N + pid_n, acc, mask=rows < M)


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
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
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
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        return torch.addmm(
            acc,
            x,
            state["router_lo_t"],
            beta=1,
            alpha=1,
            out_dtype=torch.float32,
        )

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)
    return F.linear(x, w)
