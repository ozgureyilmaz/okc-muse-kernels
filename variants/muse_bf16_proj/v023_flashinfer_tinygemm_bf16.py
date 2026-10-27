# okc: problem=muse_bf16_proj
# variant: v023_flashinfer_tinygemm_bf16
# problem: muse_bf16_proj
# generator: flashinfer_tinygemm_bf16_small_m
# idea: Preserve v016 router paths and route supported dense 1<=m<=8 BF16 GEMVs to FlashInfer TinyGEMM.
# expected_gain: unmeasured; use the production SM90 latency kernel for tiny dense batches.
# risks: first-use JIT must be preloaded; CUDA-graph capture and per-shape latency are unverified.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from flashinfer.gemm import tinygemm_bf16
from flashinfer.gemm.routergemm import get_tinygemm2_module
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
TINYGEMM_MAX_M = 8
TINYGEMM_K_ALIGNMENT = 64
TINYGEMM_N_ALIGNMENT = 16
TINYGEMM_SUPPORTED_CC = {
    (9, 0),
    (10, 0),
    (10, 3),
    (10, 7),
    (11, 0),
    (12, 0),
    (12, 1),
}


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


def prepare(weights: Weights) -> dict[str, Any]:
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

    tinygemm_sm90 = (
        torch.cuda.get_device_capability(state["qkv_gate"].device) in TINYGEMM_SUPPORTED_CC
    )
    if tinygemm_sm90:
        # Compile/load the kernel module without executing a GEMM or using input data.
        get_tinygemm2_module()
    state["tinygemm_sm90"] = tinygemm_sm90

    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hilo_t"] = torch.cat((hi, lo), dim=1).t().contiguous()
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
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
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)

    tinygemm_supported = (
        state["tinygemm_sm90"]
        and 1 <= m <= TINYGEMM_MAX_M
        and m == x.shape[0]
        and n == w.shape[0]
        and k == x.shape[1] == w.shape[1]
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_contiguous()
        and w.is_contiguous()
        and k % TINYGEMM_K_ALIGNMENT == 0
        and n % TINYGEMM_N_ALIGNMENT == 0
    )
    if tinygemm_supported:
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        tinygemm_bf16(x, w, y, bias=None, use_pdl=False)
        return y

    return F.linear(x, w)
