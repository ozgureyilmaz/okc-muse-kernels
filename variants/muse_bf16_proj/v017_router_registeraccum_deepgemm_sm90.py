# okc: problem=muse_bf16_proj
# variant: v017_router_registeraccum_deepgemm_sm90
# problem: muse_bf16_proj
# generator: triton_router_registeraccum_deepgemm_sm90
# idea: Combine v016's measured small-M exact-FP32 router GEMV with v012's measured DeepGEMM dense path, preserving v006 m=1 Hopper GEMVs and v011 hi/lo router for m>8.
# expected_gain: unmeasured integration; retain measured dense gains while lowering router m<=8 latency.
# risks: integration-level capture/tolerance and uneven per-shape dense performance.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
try:
    import deep_gemm
except ImportError:
    deep_gemm = None
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

    deep_gemm_available = (
        deep_gemm is not None
        and callable(getattr(deep_gemm, "bf16_gemm_nt", None))
        and callable(getattr(deep_gemm, "use_deterministic_algorithms", None))
    )
    if deep_gemm_available:
        # This selects the library's native SM90 implementation; it does not execute a GEMM.
        deep_gemm.use_deterministic_algorithms(True)
    state["deep_gemm_available"] = deep_gemm_available
    state["deep_gemm_path"] = "deep_gemm_native_sm90" if deep_gemm_available else "torch_f_linear_fallback"
    print(f"Dense BF16 backend path: {state['deep_gemm_path']}")

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

        # Duplicate only this call's activation; input-derived values are not cached.
        x_hilo = torch.cat((x, x), dim=1)
        return torch.mm(x_hilo, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)
    if m > 8 and state["deep_gemm_available"]:
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(x, w, y, compiled_dims="nk")
        return y
    return F.linear(x, w)
