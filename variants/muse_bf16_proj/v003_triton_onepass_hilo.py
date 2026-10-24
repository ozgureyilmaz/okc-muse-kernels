# okc: problem=muse_bf16_proj
# variant: v003_triton_onepass_hilo
# problem: muse_bf16_proj
# generator: triton
# idea: Use one tiled Triton GEMM for low-token BF16 projections and SGLang's lossless hi/lo router path.
# expected_gain: unmeasured; removes v002's partial-buffer reduction launch and fixes FP32 router regressions.
# risks: A single output-tiled kernel may underfill the GPU for small n; router uses two GEMMs above 8 tokens.

from typing import TypedDict

import torch
import torch.nn.functional as F
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
def _bf16_gemm(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX_M: tl.constexpr,
    SX_K: tl.constexpr,
    SW_N: tl.constexpr,
    SW_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, K, BLOCK_K):
        k_idx = k_offset + ks
        x = tl.load(
            X + rows[:, None] * SX_M + k_idx[None, :] * SX_K,
            mask=(rows[:, None] < M) & (k_idx[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[:, None] * SW_N + k_idx[None, :] * SW_K,
            mask=(cols[:, None] < N) & (k_idx[None, :] < K),
            other=0,
        )
        acc = tl.dot(x, tl.trans(w), acc)

    out_offsets = rows[:, None] * N + cols[None, :]
    tl.store(
        Y + out_offsets,
        acc.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def prepare(weights: Weights) -> dict[str, torch.Tensor]:
    state = {name: w.contiguous() for name, w in weights.items()}
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
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    w = state[layer][:n]
    if m > 32:
        return x @ w.T

    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    block_m, block_n, block_k = 16, 16, 128
    _bf16_gemm[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
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
        block_n,
        block_k,
        num_warps=4,
        num_stages=2,
    )
    return y
