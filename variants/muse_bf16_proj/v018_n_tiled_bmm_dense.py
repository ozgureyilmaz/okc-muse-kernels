# okc: problem=muse_bf16_proj
# variant: v018_n_tiled_bmm_dense
# problem: muse_bf16_proj
# generator: torch_bmm_n_tiled_cublas
# idea: Preserve v006's single-token Hopper GEMV and v011's router hi/lo concat; use strided-batched BF16 bmm to tile output N for short M.
# expected_gain: unmeasured; expose more independent GEMMs for the small-M dense projections.
# risks: relies on zero-stride broadcast input and an interleaved, non-contiguous bmm output view remaining direct cuBLAS operands.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F

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
N_TILED_BMM_MIN_N = 2048


def _n_tile(n: int) -> int:
    """Pick a fixed-width output tile that yields roughly 128 batch GEMMs."""
    if n < 3072:
        return 16
    if n <= 4096:
        return 32
    return 64


def _bmm_n_tiled(
    x: torch.Tensor,
    w_tiles: torch.Tensor,
    m: int,
    n: int,
    tile_n: int,
) -> torch.Tensor:
    groups = n // tile_n
    k = x.shape[1]
    x_batch = x.unsqueeze(0).expand(groups, m, k)

    # W is stored row-major as [N,K]. View it as groups of B output rows, then
    # transpose to [G,K,B] for bmm without copying the weights.
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    # This view has shape [G,M,B] and strides [B,N,1]. PyTorch 2.14.1's CUDA
    # baddbmm path recognizes it as a transposed-result layout: ldc=N and
    # strideC=B, so each batch writes its disjoint channel tile directly into
    # the final contiguous [M,N] allocation.
    out_batch = out.view(m, groups, tile_n).permute(1, 0, 2)
    torch.bmm(x_batch, w_tiles, out=out_batch)
    return out


def prepare(weights: Weights) -> dict[str, Any]:
    # Make each weight contiguous once for the SGLang M=1 GEMV contract;
    # QuACK's N-tiled operands below are views into this lossless storage.
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}

    # Preserve v006's compile-only setup for the eligible M=1 path.
    for layer in ("qkv_gate", "o", "pooler"):
        w = state[layer]
        n, k = w.shape
        if use_hopper_bf16_gemv(1, n, k):
            _jit_hopper_bf16_gemv_module(n, k)

    w = state["unpooler"]
    if use_hopper_bf16_gemv(1, UNPOOLER_DECODE_N, w.shape[1]):
        _jit_hopper_bf16_gemv_module(UNPOOLER_DECODE_N, w.shape[1])

    tiled_weights: dict[str, tuple[int, torch.Tensor]] = {}
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        weight = state[layer]
        n, k = weight.shape
        tile_n = _n_tile(n)
        if weight.is_contiguous() and n % tile_n == 0:
            groups = n // tile_n
            tiled_weights[layer] = (
                tile_n,
                weight.view(groups, tile_n, k).transpose(1, 2),
            )
    state["n_tiled_weights"] = tiled_weights

    # Preserve v011's lossless router hi/lo concatenation.
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
        # Duplicate only this call's activation; do not cache input-derived data.
        x_hilo = torch.cat((x, x), dim=1)
        return torch.mm(x_hilo, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)

    # The low-M sliced unpooler has too few output channels to amortize N tiling.
    # For other short-M shapes, use the prepared view only when it covers all N.
    tiled = state["n_tiled_weights"].get(layer)
    if (
        1 < m <= 64
        and n >= N_TILED_BMM_MIN_N
        and tiled is not None
        and n == state[layer].shape[0]
    ):
        tile_n, w_tiles = tiled
        return _bmm_n_tiled(x, w_tiles, m, n, tile_n)

    return F.linear(x, w)
