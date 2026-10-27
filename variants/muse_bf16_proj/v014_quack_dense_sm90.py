# okc: problem=muse_bf16_proj
# variant: v014_quack_dense_sm90
# problem: muse_bf16_proj
# generator: quack_cute_sm90_fixed_dense
# idea: Preserve v006's single-token Hopper GEMV and v011's router hi/lo concatenation; use fixed-config QuACK SM90 GEMM for other dense projections.
# expected_gain: unmeasured; target decode and verify dense projections while keeping the proven low-M specialization.
# risks: private compile-only helper must stay cache-key-identical to the public GEMM call; fixed WGMMA tile geometry may underfill low-M shapes.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F

from quack.cute_dsl_utils import get_device_capacity, torch2cute_dtype_map
from quack.gemm import _compile_gemm, gemm as quack_gemm
from quack.gemm_config import SplitKMode
from quack.rounding import RoundingMode
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

# These are broad M buckets rather than score-shape special cases. Smaller N
# tiles provide enough independent output CTAs on H200 when M is short.
QUACK_TILE_CONFIGS = ((64, 16), (64, 32), (128, 64))


def _compile_quack_dense(tile_m: int, tile_n: int, device_capacity: tuple[int, int]) -> None:
    """JIT-compile QuACK's generic dense kernel without launching a GEMM.

    QuACK v0.6.5's private helper builds symbolic fake tensors and calls
    cute.compile. Keep these static arguments in sync with _run_quack_dense so
    the runtime plan hits the same compilation cache entry.
    """
    bf16 = torch2cute_dtype_map[torch.bfloat16]
    # QuACK v0.6.5 jit_cache hashes the raw positional tuple, so preserve the
    # exact argument order used by quack.gemm.gemm's _compile_gemm call.
    _compile_gemm(
        bf16,  # a_dtype
        bf16,  # b_dtype
        bf16,  # d_dtype
        None,  # c_dtype
        "k",  # a_major
        "k",  # b_major
        "n",  # d_major
        None,  # c_major
        (tile_m, tile_n),  # tile_shape_mn
        (1, 1, 1),  # cluster_shape_mnk
        False,  # pingpong
        True,  # persistent
        False,  # is_dynamic_persistent
        None,  # rowvec_dtype
        None,  # colvec_dtype
        0,  # colvec_ndim
        0,  # alpha_mode: scalar_mode(1.0)
        0,  # beta_mode: scalar_mode(1.0)
        False,  # add_to_output
        (),  # concat_layout
        False,  # varlen_m
        False,  # varlen_k
        False,  # gather_A
        False,  # use_tma_gather
        False,  # has_batch_idx_permute
        device_capacity,
        RoundingMode.RN,
        0,  # sr_seed_mode
        None,  # num_warps
        None,  # sf_dtype
        None,  # sf_vec_size
        1,  # split_k
        SplitKMode.SERIAL,
        False,  # batched
        False,  # b_kn
        True,  # sf_batched; public call uses True when unscaled
        None,  # sfd_dtype
        0,  # sfd_norm_const_mode
        None,  # sfd_col_dtype
        None,  # a_mma_dtype
        None,  # b_mma_dtype
        False,  # has_ag
    )


def _tile_config(m: int) -> tuple[int, int]:
    if m <= 64:
        return 64, 16
    if m <= 256:
        return 64, 32
    return 128, 64


def _run_quack_dense(
    x: torch.Tensor,
    w: torch.Tensor,
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
) -> torch.Tensor:
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    # The low-level QuACK API consumes B as row-major [N, K] and mutates D.
    # Static persistence and split_k=1 avoid per-call workspace/sync paths.
    quack_gemm(
        x,
        w,
        out,
        None,  # C
        None,  # tile_count_semaphore; dynamic scheduling is disabled
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        cluster_K=1,
        tile_K=None,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=False,
        alpha=1.0,
        beta=1.0,
        split_k=1,
        split_k_mode=SplitKMode.SERIAL,
    )
    return out


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}

    # Keep the existing SGLang GEMV compilation-only setup for eligible M=1
    # projections. No GEMV is executed during preparation.
    for layer in ("qkv_gate", "o", "pooler"):
        w = state[layer]
        n, k = w.shape
        if use_hopper_bf16_gemv(1, n, k):
            _jit_hopper_bf16_gemv_module(n, k)

    w = state["unpooler"]
    if use_hopper_bf16_gemv(1, UNPOOLER_DECODE_N, w.shape[1]):
        _jit_hopper_bf16_gemv_module(UNPOOLER_DECODE_N, w.shape[1])

    device_capacity = get_device_capacity(state["qkv_gate"].device)
    state["quack_sm90"] = device_capacity[0] == 9
    if state["quack_sm90"]:
        # All projection GEMMs use BF16 A/B/D, K-major A/B, N-major D,
        # unbatched dense operands, and fixed tile configs. QuACK's compiler
        # creates symbolic fake tensors and does not execute the operation.
        for tile_m, tile_n in QUACK_TILE_CONFIGS:
            _compile_quack_dense(tile_m, tile_n, device_capacity)

    # Preserve v011's lossless router hi/lo K-concatenation path.
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
        # Duplicate only the current activation; never cache input-derived data.
        x_hilo = torch.cat((x, x), dim=1)
        return torch.mm(x_hilo, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)

    # The sliced decode unpooler has only 1024 output rows and is too small to
    # fill enough H200 CTAs with QuACK's minimum 64-row M tile.
    if state["quack_sm90"] and not (m <= 8 and n < 2048):
        tile_m, tile_n = _tile_config(m)
        return _run_quack_dense(x, w, m, n, tile_m, tile_n)

    return F.linear(x, w)
