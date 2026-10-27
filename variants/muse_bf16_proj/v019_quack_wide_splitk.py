# okc: problem=muse_bf16_proj
# variant: v019_quack_wide_splitk
# problem: muse_bf16_proj
# generator: quack_sm90_batched_parallel_splitk
# idea: Preserve v016's router and M=1 paths; use wide-N QuACK SM90 GEMM with batch-one 3D operands and parallel Split-K for M<=256.
# expected_gain: unmeasured; increase output-tile parallelism on the short-M dense GEMMs.
# risks: split-K workspace/zeroing is timed, and the fixed tile/split choices need correctness and capture validation.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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
QUACK_TILE_N = 128
QUACK_TILE_K = 64
QUACK_TARGET_CTAS = 128
QUACK_MAX_SPLIT_K = 8
QUACK_MIN_K_TILES_PER_SPLIT = 4
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


def _tile_m_for_m(m: int) -> int:
    return 64 if m <= 64 else 128


def _split_k_for_shape(m: int, n: int, k: int, tile_m: int) -> int:
    if k % QUACK_TILE_K != 0:
        return 1

    m_tiles = (m + tile_m - 1) // tile_m
    n_tiles = (n + QUACK_TILE_N - 1) // QUACK_TILE_N
    output_tiles = m_tiles * n_tiles
    k_tiles = k // QUACK_TILE_K

    valid_splits = tuple(
        split
        for split in (1, 2, 4, 8)
        if split <= QUACK_MAX_SPLIT_K
        and k_tiles % split == 0
        and k_tiles // split >= QUACK_MIN_K_TILES_PER_SPLIT
    )
    for split in valid_splits:
        if output_tiles * split >= QUACK_TARGET_CTAS:
            return split
    return valid_splits[-1] if valid_splits else 1


def _compile_quack_dense(
    device_capacity: tuple[int, int],
    tile_m: int,
    split_k: int,
    split_k_mode: SplitKMode,
) -> None:
    """Compile QuACK's generic batch-one BF16 kernel without executing it.

    This mirrors the positional _compile_gemm call made by quack.gemm.gemm.
    Its v0.6.5 jit_cache keys positional args and kwargs separately.
    """
    bf16 = torch2cute_dtype_map[torch.bfloat16]
    _compile_gemm(
        bf16,  # a_dtype
        bf16,  # b_dtype
        bf16,  # d_dtype
        None,  # c_dtype
        "k",  # a_major
        "k",  # b_major
        "n",  # d_major
        None,  # c_major
        (tile_m, QUACK_TILE_N),  # tile_shape_mn; tile_K defaults to 64
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
        split_k,
        split_k_mode,
        True,  # batched: runtime inputs have leading batch size one
        False,  # b_kn
        True,  # sf_batched; no scale factors are used
        None,  # sfd_dtype
        0,  # sfd_norm_const_mode
        None,  # sfd_col_dtype
        None,  # a_mma_dtype
        None,  # b_mma_dtype
        False,  # has_ag
    )


def _run_quack_dense(
    x: torch.Tensor,
    w: torch.Tensor,
    m: int,
    n: int,
    tile_m: int,
    split_k: int,
) -> torch.Tensor:
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    mode = SplitKMode.PARALLEL if split_k > 1 else SplitKMode.SERIAL

    # A/B/D use a leading batch of one so the low-level 3D QuACK path can
    # enable split-K. The per-call split workspace remains inside this timed
    # call and is captured with the GEMM.
    quack_gemm(
        x.unsqueeze(0),
        w.unsqueeze(0),
        out.unsqueeze(0),
        None,  # C
        None,  # tile_count_semaphore; static scheduler
        tile_M=tile_m,
        tile_N=QUACK_TILE_N,
        cluster_M=1,
        cluster_N=1,
        cluster_K=1,
        tile_K=None,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=False,
        alpha=1.0,
        beta=1.0,
        split_k=split_k,
        split_k_mode=mode,
    )
    return out


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}

    # Compile only; do not execute GEMV or derive outputs from input data.
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
        # Runtime uses tile_M=64 for 1..64 and tile_M=128 for 65..256;
        # precompile every split factor the runtime selector can choose.
        for tile_m in (64, 128):
            _compile_quack_dense(
                device_capacity, tile_m, 1, SplitKMode.SERIAL
            )
            for split_k in (2, 4, 8):
                _compile_quack_dense(
                    device_capacity, tile_m, split_k, SplitKMode.PARALLEL
                )

    # Preserve v016's router preparation: exact FP32 weights plus the
    # permitted BF16 hi/lo concatenation for larger batches.
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router"] = router.contiguous()
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

        x_hilo = torch.cat((x, x), dim=1)
        return torch.mm(x_hilo, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)

    low_m_unpooler_slice = (
        layer == "unpooler" and m <= 8 and n <= UNPOOLER_DECODE_N
    )
    if state["quack_sm90"] and 1 < m <= 256 and not low_m_unpooler_slice:
        tile_m = _tile_m_for_m(m)
        split_k = _split_k_for_shape(m, n, k, tile_m)
        return _run_quack_dense(x, w, m, n, tile_m, split_k)

    return F.linear(x, w)
