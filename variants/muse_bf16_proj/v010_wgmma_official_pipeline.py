# okc: problem=muse_bf16_proj
# variant: v010_wgmma_official_pipeline
# problem: muse_bf16_proj
# generator: triton_gluon_wgmma_tma_pipeline
# idea: Preserve v006 and use the official Gluon two-stage shared-memory WGMMA/TMA pipeline for verify projections.
# expected_gain: unmeasured; overlap the next TMA tile load with the current WGMMA on compute-heavy verify shapes.
# risks: Official descriptor/layout API version, shared-memory occupancy, and graph-capture behavior.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)
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


BLOCK_M = 64
BLOCK_K = 256


@gluon.jit
def _bf16_wgmma_project(
    a_desc,
    b_desc,
    y_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    INSTR_N: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # The leading stage dimension is indexed away; the descriptor layouts remain 2D.
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, INSTR_N, 16],
    )
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0
    index = 0
    for k_offset in range(0, K, BLOCK_K):
        a_tile = a_smem.index(index)
        b_tile = b_smem.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k_offset], bar, a_tile)
        tma.async_load(b_desc, [off_n, k_offset], bar, b_tile)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Wait for the previous WGMMA, then launch this one asynchronously.
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a_tile, b_tile.permute((1, 0)), acc, is_async=True)
        index ^= 1

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [4, 1], [1, 0]
    )
    out = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    rows = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))
    cols = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, out_layout))
    offsets = rows[:, None] * N + cols[None, :]
    gl.store(
        y_ptr + offsets,
        out,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


UNPOOLER_DECODE_N = 1024


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

    for layer in ("qkv_gate", "pooler", "unpooler"):
        block_n = 32 if layer == "qkv_gate" else 64
        w = state[layer]
        block_shape = [block_n, BLOCK_K]
        layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.bfloat16)
        state[f"wgmma_{layer}"] = TensorDescriptor.from_tensor(w, block_shape, layout)

    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]

    if layer == "router":
        if m <= 8:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        return torch.addmm(
            acc,
            x,
            state["router_lo_t"],
            beta=1,
            alpha=1,
            out_dtype=torch.float32,
        )

    block_n = 32 if layer == "qkv_gate" else 64 if layer in ("pooler", "unpooler") else 0
    if block_n and 128 <= m <= 256:
        a_block = [BLOCK_M, BLOCK_K]
        a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.bfloat16)
        a_desc = TensorDescriptor.from_tensor(x, a_block, a_layout)
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _bf16_wgmma_project[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, block_n))](
            a_desc,
            state[f"wgmma_{layer}"],
            y,
            m,
            n,
            k,
            BLOCK_M,
            block_n,
            BLOCK_K,
            block_n,
            num_warps=4,
        )
        return y

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)
    return F.linear(x, w)
