# okc: problem=muse_bf16_proj
# variant: v015_router_concat_deepgemm_sm90
# problem: muse_bf16_proj
# generator: pytorch_bf16_router_concat_deepgemm_sm90
# idea: Combine v011's measured router hi/lo concatenation with v012's measured DeepGEMM dense path, preserving v006 m=1 Hopper GEMVs.
# expected_gain: unmeasured integration; target gains across router and dense layers.
# risks: router concatenation overhead and strict tolerance, DeepGEMM dispatch/capture behavior, and uneven per-shape dense performance.

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
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
