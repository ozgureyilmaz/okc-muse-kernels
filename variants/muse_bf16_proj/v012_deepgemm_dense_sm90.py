# okc: problem=muse_bf16_proj
# variant: v012_deepgemm_dense_sm90
# problem: muse_bf16_proj
# generator: sglang_hopper_gemv_deepgemm_sm90
# idea: Keep v006 m=1 GEMVs and router path; use the pinned DeepGEMM BF16 NT native SM90 path for larger dense projections.
# expected_gain: unmeasured; target cross-load dense gains on decode-16/32 and verify-16/32.
# risks: DeepGEMM package/API availability, SM90 dispatch, first-call JIT during graph capture, and no public compile-only API.

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

    w = state[layer][:n]
    if use_hopper_bf16_gemv(m, n, k):
        return hopper_bf16_gemv(x, w)
    if m > 8 and state["deep_gemm_available"]:
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(x, w, y, compiled_dims="nk")
        return y
    return F.linear(x, w)
