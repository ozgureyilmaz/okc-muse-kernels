"""Static checks for a submission variant. Does not import or run the kernel.

The timed path is `kernel_fn` and the functions it calls. `prepare` is untimed
and may take tens of seconds, but it still may not touch the network, the
filesystem, or the harness.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

BANNED_MODULES = frozenset({
    "socket",
    "requests",
    "urllib",
    "http",
    "httpx",
    "aiohttp",
    "subprocess",
    "multiprocessing",
    "webbrowser",
    "ftplib",
    "smtplib",
    "ctypes",
    "pickle",
    "importlib",
    "inspect",
})
BANNED_IMPORT_NAMES = frozenset({"reference", "checks", "baseline"})
HOST_SYNC_ATTRS = frozenset({"item", "cpu", "numpy", "tolist", "synchronize"})
CACHE_DECORATORS = frozenset({"lru_cache", "cache"})
FILE_FUNCS = frozenset({
    "open",
    "write_text",
    "write_bytes",
    "unlink",
    "remove",
    "system",
    "popen",
})

# Local packages the grader image is expected to have. A miss is a warning.
GRADER_IMPORTS = frozenset({
    "torch",
    "triton",
    "cutlass",
    "cuda",
    "flashinfer",
    "deep_gemm",
    "sglang",
    "tilelang",
    "quack",
    "numpy",
})


class LintReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def lint_source(source: str, filename: str = "<variant>") -> LintReport:
    report = LintReport()
    if not source.strip():
        report.error("file is empty")
        return report
    first = source.splitlines()[0].strip()
    if not first.startswith("# okc: problem="):
        report.error("first line must be `# okc: problem=<name>` so okc can route the file")
    if "5151" in source:
        report.error("source contains 5151, the reference weight seed; do not special-case the judge's inputs")
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        report.error(f"syntax error: {exc}")
        return report
    _check_imports(tree, report)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    kernel = functions.get("kernel_fn")
    if kernel is None:
        report.error("kernel_fn is not defined")
        return report
    if len(kernel.args.args) != 2 or kernel.args.vararg or kernel.args.kwarg:
        report.error("kernel_fn must take exactly (data, state). The challenge page wins over a one-argument sketch")
    if "prepare" not in functions:
        report.warn("prepare(weights) is missing; this problem calls it once, untimed")
    reachable = _reachable(kernel, functions)
    module_caches = _module_caches(tree)
    for name in reachable:
        _check_timed(functions[name], module_caches, report)
    _check_anywhere(tree, report)
    return report


def lint_path(path: Path) -> LintReport:
    return lint_source(path.read_text(), filename=str(path))


def _check_imports(tree: ast.AST, report: LintReport) -> None:
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
            else:
                names.extend(alias.name for alias in node.names)
        for name in names:
            if name in BANNED_MODULES or name in BANNED_IMPORT_NAMES:
                report.error(f"banned import {name}")
                continue
            if name.startswith("_"):
                continue
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                spec = None
            if spec is None and name not in GRADER_IMPORTS:
                report.warn(f"import {name} does not resolve on this machine")
            elif spec is None:
                report.warn(f"import {name} is not installed here; the grader image may still have it")


def _reachable(kernel: ast.FunctionDef, functions: dict[str, ast.FunctionDef]) -> set[str]:
    calls: dict[str, set[str]] = {}
    for name, node in functions.items():
        calls[name] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in functions:
                    calls[name].add(child.func.id)
    seen = {"kernel_fn"}
    stack = ["kernel_fn"]
    while stack:
        current = stack.pop()
        for callee in calls.get(current, ()):
            if callee not in seen:
                seen.add(callee)
                stack.append(callee)
    return seen


def _module_caches(tree: ast.AST) -> set[str]:
    caches = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                caches.add(target.id)
    return caches


def _check_timed(fn: ast.FunctionDef, caches: set[str], report: LintReport) -> None:
    label = fn.name
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            _check_call(node, label, report)
        if isinstance(node, ast.Attribute) and node.attr in HOST_SYNC_ATTRS:
            report.error(f"{label}: .{node.attr}() syncs or copies off the device and breaks CUDA graph capture")
        if isinstance(node, ast.Name) and node.id in caches and isinstance(node.ctx, ast.Load):
            report.error(f"{label}: reads module cache `{node.id}` (no caches derived from inputs)")
    for decorator in fn.decorator_list:
        name = _decorator_name(decorator)
        if name in CACHE_DECORATORS:
            report.error(f"{label}: @{name} caches across calls")


def _check_call(node: ast.Call, label: str, report: LintReport) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in {"eval", "exec", "__import__"}:
            report.error(f"{label}: {func.id}()")
        if func.id == "print":
            report.error(f"{label}: print() inside the timed path can sync a tensor and breaks the graph")
        if func.id == "id":
            report.error(f"{label}: id() must not key a cache on an input")
        if func.id in FILE_FUNCS:
            report.error(f"{label}: {func.id}() is file or process I/O")
    if isinstance(func, ast.Attribute) and func.attr == "data_ptr":
        report.error(f"{label}: data_ptr() must not key a cache on an input")
    if isinstance(func, ast.Attribute) and func.attr in FILE_FUNCS:
        report.error(f"{label}: .{func.attr}() is file or process I/O")


def _check_anywhere(tree: ast.AST, report: LintReport) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            report.error("open() is filesystem I/O; the harness rejects it")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (
                (node.value.id == "os" and node.attr == "environ")
                or (node.value.id == "sys" and node.attr in {"_getframe", "modules"})
            )
        ):
            report.error(f"{node.value.id}.{node.attr} can read the harness, the environment, or secrets")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"system", "popen", "write_text", "write_bytes"}:
                report.error(f".{node.func.attr}() is process or file I/O")


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def format_report(path: Path, report: LintReport) -> str:
    lines = [f"{path}: {'ok' if report.ok else 'failed'}"]
    for message in report.errors:
        lines.append(f"  error: {message}")
    for message in report.warnings:
        lines.append(f"  warning: {message}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python tools/lint_kernel.py <variant.py>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    report = lint_path(path)
    print(format_report(path, report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
