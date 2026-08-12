# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Fail-closed syntax and capability gate for Workbench programs."""

import ast
import builtins

MAX_SOURCE_BYTES = 128 * 1024
MAX_AST_NODES = 20_000
_ALLOWED_IMPORTS = {
    "accelerate",
    "collections",
    "dataclasses",
    "functools",
    "hashlib",
    "itertools",
    "math",
    "random",
    "safetensors",
    "statistics",
    "time",
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "typing",
}
_BANNED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_BANNED_ATTRIBUTES = {
    "builtins",
    "classes",
    "connect",
    "ctypes",
    "distributed",
    "dump",
    "dumps",
    "from_file",
    "fromfile",
    "from_pretrained",
    "hub",
    "importlib",
    "jit",
    "load",
    "multiprocessing",
    "ops",
    "package",
    "load_inline",
    "load_library",
    "os",
    "pathlib",
    "popen",
    "remove",
    "removedirs",
    "rename",
    "renames",
    "replace",
    "request",
    "rmdir",
    "rmtree",
    "save",
    "serialization",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "system",
    "tofile",
    "unlink",
    "urlopen",
}
_BANNED_CHAINS = {
    "torch.classes",
    "torch.distributed",
    "torch.hub",
    "torch.jit",
    "torch.multiprocessing",
    "torch.ops",
    "torch.package",
    "torch.serialization",
    "torch.utils.cpp_extension",
}


class UnsafeScript(ValueError):
    pass


def _chain(node: ast.Attribute) -> str:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _matches_module(name: str, roots: set[str]) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
    level = node.level if isinstance(node, ast.ImportFrom) else 0
    if level or any(not _matches_module(name, _ALLOWED_IMPORTS) for name in modules):
        raise UnsafeScript("import is not available in Machine scripts")
    if any(_matches_module(name, _BANNED_CHAINS) for name in modules):
        raise UnsafeScript("unsafe module import is not allowed")
    if isinstance(node, ast.ImportFrom) and any(
        alias.name.startswith("_") or alias.name in _BANNED_ATTRIBUTES for alias in node.names
    ):
        raise UnsafeScript("private or unsafe imported attributes are not allowed")


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        raise UnsafeScript(f"{type(node).__name__.lower()} declarations are not allowed")
    if isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("__")):
        raise UnsafeScript(f"name {node.id!r} is not allowed")
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__"):
        raise UnsafeScript("dunder attribute access is not allowed")
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        _validate_import(node)
    if not isinstance(node, ast.Attribute):
        return
    chain = _chain(node)
    if node.attr.startswith("_") or node.attr in _BANNED_ATTRIBUTES:
        raise UnsafeScript(f"attribute {node.attr!r} is not allowed")
    if _matches_module(chain, _BANNED_CHAINS):
        raise UnsafeScript(f"API {chain!r} is not allowed")


def _parse(code: str) -> list[ast.AST]:
    try:
        tree = ast.parse(code, filename="<workbench:machine>", mode="exec")
    except SyntaxError as exc:
        raise UnsafeScript(f"invalid Python: {exc.msg} at line {exc.lineno}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise UnsafeScript("script is too complex")
    return nodes


def validate_script(code: str) -> ast.Module:
    """Parse and validate Machine code against the explicit capability set."""
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise UnsafeScript("script exceeds the 128 KiB safety limit")
    nodes = _parse(code)
    for node in nodes:
        _validate_node(node)
    return nodes[0]


def _builtins() -> dict:
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or not any(name == root or name.startswith(root + ".") for root in _ALLOWED_IMPORTS):
            raise ImportError(f"module {name!r} is not available in Machine scripts")
        return builtins.__import__(name, globals, locals, fromlist, level)

    names = {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "Exception",
        "IndexError",
        "KeyError",
        "MemoryError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "__build_class__",
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "hasattr",
        "hash",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
    values = {name: getattr(builtins, name) for name in names}
    values["__import__"] = guarded_import
    return values


def execute_script(code: str, namespace: dict) -> None:
    """Execute validated code with a reduced builtin and import namespace."""
    tree = validate_script(code)
    namespace["__builtins__"] = _builtins()
    namespace.setdefault("__name__", "workbench_machine")
    exec(compile(tree, "<workbench:machine>", "exec"), namespace)  # nosec B102
