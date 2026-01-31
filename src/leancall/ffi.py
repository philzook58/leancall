"""Lean FFI helpers for calling exported Lean functions via cffi.

VERY EXPERIMENTAL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, cast

from lean_interact import Command, LeanServer
from lean_interact.interface import CommandResponse, LeanError

from .leanfun import get_server

try:
    import cffi
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("cffi is required for leancall.ffi") from exc
from lark import Lark, Transformer, v_args


def _to_int(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return int(cast(Any, value))


def _to_float(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return float(cast(Any, value))


def _from_int(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return int(cast(Any, value))


def _from_float(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return float(cast(Any, value))


def _to_bool(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return 1 if bool(value) else 0


def _from_bool(_lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return bool(value)


def _to_nat(lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return lib.leancall_nat_of_uint64(int(cast(Any, value)))


def _from_nat(lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    res = lib.leancall_uint64_of_nat(value)
    lib.leancall_dec(value)
    return int(res)


def _to_int_obj(lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    return lib.leancall_int_of_int64(int(cast(Any, value)))


def _from_int_obj(lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    res = lib.leancall_int64_of_int(value)
    lib.leancall_dec(value)
    return int(res)


def _to_string(lib: Any, ffi: "cffi.FFI", value: object) -> object:
    data = str(value).encode("utf-8")
    buf = ffi.new("char[]", data)
    return lib.leancall_mk_string(buf)


def _from_string(lib: Any, ffi: "cffi.FFI", value: object) -> object:
    c_str = lib.leancall_string_cstr(value)
    py = ffi.string(c_str).decode("utf-8")
    lib.leancall_dec(value)
    return py


def _to_obj(lib: Any, _ffi: "cffi.FFI", value: object) -> object:
    if not isinstance(value, LeanObject):
        raise TypeError("Expected LeanObject for Lean.Expr argument")
    lib.leancall_inc(value.ptr)
    return value.ptr


def _from_obj(lib: Any, ffi: "cffi.FFI", value: object) -> object:
    return _wrap_owned(lib, ffi, value)


def _wrap_owned(lib: Any, ffi: "cffi.FFI", ptr: object) -> "LeanObject":
    obj = LeanObject(lib, ffi, ptr)
    lib.leancall_dec(ptr)
    return obj


def make_nat(lib: Any, ffi: "cffi.FFI", value: int) -> "LeanObject":
    ptr = lib.leancall_nat_of_uint64(int(value))
    return _wrap_owned(lib, ffi, ptr)


def make_int(lib: Any, ffi: "cffi.FFI", value: int) -> "LeanObject":
    ptr = lib.leancall_int_of_int64(int(value))
    return _wrap_owned(lib, ffi, ptr)


def make_string(lib: Any, ffi: "cffi.FFI", value: str) -> "LeanObject":
    data = str(value).encode("utf-8")
    buf = ffi.new("char[]", data)
    ptr = lib.leancall_mk_string(buf)
    return _wrap_owned(lib, ffi, ptr)


def make_tuple2(
    lib: Any, ffi: "cffi.FFI", a: "LeanObject", b: "LeanObject"
) -> "LeanObject":
    assert isinstance(a, LeanObject)
    assert isinstance(b, LeanObject)
    ptr = lib.leancall_mk_pair(a.ptr, b.ptr)
    assert lib.leancall_is_ctor(ptr) == 1
    assert lib.leancall_ctor_num_objs(ptr) == 2
    return _wrap_owned(lib, ffi, ptr)


_C_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CType:
    lean: str
    c: str
    to_c: Callable[[Any, "cffi.FFI", object], object]
    from_c: Callable[[Any, "cffi.FFI", object], object]


_TYPE_MAP = {
    "UInt8": CType("UInt8", "uint8_t", _to_int, _from_int),
    "UInt16": CType("UInt16", "uint16_t", _to_int, _from_int),
    "UInt32": CType("UInt32", "uint32_t", _to_int, _from_int),
    "UInt64": CType("UInt64", "uint64_t", _to_int, _from_int),
    "USize": CType("USize", "size_t", _to_int, _from_int),
    "Float": CType("Float", "double", _to_float, _from_float),
    "Char": CType("Char", "uint32_t", _to_int, _from_int),
    "Bool": CType("Bool", "uint8_t", _to_bool, _from_bool),
    "Nat": CType("Nat", "lean_object *", _to_nat, _from_nat),
    "Int": CType("Int", "lean_object *", _to_int_obj, _from_int_obj),
    "String": CType("String", "lean_object *", _to_string, _from_string),
    "Lean.Expr": CType("Lean.Expr", "lean_object *", _to_obj, _from_obj),
    "Expr": CType("Expr", "lean_object *", _to_obj, _from_obj),
}


class LeanFFIError(RuntimeError):
    pass


@dataclass(frozen=True)
class LeanFFISignature:
    args: list[CType]
    ret: CType

    def cdef(self, c_name: str) -> str:
        arg_list = ", ".join(arg.c for arg in self.args) if self.args else "void"
        return f"{self.ret.c} {c_name}({arg_list});"


class LeanObject:
    _live_ids: dict[int, int] = {}
    _live_count: int = 0

    @classmethod
    def live_count(cls) -> int:
        return cls._live_count

    def __init__(self, lib: Any, ffi: "cffi.FFI", ptr: object):
        assert ptr is not None
        assert ptr != ffi.NULL
        assert ffi.typeof(ptr).cname == "lean_object *"
        self._lib = lib
        self._ffi = ffi
        self._ptr = ptr
        self._closed = False
        self._is_scalar = bool(lib.leancall_is_scalar(ptr))
        if not self._is_scalar:
            obj_id = int(ffi.cast("uintptr_t", ptr))
            self._live_ids[obj_id] = self._live_ids.get(obj_id, 0) + 1
            self.__class__._live_count += 1
        lib.leancall_inc(ptr)

    @classmethod
    def from_string(cls, lib: Any, ffi: "cffi.FFI", value: str) -> "LeanObject":
        data = value.encode("utf-8")
        buf = ffi.new("char[]", data)
        ptr = lib.leancall_mk_string(buf)
        obj = cls(lib, ffi, ptr)
        lib.leancall_dec(ptr)
        return obj

    @property
    def ptr(self) -> object:
        assert not self._closed
        return self._ptr

    def __getitem__(self, key):
        assert not self._closed
        if self._lib.leancall_is_string(self._ptr):
            if not isinstance(key, (int, slice)):
                raise TypeError("String index must be int or slice")
            text = self._ffi.string(self._lib.leancall_string_cstr(self._ptr)).decode(
                "utf-8"
            )
            return text[key]
        if self._lib.leancall_is_ctor(self._ptr):
            if not isinstance(key, int):
                raise TypeError("Constructor index must be int")
            assert key >= 0
            assert key < self._lib.leancall_ctor_num_objs(self._ptr)
            ptr = self._lib.leancall_ctor_get(self._ptr, key)
            return LeanObject(self._lib, self._ffi, ptr)
        raise TypeError("LeanObject is not indexable")

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("LeanObject already closed")
        assert self._ptr != self._ffi.NULL
        self._lib.leancall_dec(self._ptr)
        if not self._is_scalar:
            obj_id = int(self._ffi.cast("uintptr_t", self._ptr))
            assert obj_id in self._live_ids
            self._live_ids[obj_id] -= 1
            if self._live_ids[obj_id] <= 0:
                del self._live_ids[obj_id]
            self.__class__._live_count -= 1
        self._closed = True
        self._ptr = self._ffi.NULL

    def __enter__(self) -> "LeanObject":
        assert not self._closed
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close()

    def __del__(self):
        try:
            if not self._closed:
                self._lib.leancall_dec(self._ptr)
                if not self._is_scalar:
                    obj_id = int(self._ffi.cast("uintptr_t", self._ptr))
                    if obj_id in self._live_ids:
                        self._live_ids[obj_id] -= 1
                        if self._live_ids[obj_id] <= 0:
                            del self._live_ids[obj_id]
                        self.__class__._live_count -= 1
                self._closed = True
        except Exception:
            pass


@dataclass(frozen=True)
class LeanFFIFun:
    name: str
    c_name: str
    signature: LeanFFISignature
    lib: Any
    ffi: "cffi.FFI"

    def __call__(self, *args):
        if len(args) != len(self.signature.args):
            raise TypeError(
                f"{self.name} expects {len(self.signature.args)} args, got {len(args)}"
            )
        converted = [
            ctype.to_c(self.lib, self.ffi, value)
            for ctype, value in zip(self.signature.args, args)
        ]
        func = getattr(self.lib, self.c_name)
        result = func(*converted)
        return self.signature.ret.from_c(self.lib, self.ffi, result)


@dataclass(frozen=True)
class LeanFFIModule:
    functions: dict[str, LeanFFIFun]
    lib_path: Path
    ffi: cffi.FFI
    _tmpdir: tempfile.TemporaryDirectory | None = None

    def __getitem__(self, name: str) -> LeanFFIFun:
        return self.functions[name]

    def __getattr__(self, name: str) -> LeanFFIFun:
        try:
            return self.functions[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def keys(self) -> list[str]:
        return list(self.functions.keys())


def _raise_on_errors(result: CommandResponse | LeanError) -> None:
    if isinstance(result, LeanError):
        raise LeanFFIError(result.message)
    for message in result.messages:
        if message.severity == "error":
            raise LeanFFIError(message.data)


def _extract_def_names(code: str) -> list[str]:
    return re.findall(
        r"^\s*(?:(?:unsafe|partial)\s+)*def\s+([A-Za-z0-9_']+)",
        code,
        re.MULTILINE,
    )


_TYPE_GRAMMAR = r"""
?start: type
?type: func_type
?func_type: app ("->" | "→") func_type   -> func
         | app
?app: atom atom*                         -> app
?atom: NAME                              -> name
     | "(" type ")"

NAME: /[A-Za-z_][A-Za-z0-9_'.]*/

%import common.WS
%ignore WS
"""


@dataclass(frozen=True)
class _TypeName:
    name: str


@dataclass(frozen=True)
class _TypeApp:
    func: Any
    args: tuple[Any, ...]


@dataclass(frozen=True)
class _TypeFun:
    left: Any
    right: Any


@v_args(inline=True)
class _LeanTypeTransformer(Transformer):
    def name(self, token):
        return _TypeName(str(token))

    def app(self, first, *rest):
        if not rest:
            return first
        return _TypeApp(first, tuple(rest))

    def func(self, left, right):
        return _TypeFun(left, right)


_TYPE_PARSER = Lark(
    _TYPE_GRAMMAR, start="start", parser="lalr", transformer=_LeanTypeTransformer()
)


def _get_type_for_name(name: str, server: LeanServer, env: int | None) -> str:
    res = server.run(Command(cmd=f"#check {name}", env=env))
    _raise_on_errors(res)
    if isinstance(res, LeanError):
        raise LeanFFIError(res.message)
    info_messages = [msg for msg in res.messages if msg.severity == "info"]
    if not info_messages:
        raise LeanFFIError(f"No type information for {name}")
    data = info_messages[0].data
    return _type_from_check_output(name, data)


def _type_from_check_output(name: str, data: str) -> str:
    text = data.strip()
    if text.startswith(name):
        text = text[len(name) :].lstrip()
    if text.startswith(":"):
        return text[1:].strip()
    if "→" in text or "->" in text:
        return text.strip()

    args: list[str] = []
    while text.startswith("("):
        match = re.match(r"^\(([^)]*)\)\s*", text)
        if not match:
            break
        inside = match.group(1)
        text = text[match.end() :]
        if ":" not in inside:
            raise LeanFFIError(f"Unexpected binder in #check output: {inside}")
        names_part, type_part = inside.split(":", 1)
        names = [name for name in names_part.split() if name]
        if not names:
            raise LeanFFIError(f"Unexpected binder in #check output: {inside}")
        arg_type = type_part.strip()
        args.extend([arg_type] * len(names))

    text = text.strip()
    if text.startswith(":"):
        ret = text[1:].strip()
    else:
        raise LeanFFIError(f"Unexpected #check output: {data}")
    return " → ".join(args + [ret])


def _lean_version(server: LeanServer, env: int | None) -> str | None:
    res = server.run(Command(cmd="#eval Lean.versionString", env=env))
    _raise_on_errors(res)
    if isinstance(res, LeanError):
        return None
    info_messages = [msg for msg in res.messages if msg.severity == "info"]
    if not info_messages:
        return None
    data = info_messages[0].data.strip()
    if data.startswith('"') and data.endswith('"'):
        data = data[1:-1]
    return data or None


def _parse_function_type(type_str: str) -> LeanFFISignature:
    if "∀" in type_str or "{" in type_str or "[" in type_str:
        raise LeanFFIError(
            "Implicit parameters and typeclass arguments are not supported for FFI"
        )
    try:
        parsed = _TYPE_PARSER.parse(type_str)
    except Exception as exc:  # pragma: no cover - surfaced in tests
        raise LeanFFIError(f"Unable to parse Lean type: {type_str}") from exc

    def _flatten_fun(node: Any) -> list[Any]:
        if isinstance(node, _TypeFun):
            return _flatten_fun(node.left) + _flatten_fun(node.right)
        return [node]

    def _to_ctype(node: Any) -> CType:
        if isinstance(node, _TypeName):
            if node.name not in _TYPE_MAP:
                raise LeanFFIError(
                    f"Unsupported Lean type in FFI: {node.name}. "
                    f"Supported: {', '.join(sorted(_TYPE_MAP))}"
                )
            return _TYPE_MAP[node.name]
        if isinstance(node, _TypeApp):
            raise LeanFFIError(f"Unsupported Lean type in FFI: {node}")
        raise LeanFFIError(f"Unsupported Lean type in FFI: {node}")

    parts = _flatten_fun(parsed)
    if not parts:
        raise LeanFFIError(f"No types found in: {type_str}")
    mapped = [_to_ctype(part) for part in parts]
    return LeanFFISignature(args=mapped[:-1], ret=mapped[-1])


def _c_symbol_name(prefix: str, name: str) -> str:
    raw = f"{prefix}{name}"
    if _C_SYMBOL_RE.match(raw):
        return raw
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _lean_tool_or_raise(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise LeanFFIError(f"{tool} not found on PATH")
    return path


def _elan_run_or_raise() -> str:
    path = shutil.which("elan")
    if not path:
        raise LeanFFIError("elan not found on PATH")
    return path


def _lean_command(tool: str, version: str | None) -> list[str]:
    if version:
        elan_bin = _elan_run_or_raise()
        return [elan_bin, "run", version, tool]
    return [_lean_tool_or_raise(tool)]


def _lean_prefix(version: str | None) -> Path:
    output = subprocess.check_output(
        _lean_command("lean", version) + ["--print-prefix"], text=True
    )
    prefix = Path(output.strip())
    lean_h = prefix / "include" / "lean" / "lean.h"
    if not lean_h.is_file():
        raise LeanFFIError(f"lean.h not found under {lean_h}")
    return prefix


def _lean_libdir(version: str | None) -> str:
    output = subprocess.check_output(
        _lean_command("lean", version) + ["--print-libdir"], text=True
    )
    return output.strip()


_LEAN_SHIM_C = r"""
#include <lean/lean.h>

LEAN_EXPORT void leancall_inc(lean_object * obj) {
    lean_inc(obj);
}

LEAN_EXPORT void leancall_dec(lean_object * obj) {
    lean_dec(obj);
}

LEAN_EXPORT uint8_t leancall_is_scalar(lean_object * obj) {
    return lean_is_scalar(obj);
}

LEAN_EXPORT uint8_t leancall_is_ctor(lean_object * obj) {
    return lean_is_ctor(obj);
}

LEAN_EXPORT uint8_t leancall_is_string(lean_object * obj) {
    return lean_is_string(obj);
}

LEAN_EXPORT uint8_t leancall_is_array(lean_object * obj) {
    return lean_is_array(obj);
}

LEAN_EXPORT uint8_t leancall_ptr_tag(lean_object * obj) {
    return lean_ptr_tag(obj);
}

LEAN_EXPORT unsigned leancall_obj_tag(lean_object * obj) {
    return lean_obj_tag(obj);
}

LEAN_EXPORT uintptr_t leancall_unbox(lean_object * obj) {
    return lean_unbox(obj);
}

LEAN_EXPORT lean_object * leancall_box(uintptr_t v) {
    return lean_box(v);
}

LEAN_EXPORT unsigned leancall_ctor_num_objs(lean_object * obj) {
    return lean_ctor_num_objs(obj);
}

LEAN_EXPORT lean_object * leancall_ctor_get(lean_object * obj, unsigned i) {
    lean_object * v = lean_ctor_get(obj, i);
    lean_inc(v);
    return v;
}

LEAN_EXPORT lean_object * leancall_mk_pair(lean_object * a, lean_object * b) {
    lean_object * r = lean_alloc_ctor(0, 2, 0);
    lean_inc(a);
    lean_inc(b);
    lean_ctor_set(r, 0, a);
    lean_ctor_set(r, 1, b);
    return r;
}

LEAN_EXPORT lean_object * leancall_nat_of_uint64(uint64_t v) {
    return lean_uint64_to_nat(v);
}

LEAN_EXPORT uint64_t leancall_uint64_of_nat(lean_object * obj) {
    return lean_uint64_of_nat(obj);
}

LEAN_EXPORT lean_object * leancall_int_of_int64(int64_t v) {
    return lean_int64_to_int(v);
}

LEAN_EXPORT int64_t leancall_int64_of_int(lean_object * obj) {
    return (int64_t)lean_int64_of_int(obj);
}

LEAN_EXPORT lean_object * leancall_mk_string(char const * s) {
    return lean_mk_string(s);
}

LEAN_EXPORT char const * leancall_string_cstr(lean_object * obj) {
    return lean_string_cstr(obj);
}
"""


def _find_initializer_symbol(c_path: Path) -> str:
    text = c_path.read_text(encoding="utf-8")
    match = re.search(
        r"lean_object\s*\*\s*(initialize_[A-Za-z0-9_]+)\s*\(uint8_t builtin\)\s*\{",
        text,
    )
    if not match:
        raise LeanFFIError("Could not find module initializer in generated C code")
    return match.group(1)


def _build_shared_lib(
    code: str, module_name: str, build_dir: Path, lean_version: str | None
) -> tuple[Path, str]:
    lean_cmd = _lean_command("lean", lean_version)
    leanc_cmd = _lean_command("leanc", lean_version)

    lean_path = build_dir / f"{module_name}.lean"
    c_path = build_dir / f"{module_name}.c"
    o_path = build_dir / f"{module_name}.o"
    shim_c_path = build_dir / f"{module_name}_shim.c"
    shim_o_path = build_dir / f"{module_name}_shim.o"
    so_path = build_dir / f"{module_name}.so"

    lean_path.write_text(code, encoding="utf-8")
    shim_c_path.write_text(_LEAN_SHIM_C, encoding="utf-8")

    subprocess.run(
        [*lean_cmd, "-c", str(c_path), str(lean_path)],
        check=True,
    )
    prefix = _lean_prefix(lean_version)
    include_dir = prefix / "include"
    subprocess.run(
        [
            *leanc_cmd,
            "-I",
            str(include_dir),
            "-c",
            "-DLEAN_EXPORTING",
            "-fPIC",
            "-o",
            str(o_path),
            str(c_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            *leanc_cmd,
            "-I",
            str(include_dir),
            "-c",
            "-DLEAN_EXPORTING",
            "-fPIC",
            "-o",
            str(shim_o_path),
            str(shim_c_path),
        ],
        check=True,
    )

    libdir = _lean_libdir(lean_version)
    subprocess.run(
        [
            *leanc_cmd,
            "-shared",
            "-o",
            str(so_path),
            str(o_path),
            str(shim_o_path),
            f"-L{libdir}",
            "-lleanshared_1",
            "-lleanshared",
            f"-Wl,-rpath,{libdir}",
        ],
        check=True,
    )
    initializer = _find_initializer_symbol(c_path)
    return so_path, initializer


def _generate_wrapper_code(
    code: str,
    exports: dict[str, tuple[str, str]],
    signatures: dict[str, LeanFFISignature],
) -> str:
    lines = [code.rstrip(), ""]
    for lean_name, (wrapper_name, export_name) in exports.items():
        signature = signatures[lean_name]
        arg_parts = [
            f"(x{idx} : {ctype.lean})"
            for idx, ctype in enumerate(signature.args, start=1)
        ]
        arg_names = " ".join(f"x{idx}" for idx in range(1, len(signature.args) + 1))
        lines.append(f"@[export {export_name}]")
        if arg_parts:
            args = " ".join(arg_parts)
            lines.append(
                f"def {wrapper_name} {args} : {signature.ret.lean} := {lean_name} {arg_names}"
            )
        else:
            lines.append(f"def {wrapper_name} : {signature.ret.lean} := {lean_name}")
        lines.append("")
    return "\n".join(lines)


def _build_cdefs(
    module_initializer: str, signatures: dict[str, LeanFFISignature]
) -> str:
    lines = [
        "typedef unsigned char uint8_t;",
        "typedef unsigned short uint16_t;",
        "typedef unsigned int uint32_t;",
        "typedef unsigned long long uint64_t;",
        "typedef unsigned long uintptr_t;",
        "typedef long long int64_t;",
        "typedef unsigned long size_t;",
        "typedef struct lean_object lean_object;",
        "typedef _Bool bool;",
        "void lean_initialize_runtime_module(void);",
        f"lean_object * {module_initializer}(uint8_t builtin);",
        "void leancall_inc(lean_object * obj);",
        "void leancall_dec(lean_object * obj);",
        "uint8_t leancall_is_scalar(lean_object * obj);",
        "uint8_t leancall_is_ctor(lean_object * obj);",
        "uint8_t leancall_is_string(lean_object * obj);",
        "uint8_t leancall_is_array(lean_object * obj);",
        "uint8_t leancall_ptr_tag(lean_object * obj);",
        "unsigned leancall_obj_tag(lean_object * obj);",
        "uintptr_t leancall_unbox(lean_object * obj);",
        "lean_object * leancall_box(uintptr_t v);",
        "unsigned leancall_ctor_num_objs(lean_object * obj);",
        "lean_object * leancall_ctor_get(lean_object * obj, unsigned i);",
        "lean_object * leancall_mk_pair(lean_object * a, lean_object * b);",
        "lean_object * leancall_nat_of_uint64(uint64_t v);",
        "uint64_t leancall_uint64_of_nat(lean_object * obj);",
        "lean_object * leancall_int_of_int64(int64_t v);",
        "int64_t leancall_int64_of_int(lean_object * obj);",
        "lean_object * leancall_mk_string(char const * s);",
        "char const * leancall_string_cstr(lean_object * obj);",
        "",
    ]
    for c_name, signature in signatures.items():
        lines.append(signature.cdef(c_name))
    return "\n".join(lines)


def _initialize_module(lib: Any, module_initializer: str) -> None:
    init_fn = getattr(lib, module_initializer)
    res = init_fn(1)
    lib.leancall_dec(res)


def _ensure_runtime_initialized(lib: Any) -> None:
    lib.lean_initialize_runtime_module()


def _prepare_exports(
    names: list[str], export_prefix: str
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    exports: dict[str, tuple[str, str]] = {}
    c_names: dict[str, str] = {}
    for name in names:
        export_name = _c_symbol_name(export_prefix, name)
        wrapper_name = f"__leancall_ffi_{name}"
        exports[name] = (wrapper_name, export_name)
        c_names[name] = export_name
    return exports, c_names


def from_string(
    code: str,
    names: list[str] | None = None,
    module_name: str = "LeancallFFI",
    export_prefix: str = "ffi_",
    build_dir: Path | None = None,
    server: LeanServer | None = None,
) -> LeanFFIModule:
    server_obj = server if server is not None else get_server()
    res = server_obj.run(Command(cmd=code, env=None))
    _raise_on_errors(res)
    if isinstance(res, LeanError):
        raise LeanFFIError(res.message)

    def_names = _extract_def_names(code)
    if names is None:
        names = def_names
    missing = sorted(set(names) - set(def_names))
    if missing:
        raise LeanFFIError(f"Missing definitions: {', '.join(missing)}")
    if not names:
        raise LeanFFIError("No Lean definitions found")

    type_map = {name: _get_type_for_name(name, server_obj, res.env) for name in names}

    signatures = {name: _parse_function_type(type_map[name]) for name in names}
    exports, c_names = _prepare_exports(names, export_prefix)
    wrapped_code = _generate_wrapper_code(code, exports, signatures)

    tmpdir = None
    if build_dir is None:
        tmp_root = Path.cwd() / ".tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.TemporaryDirectory(prefix="leancall-ffi-", dir=tmp_root)
        build_dir = Path(tmpdir.name)
    build_dir.mkdir(parents=True, exist_ok=True)

    lean_version = _lean_version(server_obj, res.env)
    so_path, initializer = _build_shared_lib(
        wrapped_code, module_name, build_dir, lean_version
    )

    c_signatures = {c_names[name]: signatures[name] for name in names}
    ffi = cffi.FFI()
    cdefs = _build_cdefs(initializer, c_signatures)
    ffi.cdef(cdefs)
    lib = ffi.dlopen(str(so_path))

    _ensure_runtime_initialized(lib)
    _initialize_module(lib, initializer)

    funcs = {
        name: LeanFFIFun(
            name=name,
            c_name=c_names[name],
            signature=signatures[name],
            lib=lib,
            ffi=ffi,
        )
        for name in names
    }
    return LeanFFIModule(functions=funcs, lib_path=so_path, ffi=ffi, _tmpdir=tmpdir)


def from_file(
    filename: str,
    names: list[str] | None = None,
    module_name: str = "LeancallFFI",
    export_prefix: str = "ffi_",
    build_dir: Path | None = None,
    server: LeanServer | None = None,
) -> LeanFFIModule:
    path = Path(filename)
    code = path.read_text(encoding="utf-8")
    return from_string(
        code,
        names=names,
        module_name=module_name,
        export_prefix=export_prefix,
        build_dir=build_dir,
        server=server,
    )
