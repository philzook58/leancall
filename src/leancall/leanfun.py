from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Callable
from functools import singledispatch
import json
import re
from pathlib import Path

from lean_interact import LeanREPLConfig, LeanServer, Command, FileCommand
from lean_interact.interface import CommandResponse, LeanError


_DEFAULT_CONFIG = LeanREPLConfig(
    verbose=True, build_repl=True
)  # download and build Lean REPL
_SERVER: LeanServer | None = None
FromLeanHandler = Callable[[str, str], object]
_FROM_LEAN_REGISTRY: dict[str, FromLeanHandler] = {}


def register_from_lean(type_name: str):
    def decorator(func):
        _FROM_LEAN_REGISTRY[type_name] = func
        return func

    return decorator


def get_server(config: LeanREPLConfig | None = None) -> LeanServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = LeanServer(config or _DEFAULT_CONFIG)
    return _SERVER


@singledispatch
def to_lean(x: object) -> str:
    if is_dataclass(x) and not isinstance(x, type):
        items = []
        for field in fields(x):
            items.append(f"{field.name} := {to_lean(getattr(x, field.name))}")
        return "{" + ", ".join(items) + "}"
    raise Exception(
        f"Cannot convert {x} of type {type(x)} to Lean using `leancall.to_lean`. Maybe you need to import an optional module?"
    )


@to_lean.register
def _(x: int) -> str:
    value = str(x)
    return f"({value})" if value.startswith("-") else value


@to_lean.register
def _(x: bool) -> str:
    return "true" if x else "false"


@to_lean.register
def _(x: str) -> str:
    return json.dumps(x)


@to_lean.register
def _(x: float) -> str:
    value = str(x)
    return f"({value})" if value.startswith("-") else value


@to_lean.register(list)
def _(x: list) -> str:
    return "[" + ", ".join(map(to_lean, x)) + "]"


@to_lean.register(tuple)
def _(x: tuple) -> str:
    return "(" + ", ".join(map(to_lean, x)) + ")"


@to_lean.register(type(None))
def _(x: None) -> str:
    return "none"


@to_lean.register(dict)
def _(x: dict) -> str:
    return _to_lean_json(x)


def _float_to_json_number(value: float) -> tuple[int, int]:
    dec = Decimal(str(value)).normalize()
    sign, digits, exp = dec.as_tuple()
    if not isinstance(exp, int):
        raise ValueError("Non-finite float for Json encoding")
    if not digits:
        return 0, 0
    mantissa = int("".join(str(d) for d in digits))
    if sign:
        mantissa = -mantissa
    if exp >= 0:
        mantissa *= 10**exp
        exponent = 0
    else:
        exponent = -exp
    return mantissa, exponent


def _to_lean_json(value: object) -> str:
    if value is None:
        return "Lean.Json.null"
    if isinstance(value, bool):
        return f"Lean.Json.bool {to_lean(value)}"
    if isinstance(value, int):
        return f"({value} : Lean.Json)"
    if isinstance(value, float):
        mantissa, exponent = _float_to_json_number(value)
        return f"Lean.Json.num (JsonNumber.mk {mantissa} {exponent})"
    if isinstance(value, str):
        return f"Lean.Json.str {to_lean(value)}"
    if isinstance(value, list):
        elems = ", ".join(_to_lean_json(elem) for elem in value)
        return f"Lean.Json.arr #[{elems}]"
    if isinstance(value, dict):
        items = ", ".join(
            f"({to_lean(str(key))}, {_to_lean_json(val)})" for key, val in value.items()
        )
        return f"Lean.Json.mkObj [{items}]"
    raise Exception(f"Cannot convert {value} to Lean.Json")


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _split_product_types(typ: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in typ:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "×" and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _parse_string(text: str) -> str:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip('"')


def _strip_outer_parens(text: str) -> str:
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return text
    return text[1:-1].strip()


def from_lean(x: str, typ: str) -> object:
    x = x.strip()
    typ = _strip_outer_parens(typ)
    if typ in _FROM_LEAN_REGISTRY:
        return _FROM_LEAN_REGISTRY[typ](x, typ)
    if typ in {"Nat", "Int"}:
        value = int(x)
        if typ == "Nat" and value < 0:
            raise ValueError(f"Expected Nat, got {x}")
        return value
    if typ == "Bool":
        if x not in {"true", "false"}:
            raise ValueError(f"Expected Bool, got {x}")
        return x == "true"
    if typ == "String":
        return _parse_string(x)
    if typ in {"Json", "Lean.Json"}:
        return json.loads(x)
    if typ == "Float":
        return float(x)
    if typ == "Unit":
        if x != "()":
            raise ValueError(f"Expected Unit, got {x}")
        return ()
    if typ.startswith("Option "):
        option_x = _strip_outer_parens(x)
        if option_x == "none":
            return None
        if option_x.startswith("some "):
            inner_typ = typ[len("Option ") :].strip()
            return from_lean(option_x[len("some ") :], inner_typ)
        raise ValueError(f"Expected Option, got {option_x}")
    if typ.startswith("List "):
        inner_typ = typ[len("List ") :].strip()
        if not (x.startswith("[") and x.endswith("]")):
            raise ValueError(f"Expected List, got {x}")
        content = x[1:-1].strip()
        if not content:
            return []
        return [from_lean(part, inner_typ) for part in _split_top_level(content)]
    if "×" in typ:
        typ_parts = _split_product_types(typ)
        if not (x.startswith("(") and x.endswith(")")):
            raise ValueError(f"Expected tuple, got {x}")
        content = x[1:-1].strip()
        values = _split_top_level(content)
        if len(values) != len(typ_parts):
            raise ValueError(f"Tuple arity mismatch: {x} : {typ}")
        return tuple(from_lean(val, t) for val, t in zip(values, typ_parts))
    raise ValueError(f"Unsupported Lean type {typ}")


def _extract_def_names(code: str) -> list[str]:
    return re.findall(r"^\s*def\s+([A-Za-z0-9_']+)", code, re.MULTILINE)


def _raise_on_errors(result: CommandResponse | LeanError) -> None:
    if isinstance(result, LeanError):
        raise ValueError(result.message)
    for message in result.messages:
        if message.severity == "error":
            raise ValueError(message.data)


def _infer_return_type(server: LeanServer, name: str, env: int | None) -> str:
    res = server.run(Command(cmd=f"#check {name}", env=env))
    _raise_on_errors(res)
    if isinstance(res, LeanError) or not res.messages:
        raise ValueError(f"No type information for {name}")
    full_type = res.messages[0].data.split(":")[-1].strip()
    if "->" in full_type:
        return full_type.split("->")[-1].strip()
    if "→" in full_type:
        return full_type.split("→")[-1].strip()
    return full_type


@dataclass(frozen=True)
class LeanFun:
    name: str
    env: int | None
    res_type: str
    server: LeanServer
    code: str | None = None

    def __call__(self, *args, **kwargs):
        arg_parts = [to_lean(arg) for arg in args]
        for key, value in kwargs.items():
            arg_parts.append(f"({key} := {to_lean(value)})")
        res = self.server.run(
            Command(cmd=f"#eval {self.name} " + " ".join(arg_parts), env=self.env)
        )
        _raise_on_errors(res)
        if isinstance(res, LeanError):
            raise ValueError(res.message)
        info_messages = [
            message for message in res.messages if message.severity == "info"
        ]
        if not info_messages:
            raise ValueError("No evaluation result returned")
        return from_lean(info_messages[0].data, self.res_type)


@dataclass(frozen=True)
class LeanModule:
    functions: dict[str, LeanFun]

    def __getitem__(self, name: str) -> LeanFun:
        return self.functions[name]

    def __getattr__(self, name: str) -> LeanFun:
        try:
            return self.functions[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def keys(self) -> list[str]:
        return list(self.functions.keys())


def from_string(
    code: str,
    names: list[str] | None = None,
    env: int | None = None,
    server: LeanServer | None = None,
) -> LeanModule:
    server_obj = server if server is not None else get_server()
    full_code = "import Lean\n" + code
    res = server_obj.run(Command(cmd=full_code, env=env))
    _raise_on_errors(res)
    if isinstance(res, LeanError):
        raise ValueError(res.message)
    funcs = _build_funcs(code, server=server_obj, names=names, env=res.env)
    return LeanModule({func.name: func for func in funcs})


def from_file(
    filename: str,
    names: list[str] | None = None,
    env: int | None = None,
    server: LeanServer | None = None,
) -> LeanModule:
    server_obj = server if server is not None else get_server()
    res = server_obj.run(FileCommand(path=filename, env=env))
    _raise_on_errors(res)
    if isinstance(res, LeanError):
        raise ValueError(res.message)
    code = Path(filename).read_text(encoding="utf-8")
    funcs = _build_funcs(code, server=server_obj, names=names, env=res.env)
    return LeanModule({func.name: func for func in funcs})


def _build_funcs(
    code: str,
    server: LeanServer,
    names: list[str] | None = None,
    env: int | None = None,
) -> list["LeanFun"]:
    found = _extract_def_names(code)
    if names is None:
        names = found
    missing = sorted(set(names) - set(found))
    if missing:
        raise ValueError(f"Missing definitions: {', '.join(missing)}")
    if not names:
        raise ValueError("No Lean definitions found")
    return [
        LeanFun(
            name=name,
            env=env,
            res_type=_infer_return_type(server, name, env),
            server=server,
            code=code,
        )
        for name in names
    ]
