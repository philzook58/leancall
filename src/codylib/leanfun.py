from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
import json
import re
from pathlib import Path

from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import CommandResponse, LeanError


_DEFAULT_CONFIG = LeanREPLConfig(
    verbose=True, build_repl=True
)  # download and build Lean REPL
_SERVER: LeanServer | None = None


def get_server(config: LeanREPLConfig | None = None) -> LeanServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = LeanServer(config or _DEFAULT_CONFIG)
    return _SERVER


@singledispatch
def to_lean(x: object) -> str:
    raise Exception(f"Cannot convert {x} to Lean")


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

    @classmethod
    def from_string(
        cls,
        code: str,
        names: list[str] | None = None,
        env: int | None = None,
        server: LeanServer | None = None,
    ) -> "LeanFun | list[LeanFun]":
        return cls._from_source(code, names=names, env=env, server=server)

    """
    Refactor __init__ to be less smart, and have the smarts come from here
    """

    @classmethod
    def from_file(
        cls,
        filename: str,
        names: list[str] | None = None,
        env: int | None = None,
        server: LeanServer | None = None,
    ) -> "LeanFun | list[LeanFun]":
        code = Path(filename).read_text(encoding="utf-8")
        return cls._from_source(code, names=names, env=env, server=server)

    @classmethod
    def _from_source(
        cls,
        code: str,
        names: list[str] | None = None,
        env: int | None = None,
        server: LeanServer | None = None,
    ) -> "LeanFun | list[LeanFun]":
        server_obj = server if server is not None else get_server()
        res = server_obj.run(Command(cmd=code, env=env))
        _raise_on_errors(res)
        if isinstance(res, LeanError):
            raise ValueError(res.message)
        env = res.env
        found = _extract_def_names(code)
        if names is None:
            names = found
        missing = sorted(set(names) - set(found))
        if missing:
            raise ValueError(f"Missing definitions: {', '.join(missing)}")
        if not names:
            raise ValueError("No Lean definitions found")
        funcs = [
            cls(
                name=name,
                env=env,
                res_type=_infer_return_type(server_obj, name, env),
                server=server_obj,
                code=code,
            )
            for name in names
        ]
        return funcs[0] if len(funcs) == 1 else funcs

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError("Keyword arguments are not supported")
        res = self.server.run(
            Command(
                cmd=f"#eval {self.name} " + " ".join(map(to_lean, args)), env=self.env
            )
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
