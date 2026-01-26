from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Callable
from functools import singledispatch
import json
import re
from pathlib import Path

from lark import Lark, Transformer

from lean_interact import LeanREPLConfig, LeanServer, Command, FileCommand
from lean_interact.interface import CommandResponse, LeanError


_DEFAULT_CONFIG = LeanREPLConfig(
    verbose=True, build_repl=True
)  # download and build Lean REPL
_SERVER: LeanServer | None = None
type FromLeanHandler = Callable[[str], object]
_FROM_LEAN_REGISTRY: dict[str, FromLeanHandler] = {}


_LEAN_VALUE_GRAMMAR = r"""
?start: value

?value: record
      | list
      | array
      | tuple
      | unit
      | SOME value        -> some
      | NONE              -> none
      | BOOL              -> bool
      | SIGNED_NUMBER     -> number
      | STRING            -> string
      | NAME              -> name
      | "(" value ")"     -> parens

record: "{" [record_fields] "}"
record_fields: record_field ("," record_field)* ","?
record_field: NAME ":=" value

list: "[" [value_list] "]"
array: "#[" [value_list] "]"
value_list: value ("," value)* ","?

tuple: "(" value "," value ("," value)* ","? ")"
unit: "(" ")"

BOOL.2: "true" | "false"
SOME.2: "some"
NONE.2: "none"
NAME.1: /[A-Za-z_][A-Za-z0-9_']*/

%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
"""


class _LeanValueTransformer(Transformer):
    def string(self, items):
        return json.loads(items[0])

    def number(self, items):
        text = str(items[0])
        if "." in text or "e" in text or "E" in text:
            return float(text)
        return int(text)

    def bool(self, items):
        return str(items[0]) == "true"

    def none(self, _items):
        return None

    def some(self, items):
        value = items[-1]
        return value

    def name(self, items):
        return str(items[0])

    def parens(self, items):
        return items[0]

    def unit(self, _items):
        return ()

    def list(self, items):
        if not items:
            return []
        return list(items[0])

    def array(self, items):
        if not items:
            return []
        return list(items[0])

    def value_list(self, items):
        return list(items)

    def tuple(self, items):
        return tuple(items)

    def record(self, items):
        if not items:
            return {}
        return dict(items[0])

    def record_fields(self, items):
        return list(items)

    def record_field(self, items):
        return (str(items[0]), items[1])


_LEAN_VALUE_PARSER = Lark(
    _LEAN_VALUE_GRAMMAR, start="start", parser="lalr", maybe_placeholders=False
)
_LEAN_VALUE_TRANSFORMER = _LeanValueTransformer()


def parse_lean_value(text: str) -> object:
    try:
        tree = _LEAN_VALUE_PARSER.parse(text)
    except Exception as exc:  # pragma: no cover - error path is exercised in tests
        raise ValueError(f"Failed to parse Lean value: {text}") from exc
    return _LEAN_VALUE_TRANSFORMER.transform(tree)


def register_from_lean(type_name: str) -> Callable[[FromLeanHandler], FromLeanHandler]:
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


# Or should this use anonymous record syntax?
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


def from_lean(x: str, typ: str | None = None) -> object:
    x = x.strip()
    if typ in _FROM_LEAN_REGISTRY:
        return _FROM_LEAN_REGISTRY[typ](x)
    try:
        return parse_lean_value(x)
    except ValueError:
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            raise ValueError("Could not parse lean response", x)


def _extract_def_names(code: str) -> list[str]:
    return re.findall(
        r"^\s*(?:(?:unsafe|partial)\s+)*def\s+([A-Za-z0-9_']+)",
        code,
        re.MULTILINE,
    )


def _raise_on_errors(result: CommandResponse | LeanError) -> None:
    if isinstance(result, LeanError):
        raise ValueError(result.message)
    for message in result.messages:
        if message.severity == "error":
            raise ValueError(message.data)


@dataclass(frozen=True)
class LeanFun:
    name: str
    env: int | None
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
        return from_lean(info_messages[0].data)


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
    res = server_obj.run(Command(cmd=code, env=env))
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
            server=server,
            code=code,
        )
        for name in names
    ]
