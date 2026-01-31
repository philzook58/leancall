"""
Datatypes to mirror Lean.Expr.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import cast

from lark import Lark, Transformer

from .leanfun import to_lean, register_from_lean


class Expr: ...


@dataclass(frozen=True)
class BVar(Expr):
    deBruijnIndex: int

    @staticmethod
    def of_dict(d: dict) -> "BVar":
        return BVar(deBruijnIndex=int(d["deBruijnIndex"]))


@dataclass(frozen=True)
class FVar(Expr):
    fvarId: int | str


@dataclass(frozen=True)
class MVar(Expr):
    mvarId: int | str


@dataclass(frozen=True)
class Sort(Expr):
    u: int


@dataclass(frozen=True)
class Const(Expr):
    declName: str
    us: list[int]


@dataclass(frozen=True)
class App(Expr):
    fn: Expr
    arg: Expr


@dataclass(frozen=True)
class Lam(Expr):
    binderName: str
    binderType: Expr
    body: Expr
    binderInfo: str


@dataclass(frozen=True)
class ForallE(Expr):
    binderName: str
    binderType: Expr
    body: Expr
    binderInfo: str


@dataclass(frozen=True)
class LetE(Expr):
    declName: str
    type: Expr
    value: Expr
    body: Expr
    nondep: bool


@dataclass(frozen=True)
class Lit(Expr):
    value: object


@dataclass(frozen=True)
class MData(Expr):
    data: dict
    expr: Expr


@dataclass(frozen=True)
class Proj(Expr):
    typeName: str
    idx: int
    struct: Expr


def _name_to_lean(name: str) -> str:
    if name.startswith("`"):
        return name
    return f"`{name}"


def _binder_info_to_lean(info: str) -> str:
    info = info.strip(".")
    return f"Lean.BinderInfo.{info}"


def _level_to_lean(u: int) -> str:
    if u < 0:
        raise ValueError("Level must be non-negative")
    value = "Lean.Level.zero"
    for _ in range(u):
        value = f"Lean.Level.succ {value}"
    return value


def _level_list_to_lean(us: list[int]) -> str:
    parts = ", ".join(_level_to_lean(u) for u in us)
    return f"[{parts}]"


def _lit_to_lean(value: object) -> str:
    if isinstance(value, int):
        return f".natVal {value}"
    if isinstance(value, str):
        return f".strVal {json.dumps(value)}"
    raise ValueError(f"Unsupported literal: {value!r}")


def _name_from_id(prefix: str, value: int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{prefix}{value}"


@to_lean.register
def _(x: BVar) -> str:
    return f"(Lean.Expr.bvar {x.deBruijnIndex})"


@to_lean.register
def _(x: FVar) -> str:
    name = _name_from_id("fvar", x.fvarId)
    return f"(Lean.Expr.fvar {_name_to_lean(name)})"


@to_lean.register
def _(x: MVar) -> str:
    name = _name_from_id("mvar", x.mvarId)
    return f"(Lean.Expr.mvar {_name_to_lean(name)})"


@to_lean.register
def _(x: Sort) -> str:
    return f"(Lean.Expr.sort {_level_to_lean(x.u)})"


@to_lean.register
def _(x: Const) -> str:
    return f"(Lean.Expr.const {_name_to_lean(x.declName)} {_level_list_to_lean(x.us)})"


@to_lean.register
def _(x: App) -> str:
    return f"(Lean.Expr.app {to_lean(x.fn)} {to_lean(x.arg)})"


@to_lean.register
def _(x: Lam) -> str:
    return (
        "(Lean.Expr.lam "
        f"{_name_to_lean(x.binderName)} {to_lean(x.binderType)} {to_lean(x.body)} "
        f"{_binder_info_to_lean(x.binderInfo)})"
    )


@to_lean.register
def _(x: ForallE) -> str:
    return (
        "(Lean.Expr.forallE "
        f"{_name_to_lean(x.binderName)} {to_lean(x.binderType)} {to_lean(x.body)} "
        f"{_binder_info_to_lean(x.binderInfo)})"
    )


@to_lean.register
def _(x: LetE) -> str:
    return (
        "(Lean.Expr.letE "
        f"{_name_to_lean(x.declName)} {to_lean(x.type)} {to_lean(x.value)} "
        f"{to_lean(x.body)} {'true' if x.nondep else 'false'})"
    )


@to_lean.register
def _(x: Lit) -> str:
    return f"(Lean.Expr.lit ({_lit_to_lean(x.value)}))"


@to_lean.register
def _(x: MData) -> str:
    if x.data:
        raise ValueError("Only empty MData is supported for now")
    return f"(Lean.Expr.mdata Lean.MData.empty {to_lean(x.expr)})"


@to_lean.register
def _(x: Proj) -> str:
    return f"(Lean.Expr.proj {_name_to_lean(x.typeName)} {x.idx} {to_lean(x.struct)})"


_LEAN_EXPR_GRAMMAR = r"""
?start: expr

?expr: parens
     | bvar
     | fvar
     | mvar
     | sort
     | const
     | app
     | lam
     | forall_e
     | let_e
     | lit
     | mdata
     | proj

parens: "(" expr ")"    -> parens

bvar: ".bvar" NUMBER
fvar: ".fvar" name
mvar: ".mvar" name
sort: ".sort" level
const: ".const" name level_list
app: ".app" expr expr
lam: ".lam" name expr expr binder_info
forall_e: ".forallE" name expr expr binder_info
let_e: ".letE" name expr expr expr BOOL
lit: ".lit" literal
   | ".lit" "(" literal ")" -> lit_parens
mdata: ".mdata" mdata_val expr
proj: ".proj" name NUMBER expr

literal: ".natVal" NUMBER -> nat_lit
       | ".strVal" STRING -> str_lit

level_list: "[" [level_list_items] "]"
level_list_items: level ("," level)* ","?

level: ".zero" -> level_zero
     | ".succ" level -> level_succ

binder_info: ".default" -> bi_default
           | ".implicit" -> bi_implicit
           | ".strictImplicit" -> bi_strict
           | ".instImplicit" -> bi_inst

mdata_val: "empty" -> mdata_empty
         | "{" "}" -> mdata_empty

name: BACKTICK -> name_bt
    | NAME      -> name_plain

BOOL.2: "true" | "false"
BACKTICK: /`[A-Za-z0-9_'.]+/
NAME: /[A-Za-z_][A-Za-z0-9_'.]*/

%import common.SIGNED_NUMBER -> NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
"""


class _LeanExprTransformer(Transformer):
    def parens(self, items):
        return items[0]

    def name_bt(self, items):
        return str(items[0])[1:]

    def name_plain(self, items):
        return str(items[0])

    def bvar(self, items):
        return BVar(int(items[0]))

    def fvar(self, items):
        name = str(items[0])
        match = re.fullmatch(r"fvar(\d+)", name)
        return FVar(int(match.group(1)) if match else name)

    def mvar(self, items):
        name = str(items[0])
        match = re.fullmatch(r"mvar(\d+)", name)
        return MVar(int(match.group(1)) if match else name)

    def sort(self, items):
        return Sort(int(items[0]))

    def const(self, items):
        return Const(str(items[0]), list(items[1]))

    def app(self, items):
        return App(items[0], items[1])

    def lam(self, items):
        return Lam(str(items[0]), items[1], items[2], str(items[3]))

    def forall_e(self, items):
        return ForallE(str(items[0]), items[1], items[2], str(items[3]))

    def let_e(self, items):
        return LetE(str(items[0]), items[1], items[2], items[3], items[4])

    def lit(self, items):
        return Lit(items[0])

    def lit_parens(self, items):
        return Lit(items[0])

    def mdata(self, items):
        return MData(items[0], items[1])

    def proj(self, items):
        return Proj(str(items[0]), int(items[1]), items[2])

    def nat_lit(self, items):
        return int(items[0])

    def str_lit(self, items):
        return json.loads(items[0])

    def level_list(self, items):
        return items[0] if items else []

    def level_list_items(self, items):
        return list(items)

    def level_zero(self, _items):
        return 0

    def level_succ(self, items):
        return int(items[0]) + 1

    def bi_default(self, _items):
        return "default"

    def bi_implicit(self, _items):
        return "implicit"

    def bi_strict(self, _items):
        return "strictImplicit"

    def bi_inst(self, _items):
        return "instImplicit"

    def mdata_empty(self, _items):
        return {}

    def BOOL(self, token):
        return str(token) == "true"


_LEAN_EXPR_PARSER = Lark(
    _LEAN_EXPR_GRAMMAR,
    start="start",
    parser="lalr",
    maybe_placeholders=False,
    transformer=_LeanExprTransformer(),
)


def _normalize_expr_text(text: str) -> str:
    replacements = {
        "Lean.Expr.": ".",
        "Expr.": ".",
        "Lean.Level.": ".",
        "Lean.BinderInfo.": ".",
        "Lean.MData.": "",
        "Lean.Literal.": ".",
        "Literal.": ".",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def parse_lean_expr(text: str) -> Expr:
    normalized = _normalize_expr_text(text.strip())
    return cast(Expr, _LEAN_EXPR_PARSER.parse(normalized))


@register_from_lean("Lean.Expr")
def _from_lean_expr(text: str) -> Expr:
    return parse_lean_expr(text)
