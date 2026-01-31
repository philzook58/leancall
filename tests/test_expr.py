from leancall.expr import (
    App,
    BVar,
    Const,
    ForallE,
    Lam,
    LetE,
    Lit,
    MData,
    Proj,
    Sort,
    parse_lean_expr,
)
from leancall.leanfun import from_string, to_lean


def test_expr_to_lean_roundtrip():
    exprs = [
        BVar(0),
        Sort(2),
        Const("Nat", []),
        App(Const("Nat.succ", []), BVar(0)),
        Lam("x", Const("Nat", []), BVar(0), "default"),
        ForallE("x", Const("Nat", []), Const("Bool", []), "default"),
        LetE("x", Const("Nat", []), Lit(2), App(Const("Nat.succ", []), BVar(0)), True),
        Lit(3),
        Lit("hi"),
        MData({}, BVar(0)),
        Proj("Prod", 0, BVar(0)),
    ]
    for expr in exprs:
        text = to_lean(expr)
        assert parse_lean_expr(text) == expr


def test_expr_roundtrip_through_lean():
    code = """
    import Lean
    open Lean

    def exprToString (e : Expr) : String :=
      reprStr e

    def wrapSucc (e : Expr) : String :=
      reprStr (Expr.app (Expr.const `Nat.succ []) e)

    def mkLet (e : Expr) : String :=
      reprStr (Expr.letE `x (Expr.const `Nat []) e
        (Expr.app (Expr.const `Nat.succ []) (Expr.bvar 0)) true)
    """
    module = from_string(code, names=["exprToString", "wrapSucc", "mkLet"])

    base = Const("Nat.zero", [])
    text = module.exprToString(base)
    assert parse_lean_expr(text) == base

    wrapped_text = module.wrapSucc(base)
    expected = App(Const("Nat.succ", []), base)
    assert parse_lean_expr(wrapped_text) == expected

    let_text = module.mkLet(base)
    expected_let = LetE(
        "x",
        Const("Nat", []),
        base,
        App(Const("Nat.succ", []), BVar(0)),
        True,
    )
    assert parse_lean_expr(let_text) == expected_let
