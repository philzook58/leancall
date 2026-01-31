from __future__ import annotations

import shutil

import pytest

pytest.importorskip("cffi")

if not shutil.which("lean") or not shutil.which("leanc"):
    pytest.skip("lean/leanc not available", allow_module_level=True)

from leancall.ffi import (
    LeanObject,
    from_string,
    make_int,
    make_nat,
    make_string,
    make_tuple2,
)
from leancall.expr import App, BVar, Const, LetE, Lit, parse_lean_expr


def test_ffi_from_string_uint64_and_float():
    code = """
    def addU64 (a b : UInt64) : UInt64 := a + b
    def scaleFloat (x : Float) : Float := x * 2.5
    """
    module = from_string(
        code,
        names=["addU64", "scaleFloat"],
        module_name="LeancallFFITest",
        export_prefix="ffi_test_",
    )
    assert module.addU64(5, 7) == 12
    assert module.scaleFloat(2.0) == pytest.approx(5.0, rel=1e-6, abs=1e-6)


def test_ffi_nat_int_bool_string():
    code = """
    def addNat (a b : Nat) : Nat := a + b
    def subInt (a b : Int) : Int := a - b
    def flipBool (b : Bool) : Bool := !b
    def greet (name : String) : String := "hi, " ++ name
    """
    module = from_string(
        code,
        names=["addNat", "subInt", "flipBool", "greet"],
        module_name="LeancallFFITest2",
        export_prefix="ffi_test2_",
    )
    assert module.addNat(5, 7) == 12
    assert module.subInt(5, 7) == -2
    assert module.flipBool(True) is False
    assert module.greet("lean") == "hi, lean"


def test_lean_object_refcounting_smoke():
    code = """
def idStr (s : String) : String := s
    """
    module = from_string(
        code,
        names=["idStr"],
        module_name="LeancallFFITest3",
        export_prefix="ffi_test3_",
    )
    lib = module.idStr.lib
    ffi = module.ffi
    obj = LeanObject.from_string(lib, ffi, "hello")
    assert ffi.string(lib.leancall_string_cstr(obj.ptr)).decode("utf-8") == "hello"
    assert lib.leancall_is_scalar(obj.ptr) == 0
    obj.close()
    with pytest.raises(RuntimeError):
        obj.close()


def test_lean_object_debug_helpers():
    code = """
def idNat (x : Nat) : Nat := x
    """
    module = from_string(
        code,
        names=["idNat"],
        module_name="LeancallFFITest4",
        export_prefix="ffi_test4_",
    )
    lib = module.idNat.lib
    ffi = module.ffi
    obj = LeanObject.from_string(lib, ffi, "dbg")
    assert lib.leancall_is_string(obj.ptr) == 1
    assert lib.leancall_ptr_tag(obj.ptr) != 0
    assert lib.leancall_obj_tag(obj.ptr) == lib.leancall_ptr_tag(obj.ptr)
    obj.close()
    boxed = lib.leancall_box(7)
    assert lib.leancall_is_scalar(boxed) == 1
    assert lib.leancall_unbox(boxed) == 7
    assert lib.leancall_obj_tag(boxed) == 7


def test_lean_object_live_count_and_context_manager():
    code = """
def idStr2 (s : String) : String := s
    """
    module = from_string(
        code,
        names=["idStr2"],
        module_name="LeancallFFITest5",
        export_prefix="ffi_test5_",
    )
    lib = module.idStr2.lib
    ffi = module.ffi
    assert LeanObject.live_count() == 0
    with LeanObject.from_string(lib, ffi, "ctx") as obj:
        assert LeanObject.live_count() == 1
        assert ffi.string(lib.leancall_string_cstr(obj.ptr)).decode("utf-8") == "ctx"
    assert LeanObject.live_count() == 0


def test_lean_object_double_close_safety():
    code = """
def idStr3 (s : String) : String := s
    """
    module = from_string(
        code,
        names=["idStr3"],
        module_name="LeancallFFITest6",
        export_prefix="ffi_test6_",
    )
    lib = module.idStr3.lib
    ffi = module.ffi
    obj = LeanObject.from_string(lib, ffi, "double")
    obj.close()
    with pytest.raises(RuntimeError):
        obj.close()


def test_ffi_nontrivial_nat_int_string():
    code = """
def fib : Nat -> Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def mix (n : Nat) (x : Int) (name : String) : String :=
  let v := fib n
  s!"{name}:{v}:{x - 3}"
    """
    module = from_string(
        code,
        names=["fib", "mix"],
        module_name="LeancallFFITest7",
        export_prefix="ffi_test7_",
    )
    assert module.fib(10) == 55
    assert module.mix(6, 10, "lean") == "lean:8:7"


def test_make_helpers_and_tuple():
    code = """
def idNat2 (x : Nat) : Nat := x
    """
    module = from_string(
        code,
        names=["idNat2"],
        module_name="LeancallFFITest8",
        export_prefix="ffi_test8_",
    )
    lib = module.idNat2.lib
    ffi = module.ffi

    nat = make_nat(lib, ffi, 9)
    assert lib.leancall_uint64_of_nat(nat.ptr) == 9

    integer = make_int(lib, ffi, -5)
    assert lib.leancall_int64_of_int(integer.ptr) == -5

    s = make_string(lib, ffi, "pair")
    assert ffi.string(lib.leancall_string_cstr(s.ptr)).decode("utf-8") == "pair"

    pair = make_tuple2(lib, ffi, nat, s)
    assert lib.leancall_ctor_num_objs(pair.ptr) == 2
    fst = lib.leancall_ctor_get(pair.ptr, 0)
    snd = lib.leancall_ctor_get(pair.ptr, 1)
    try:
        assert lib.leancall_uint64_of_nat(fst) == 9
        assert ffi.string(lib.leancall_string_cstr(snd)).decode("utf-8") == "pair"
    finally:
        lib.leancall_dec(fst)
        lib.leancall_dec(snd)

    first = pair[0]
    second = pair[1]
    try:
        assert lib.leancall_uint64_of_nat(first.ptr) == 9
        assert ffi.string(lib.leancall_string_cstr(second.ptr)).decode("utf-8") == "pair"
    finally:
        first.close()
        second.close()


def test_make_helpers_scalar_and_tags():
    code = """
def idInt2 (x : Int) : Int := x
    """
    module = from_string(
        code,
        names=["idInt2"],
        module_name="LeancallFFITest9",
        export_prefix="ffi_test9_",
    )
    lib = module.idInt2.lib
    ffi = module.ffi

    boxed = lib.leancall_box(42)
    assert lib.leancall_is_scalar(boxed) == 1
    assert lib.leancall_unbox(boxed) == 42
    assert lib.leancall_obj_tag(boxed) == 42

    nat = make_nat(lib, ffi, 0)
    assert lib.leancall_is_scalar(nat.ptr) == 1
    assert lib.leancall_unbox(nat.ptr) == 0

    integer = make_int(lib, ffi, -11)
    assert lib.leancall_is_scalar(integer.ptr) in (0, 1)
    assert lib.leancall_int64_of_int(integer.ptr) == -11

    s = make_string(lib, ffi, "tag")
    assert lib.leancall_is_string(s.ptr) == 1
    assert lib.leancall_is_array(s.ptr) == 0
    assert lib.leancall_ptr_tag(s.ptr) == lib.leancall_obj_tag(s.ptr)

    assert s[0] == "t"
    assert s[0:2] == "ta"


def test_lean_object_ptr_after_close():
    code = """
def idStr4 (s : String) : String := s
    """
    module = from_string(
        code,
        names=["idStr4"],
        module_name="LeancallFFITest10",
        export_prefix="ffi_test10_",
    )
    lib = module.idStr4.lib
    ffi = module.ffi
    obj = LeanObject.from_string(lib, ffi, "closed")
    obj.close()
    with pytest.raises(AssertionError):
        _ = obj.ptr


def test_ffi_expr_roundtrip():
    code = """
import Lean
open Lean

def mkZero (_ : Nat) : Lean.Expr := Lean.Expr.const `Nat.zero []

def mkSucc (e : Lean.Expr) : Lean.Expr :=
  Lean.Expr.app (Lean.Expr.const `Nat.succ []) e

def mkLetExpr (_ : Nat) : Lean.Expr :=
  Lean.Expr.letE `x (Lean.Expr.const `Nat []) (Lean.Expr.lit (.natVal 2))
    (Lean.Expr.app (Lean.Expr.const `Nat.succ []) (Lean.Expr.bvar 0)) true

def exprToString (e : Lean.Expr) : String :=
  reprStr e
    """
    module = from_string(
        code,
        names=["mkZero", "mkSucc", "mkLetExpr", "exprToString"],
        module_name="LeancallFFIExprTest",
        export_prefix="ffi_expr_",
    )
    zero = module.mkZero(0)
    try:
        zero_text = module.exprToString(zero)
        assert parse_lean_expr(zero_text) == Const("Nat.zero", [])

        succ = module.mkSucc(zero)
        try:
            succ_text = module.exprToString(succ)
            expected = App(Const("Nat.succ", []), Const("Nat.zero", []))
            assert parse_lean_expr(succ_text) == expected
        finally:
            succ.close()
    finally:
        zero.close()

    let_expr = module.mkLetExpr(0)
    try:
        let_text = module.exprToString(let_expr)
        expected_let = LetE(
            "x",
            Const("Nat", []),
            Lit(2),
            App(Const("Nat.succ", []), BVar(0)),
            True,
        )
        assert parse_lean_expr(let_text) == expected_let
    finally:
        let_expr.close()
