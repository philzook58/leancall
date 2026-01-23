from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codylib.leanfun import LeanFun, from_lean, to_lean


def test_to_lean_string_escape():
    assert to_lean('a"b') == '"a\\"b"'


def test_lean_num_string():
    assert to_lean(3) == "3"
    assert to_lean(-3) == "(-3)"
    assert to_lean(1.25) == "1.25"
    assert to_lean(True) == "true"
    assert to_lean("lean") == '"lean"'


def test_from_lean_primitives():
    assert from_lean("2", "Nat") == 2
    assert from_lean("-3", "Int") == -3
    assert from_lean("true", "Bool") is True
    assert from_lean('"hi"', "String") == "hi"
    assert from_lean("1.5", "Float") == 1.5
    assert from_lean("()", "Unit") == ()


def test_from_lean_list_tuple_option():
    assert from_lean("[1, 2, 3]", "List Int") == [1, 2, 3]
    assert from_lean("(1, 2)", "Nat × Nat") == (1, 2)
    assert from_lean("none", "Option Int") is None
    assert from_lean("some 4", "Option Nat") == 4
    assert from_lean("some (some 4)", "Option (Option Nat)") == 4
    assert from_lean("[[], [1, 2], [3]]", "List (List Nat)") == [[], [1, 2], [3]]
    assert from_lean("(1, (2, 3))", "Nat × (Nat × Nat)") == (1, (2, 3))


def test_from_lean_nested_structures():
    assert from_lean("[[1], [2, 3]]", "List (List Nat)") == [[1], [2, 3]]
    assert from_lean("((1, 2), (3, 4))", "(Nat × Nat) × (Nat × Nat)") == (
        (1, 2),
        (3, 4),
    )
    assert (
        from_lean("some [1, 2]", "Option (List Nat)") == [1, 2]
    )
    assert (
        from_lean("some (1, 2)", "Option (Nat × Nat)") == (1, 2)
    )
    assert from_lean("(1, ((2, none), 3))", "Nat × ((Nat × Option Nat) × Nat)") == (
        1,
        ((2, None), 3),
    )


def test_leanfun_from_string_and_call():
    code = "def addFromString (x : Int) := x + 1"
    func = LeanFun.from_string(code)
    assert isinstance(func, LeanFun)
    assert func(1) == 2


def test_leanfun_from_num():
    code = 'def addFromNum (x : Nat) : Nat × (String × (Option (Option Nat))) := (x + 1, ("hello", some (some 42)))'
    func = LeanFun.from_string(code)
    assert isinstance(func, LeanFun)
    assert func(1) == (2, "hello", 42)


def test_leanfun_missing_definition_raises():
    code = "def fooMissing : Int := 1"
    with pytest.raises(ValueError, match="Missing definitions"):
        LeanFun.from_string(code, names=["barMissing"])


def test_leanfun_kwargs_rejected():
    code = "def idKwargs (x : Int) := x"
    func = LeanFun.from_string(code)
    assert isinstance(func, LeanFun)
    with pytest.raises(ValueError, match="Keyword arguments"):
        func(x=1)


def test_to_lean_numpy_scalars():
    np = pytest.importorskip("numpy")
    import codylib.numpy as cody_numpy

    assert cody_numpy is not None
    assert to_lean(np.int64(-3)) == "(-3)"
    assert to_lean(np.float32(1.5)) == "1.5"
    assert to_lean(np.bool_(True)) == "true"


def test_to_lean_numpy_array():
    np = pytest.importorskip("numpy")
    import codylib.numpy as cody_numpy

    assert cody_numpy is not None
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert to_lean(arr) == "[[1, 2], [3, 4]]"


@pytest.fixture(scope="module")
def lean_id_functions():
    fixture = Path(__file__).parent / "fixtures" / "lean_ids.lean"
    names = [
        "idNat",
        "idInt",
        "idBool",
        "idString",
        "idListInt",
        "idPairNat",
        "idFloat",
    ]
    funcs = LeanFun.from_file(str(fixture), names=names)
    assert isinstance(funcs, list)
    assert all(isinstance(func, LeanFun) for func in funcs)
    return dict(zip(names, funcs))


@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_roundtrip_nat(lean_id_functions, x: int):
    assert lean_id_functions["idNat"](x) == x


@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=-10_000, max_value=10_000))
def test_roundtrip_int(lean_id_functions, x: int):
    assert lean_id_functions["idInt"](x) == x


@settings(max_examples=10, deadline=None)
@given(st.booleans())
def test_roundtrip_bool(lean_id_functions, x: bool):
    assert lean_id_functions["idBool"](x) is x


@settings(max_examples=10, deadline=None)
@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
def test_roundtrip_string(lean_id_functions, x: str):
    assert lean_id_functions["idString"](x) == x


@settings(max_examples=10, deadline=None)
@given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=10))
def test_roundtrip_list_int(lean_id_functions, x: list[int]):
    assert lean_id_functions["idListInt"](x) == x


@settings(max_examples=10, deadline=None)
@given(
    st.tuples(
        st.integers(min_value=0, max_value=10_000),
        st.integers(min_value=0, max_value=10_000),
    )
)
def test_roundtrip_pair_nat(lean_id_functions, x: tuple[int, int]):
    assert lean_id_functions["idPairNat"](x) == x


@settings(max_examples=10, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False, width=32))
def test_roundtrip_float(lean_id_functions, x: float):
    assert lean_id_functions["idFloat"](x) == pytest.approx(x, rel=1e-6, abs=1e-6)
