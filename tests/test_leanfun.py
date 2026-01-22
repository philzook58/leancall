from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pathlib import Path

from codylib.leanfun import LeanFun, from_lean, to_lean


@dataclass
class FakeMessage:
    severity: str
    data: str


@dataclass
class FakeResult:
    messages: list[FakeMessage]
    env: object


class FakeServer:
    def __init__(self, responses: dict[str, FakeResult]):
        self.responses = responses
        self.commands: list[str] = []

    def run(self, command):
        self.commands.append(command.cmd)
        return self.responses[command.cmd]


def test_to_lean_string_escape():
    assert to_lean('a"b') == '"a\\"b"'


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


def test_leanfun_from_string_and_call():
    code = "def add (x : Int) := x + 1"
    responses = {
        code: FakeResult([], env=0),
        "#check add": FakeResult([FakeMessage("info", "add : Int -> Int")], env=0),
        "#eval add 1": FakeResult([FakeMessage("info", "2")], env=0),
    }
    server = FakeServer(responses)
    func = LeanFun.from_string(code, server=server)
    assert isinstance(func, LeanFun)
    assert func(1) == 2
    assert server.commands == [code, "#check add", "#eval add 1"]


def test_leanfun_missing_definition_raises():
    code = "def foo : Int := 1"
    responses = {
        code: FakeResult([], env=0),
        "#check foo": FakeResult([FakeMessage("info", "foo : Int")], env=0),
    }
    server = FakeServer(responses)
    with pytest.raises(ValueError, match="Missing definitions"):
        LeanFun.from_string(code, names=["bar"], server=server)


def test_leanfun_kwargs_rejected():
    code = "def id (x : Int) := x"
    responses = {
        code: FakeResult([], env=0),
        "#check id": FakeResult([FakeMessage("info", "id : Int -> Int")], env=0),
    }
    server = FakeServer(responses)
    func = LeanFun.from_string(code, server=server)
    assert isinstance(func, LeanFun)
    with pytest.raises(ValueError, match="Keyword arguments"):
        func(x=1)


def test_from_file_roundtrip():
    fixture = Path(__file__).parent / "fixtures" / "lean_ids.lean"
    funcs = LeanFun.from_file(
        str(fixture),
        names=["idNat", "idInt", "idBool", "idString"],
    )
    assert isinstance(funcs, list)
    assert all(isinstance(func, LeanFun) for func in funcs)
    typed_funcs = [func for func in funcs if isinstance(func, LeanFun)]
    func_map = {func.name: func for func in typed_funcs}
    assert func_map["idNat"](5) == 5
    assert func_map["idInt"](-7) == -7
    assert func_map["idBool"](False) is False
    assert func_map["idString"]("hello") == "hello"


@pytest.fixture(scope="module")
def lean_id_functions():
    code = "\n".join(
        [
            "def idNat (x : Nat) := x",
            "def idInt (x : Int) := x",
            "def idBool (x : Bool) := x",
            "def idString (x : String) := x",
            "def idListInt (x : List Int) := x",
            "def idPairNat (x : Nat × Nat) := x",
            "def idFloat (x : Float) := x",
        ]
    )
    names = [
        "idNat",
        "idInt",
        "idBool",
        "idString",
        "idListInt",
        "idPairNat",
        "idFloat",
    ]
    funcs = LeanFun.from_string(code, names=names)
    assert isinstance(funcs, list)
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
