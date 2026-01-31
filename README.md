# leancall

Small helpers for calling Lean definitions from Python. Based on [lean-interact](https://github.com/augustepoiroux/LeanInteract) for core functionality.

## Setup

Lean can be installed via `lean-interact`'s installation script:

```bash
uv run install-lean
```

or by installing elan manually.

## Install from GitHub

Using pip:

```bash
python3 -m pip install git+https://github.com/philzook58/leancall.git
```

Using uv pip:

```bash
uv pip install git+https://github.com/philzook58/leancall.git
```

## Usage

Try it on [colab](https://colab.research.google.com/github/philzook58/leancall/blob/main/example.ipynb)

Define Lean code inline and call it:

```python
from leancall import from_string

code = "def add (x y : Int) := x + y"
mod = from_string(code)

assert mod.add(2, 3) == 5
```

Load from a `.lean` file:

```python
from pathlib import Path
from leancall import from_file

fixture = Path("tests/fixtures/lean_ids.lean").resolve()
mod = from_file(
    str(fixture),
    names=["idNat", "idInt", "idBool", "idString"],
)

assert mod.idBool(False) is False
```

Raw output (skip parsing):

```python
from leancall import from_string

code = "def addStr (x y : Int) := x + y"
mod = from_string(code)

assert mod.addStr(2, 3, parse=None) == "5"
```

JSON roundtrip (deserialize to Python dict):

```python
from leancall import from_string
from leancall.leanfun import Json

code = "import Lean\ndef idJson (x : Json) := x"
mod = from_string(code)

payload = {"a": 1, "b": [2, 3]}
assert mod.idJson(Json(payload)) == {"a": 1, "b": [2, 3]}
```

Dataclass + Lean structure hookup:

```python
from dataclasses import dataclass
from leancall import from_string
from leancall.leanfun import from_lean


@dataclass(frozen=True)
class Foo:
    x: int
    y: str


code = """
structure Foo where
  x : Nat
  y : String
deriving Repr

def showFoo (f : Foo) : String := reprStr f
"""
mod = from_string(code)
raw = mod.showFoo(Foo(3, "hi"))
data = from_lean(raw)
assert Foo(**data) == Foo(3, "hi")
```

## Type Mapping

| Python | Lean | Notes |
| --- | --- | --- |
| `int` | `Nat` / `Int` | `Nat` requires non-negative values. |
| `bool` | `Bool` | `True` → `true`, `False` → `false`. |
| `str` | `String` | Escaped via JSON quoting. |
| `float` | `Float` | Uses Lean `Float` literals. |
| `list[T]` | `List T` | Recurses on element type. |
| `tuple` | `α × β × ...` | Uses Lean product types. |
| `None` | `Option _` | `none` on input; `Option` deserializes to `None`. |
| `dict` | `Std.HashMap` | Encoded as `Std.HashMap.ofList`. |
| `Json` | `Json` / `Lean.Json` | Use `leancall.leanfun.Json` wrapper to emit Lean JSON constructors. |
| `set` / `frozenset` | `Std.HashSet` | Encoded as `Std.HashSet.ofList`. |
| `Array` | `Array` | Use `leancall.leanfun.Array` wrapper to emit Lean array literals (`#[...]`). |
| `dataclass` | record literal | Uses `{field := value}` notation; register `from_lean` to deserialize. |
| `numpy.ndarray` | `Array` / `List` | Requires importing `leancall.numpy` |

## Faster parsing with numpy

When numpy is available, you can use `leancall.numpy.parse` to decode large arrays faster.

```python
import numpy as np
from leancall import from_string
import leancall.numpy as lcnp

code = "def makeArray (n : Nat) : Array Nat := Array.range n"
mod = from_string(code)

raw = mod.makeArray(100000, parse=None)
assert len(raw) > 0
arr = lcnp.parse(raw)
assert arr.shape == (100000,)
```
