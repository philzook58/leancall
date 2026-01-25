# leancall

Small helpers for calling Lean definitions from Python. Based on [lean-interact](https://github.com/augustepoiroux/LeanInteract) for core functionality.

## Setup

Lean can be installed via `lean-interact` or by installing elan manually:

```bash
uv run install-lean
```

## Install from GitHub

Using pip:

```bash
python3 -m pip install git+https://github.com/philzook58/codylib.git
```

Using uv pip:

```bash
uv pip install git+https://github.com/philzook58/codylib.git
```

## Usage

Try it on [colab](https://colab.research.google.com/github/philzook58/codylib/blob/main/example.ipynb)
Define Lean code inline and call it:

```python
from leancall import from_string

code = "def add (x y : Int) := x + y"
mod = from_string(code)

print(mod.add(2, 3))  # 5
```

Load from a `.lean` file:

```python
from leancall import from_file

mod = from_file(
    "tests/fixtures/lean_ids.lean",
    names=["idNat", "idInt", "idBool", "idString"],
)

print(mod.idBool(False))  # False
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
| `dict` | `Json` / `Lean.Json` | Encoded as Lean JSON constructors. |
| `dataclass` | record literal | Uses `{field := value}` notation; register `from_lean` to deserialize. |
| `numpy.ndarray` | `Array` / `List` | Converts via `leancall.numpy` helpers. |
