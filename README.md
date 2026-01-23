# codylib

Small helpers for calling Lean definitions from Python.

## Setup

Lean is installed via `lean-interact`:

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

Try it on [colab](https://colab.research.google.com/github/philzook58/codylib/example.ipynb)

Define Lean code inline and call it:

```python
from codylib import LeanFun

code = "def add (x y : Int) := x + y"
add = LeanFun.from_string(code)

print(add(2, 3))  # 5
```

Load from a `.lean` file:

```python
from codylib import LeanFun

funcs = LeanFun.from_file(
    "tests/fixtures/lean_ids.lean",
    names=["idNat", "idInt", "idBool", "idString"],
)
func_map = {func.name: func for func in funcs}

print(func_map["idBool"](False))  # False
```
