from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("numpy is required for leancall.numpy") from exc

from .leanfun import to_lean
from io import StringIO


@to_lean.register(np.generic)
def _(x) -> str:
    return to_lean(x.item())


@to_lean.register(np.ndarray)
def _(x) -> str:
    if x.shape == ():
        return to_lean(x.item())
    return "#[" + ", ".join(map(to_lean, x)) + "]"


_trans_table = str.maketrans("#,[()]\n", "       ")  # remove all delimiters to spaces


def parse(s: str) -> np.ndarray:
    """
    Parsing large array output using this function can be much faster
    """
    return np.loadtxt(StringIO(s.translate(_trans_table)))
