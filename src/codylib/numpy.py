from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("numpy is required for codylib.numpy") from exc

from .leanfun import to_lean


@to_lean.register(np.generic)
def _(x) -> str:
    return to_lean(x.item())


@to_lean.register(np.ndarray)
def _(x) -> str:
    return to_lean(x.tolist())
