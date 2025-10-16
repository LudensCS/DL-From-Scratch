from typing import Callable

import numpy as np
from numpy.typing import NDArray


def numerical_gradient(f: Callable, x: NDArray) -> NDArray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    delta = 1e-7
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        val = x[idx]
        x[idx] = val + delta
        f1 = f(x)
        x[idx] = val - delta
        f2 = f(x)
        x[idx] = val
        grad[idx] = (f1 - f2) / (2 * delta)
    return grad
