import numpy as np
from numpy.typing import NDArray


def mean_squared(a: NDArray, y: NDArray) -> float:
    """
    mean squared error function

    a : Model predictions

    y : True labels
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
        y = y.reshape(1, -1)
    batch_size = y.shape[0]
    return np.sum((a - y) ** 2 / 2) / batch_size


def cross_entropy(a: NDArray, y: NDArray) -> float:
    """
    cross entropy error function

    a : Model predictions

    y : True labels

    offset : a protective offset to avoid calculating log(0)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
        y = y.reshape(1, -1)
    offset = 1e-7
    batch_size = y.shape[0]
    return -np.sum(y * np.log(a + offset)) / batch_size
