from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

Var = Union[NDArray, int, float]


def sigmoid(z: Var) -> Var:
    """
    sigmoid function

    support tensor input
    """
    return 1 / (1 + np.exp(-z))


def tahn(z: Var) -> Var:
    """
    tahn fuction

    support tensor input
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu(z: Var) -> Var:
    """
    relu function

    support tensor input
    """
    return np.maximum(0, z)
