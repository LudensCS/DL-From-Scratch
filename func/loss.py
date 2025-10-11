import numpy as np
from func.activations import Var


def mean_squared(a: Var, y: Var) -> Var:
    """
    mean squared error function

    a : Model predictions

    y : True labels
    """
    return np.sum((a - y) ** 2) / 2


def cross_entropy(a: Var, y: Var) -> Var:
    """
    cross entropy error function

    a : Model predictions

    y : True labels

    offset : a protective offset to avoid calculating log(0)
    """
    offset = 1e-7
    return -np.sum(y * np.log(a + offset))
