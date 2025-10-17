import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    """
    sigmoid function

    support tensor input
    """
    return 1 / (1 + np.exp(-z))


def tahn(z: NDArray) -> NDArray:
    """
    tahn fuction

    support tensor input
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu(z: NDArray) -> NDArray:
    """
    relu function

    support tensor input
    """
    return np.maximum(0, z)
