import numpy as np
from numpy.typing import NDArray


def softmax(z: NDArray) -> NDArray:
    """
    softmax function

    support tensor input

    use it to normalize output layer
    """
    if z.ndim == 1:
        z = z.reshape(1, -1)
    C = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - C)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
