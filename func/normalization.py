import numpy as np

from func.activations import Var


def softmax(z: Var) -> Var:
    """
    softmax function

    support tensor input

    use it to normalize output layer
    """
    C = np.max(z)
    exp_z = np.exp(z - C)
    return exp_z / np.sum(exp_z)
