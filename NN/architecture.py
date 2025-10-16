import func
import numpy as np
from numpy.typing import NDArray


class TwoLayerNet:
    """
    neural network with single hidden layer
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.params: dict[str, NDArray] = dict()
        self.params["w1"] = np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros((1, hidden_size))
        self.params["w2"] = np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros((1, output_size))

    def predict(self, x: NDArray) -> NDArray:
        a1 = np.dot(x, self.params["w1"]) + self.params["b1"]
        z1 = func.sigmoid(a1)
        a2 = np.dot(z1, self.params["w2"]) + self.params["b2"]
        y = func.softmax(a2)
        return y

    def loss(self, x: NDArray, y: NDArray) -> float:
        a = self.predict(x)
        return func.cross_entropy(a, y)

    def accuracy(self, x: NDArray, y: NDArray) -> float:
        a = self.predict(x)
        batch_size = a.shape[0]
        return np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1)) / float(batch_size)

    def numerical_gradient(self, x: NDArray, y: NDArray) -> dict:
        loss_param = lambda W: self.loss(x, y)
        grads: dict[str, NDArray] = dict()
        for key in ("w1", "b1", "w2", "b2"):
            grads[key] = func.numerical_gradient(loss_param, self.params[key])
        return grads

    def gradient_descent(self, x: NDArray, y: NDArray, learning_rate: float = 0.01):
        grads = self.numerical_gradient(x, y)
        for key in ("w1", "b1", "w2", "b2"):
            self.params[key] -= grads[key] * learning_rate
