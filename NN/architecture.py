from typing import Dict, OrderedDict

import func
import numpy as np
from numpy.typing import NDArray

import NN.layers


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


class MutiLayerNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init: float = 0.01,
    ) -> None:
        # initial params
        self.params: Dict[str, NDArray] = dict()
        self.params["w1"] = weight_init * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros((1, hidden_size))
        self.params["w2"] = weight_init * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros((1, output_size))

        # initial layers
        self.layers: OrderedDict = OrderedDict()
        self.layers["Affine1"] = NN.Affine(self.params["w1"], self.params["b1"])
        self.layers["Relu1"] = NN.ReLU()
        self.layers["Affine2"] = NN.Affine(self.params["w2"], self.params["b2"])

        self.lastLayer = NN.SoftmaxWithLoss()

    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: NDArray, y: NDArray) -> float:
        a = self.predict(x)
        return self.lastLayer.forward(a, y)

    def accuracy(self, x: NDArray, y: NDArray) -> float:
        a = self.predict(x)
        batch_size = y.shape[0]
        return np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1)) / batch_size

    def numerical_gradient(self, x: NDArray, y: NDArray) -> dict:
        loss_param = lambda W: self.loss(x, y)
        grads: dict[str, NDArray] = dict()
        for key in ("w1", "b1", "w2", "b2"):
            grads[key] = func.numerical_gradient(loss_param, self.params[key])
        return grads

    def gradient(self, x: NDArray, y: NDArray) -> dict:
        """
        calc gradient by backward propagation
        """
        # forward
        self.loss(x, y)
        # backward
        dout = self.lastLayer.backward(dout=1)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads: dict[str, NDArray] = dict()
        grads["w1"] = self.layers["Affine1"].dw
        grads["b1"] = self.layers["Affine1"].db
        grads["w2"] = self.layers["Affine2"].dw
        grads["b2"] = self.layers["Affine2"].db
        return grads

    def gradient_descent(
        self, x: NDArray, y: NDArray, learning_rate: float = 0.01
    ) -> None:
        grads = self.gradient(x, y)
        for key in ("w1", "b1", "w2", "b2"):
            self.params[key] -= grads[key] * learning_rate
