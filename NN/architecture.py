from typing import Dict, Optional, OrderedDict, Protocol, cast, runtime_checkable

import func
import numpy as np
from numpy.typing import NDArray

from NN import layers, optimizers


@runtime_checkable
class NeuralNetwork(Protocol):
    optimizer: Optional[optimizers.Optimizer]
    params: dict[str, NDArray]
    layers: OrderedDict[str, layers.Layer]

    def predict(self, x: NDArray) -> NDArray: ...
    def loss(self, x: NDArray, y: NDArray) -> float: ...
    def configure_optimizer(self, optimizer: optimizers.Optimizer) -> None: ...
    def accuracy(self, x: NDArray, y: NDArray) -> float: ...
    def autograd(self, x: NDArray, y: NDArray) -> tuple[dict[str, NDArray], float]: ...
    def train_step(
        self, x: NDArray, y: NDArray, learning_rate: Optional[float]
    ) -> float: ...


class MultiLayerNet:
    """
    neural network with multiple hidden layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        optimizer: Optional[optimizers.Optimizer] = None,
        weight_init: float = 0.01,
    ) -> None:
        self.optimizer = optimizer or optimizers.SGD()
        # initial params
        self.params: Dict[str, NDArray] = dict()
        self.params["w1"] = weight_init * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros((1, hidden_size))
        self.params["w2"] = weight_init * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros((1, output_size))

        # initial layers
        self.layers: OrderedDict[str, layers.Layer] = OrderedDict()
        self.layers["Affine1"] = layers.Affine(self.params["w1"], self.params["b1"])
        self.layers["Relu1"] = layers.ReLU()
        self.layers["Affine2"] = layers.Affine(self.params["w2"], self.params["b2"])

        self.lastLayer = layers.SoftmaxWithLoss()

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

    def autograd(self, x: NDArray, y: NDArray) -> tuple[dict[str, NDArray], float]:
        """
        calc gradient by backward propagation
        """
        # forward
        loss = self.loss(x, y)
        # backward
        dout = self.lastLayer.backward(dout=1)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads: dict[str, NDArray] = dict()
        grads["w1"] = cast(layers.Affine, self.layers["Affine1"]).dw
        grads["b1"] = cast(layers.Affine, self.layers["Affine1"]).db
        grads["w2"] = cast(layers.Affine, self.layers["Affine2"]).dw
        grads["b2"] = cast(layers.Affine, self.layers["Affine2"]).db
        return grads, loss

    def configure_optimizer(self, optimizer: optimizers.Optimizer) -> None:
        self.optimizer = optimizer

    def train_step(
        self, x: NDArray, y: NDArray, learning_rate: Optional[float] = None
    ) -> float:
        grads, loss = self.autograd(x, y)
        if learning_rate is not None:
            old_lr = self.optimizer.learning_rate
            try:
                self.optimizer.learning_rate = learning_rate
                self.optimizer.update(self.params, grads)
            finally:
                self.optimizer.learning_rate = old_lr
        else:
            self.optimizer.update(self.params, grads)
        return loss
