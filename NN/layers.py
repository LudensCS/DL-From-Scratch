from typing import Any, Protocol

import func
import numpy as np
from numpy.typing import NDArray


class Layer(Protocol):
    def forward(self, x: NDArray) -> NDArray: ...
    def backward(self, dout: Any) -> NDArray: ...


class ReLU:
    def __init__(self) -> None:
        self.mask: NDArray

    def forward(self, x: NDArray) -> NDArray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out: NDArray

    def forward(self, x: NDArray) -> NDArray:
        self.out = func.sigmoid(x)
        return self.out

    def backward(self, dout: NDArray) -> NDArray:
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, w: NDArray, b: NDArray) -> None:
        self.w: NDArray = w
        self.b: NDArray = b
        self.x: NDArray
        self.dw: NDArray
        self.db: NDArray

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.out: NDArray
        self.y: NDArray  # labels
        self.loss: float

    def forward(self, x: NDArray, y: NDArray) -> float:
        self.y = y
        self.out = func.softmax(x)
        self.loss = func.cross_entropy(self.out, self.y)
        return self.loss

    def backward(self, dout: float = 1) -> NDArray:
        batch_size: float = self.y.shape[0]
        dx = dout * (self.out - self.y) / batch_size
        return dx
