from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Optimizer(Protocol):
    learning_rate: float

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None: ...


class SGD:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate: float = learning_rate

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        for key in params.keys():
            params[key] -= grads[key] * self.learning_rate


class Momentum:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.v: Optional[dict[str, NDArray]] = None
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.v is None:
            self.v = dict()
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)
        for key in params.keys():
            self.v[key] = self.v[key] * self.momentum - self.learning_rate * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.h: Optional[dict[str, NDArray]] = None
        self.learning_rate = learning_rate

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.h is None:
            self.h = dict()
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        offset: float = 1e-6
        for key in params.keys():
            self.h[key] += np.square(grads[key])
            params[key] -= (
                self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + offset)
            )


class RMSProp:
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.99) -> None:
        self.h: Optional[dict[str, NDArray]] = None
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.h is None:
            self.h = dict()
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        offset: float = 1e-6
        for key in params.keys():
            self.h[key] = self.decay_rate * self.h[key] + (
                1 - self.decay_rate
            ) * np.square(grads[key])
            params[key] -= (
                self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + offset)
            )


class Adam:
    def __init__(
        self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m: dict[str, NDArray]
        self.v: dict[str, NDArray]
        self.iter: int = 0

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.iter == 0:
            self.m = dict()
            self.v = dict()
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)
        self.iter += 1
        offset: float = 1e-6
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(
                grads[key]
            )
            m_hat = self.m[key] / (1 - self.beta1**self.iter)
            v_hat = self.v[key] / (1 - self.beta2**self.iter)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + offset)
