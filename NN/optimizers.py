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
    def __init__(
        self, learning_rate: float = 0.01, resistance_factor: float = 0.9
    ) -> None:
        self.v: Optional[dict[str, NDArray]] = None
        self.learning_rate: float = learning_rate
        self.resistance_factor: float = resistance_factor

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.v is None:
            self.v = dict()
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)
        for key in params.keys():
            self.v[key] = (
                self.v[key] * self.resistance_factor - self.learning_rate * grads[key]
            )
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
        for key in params.keys():
            offset: float = 1e-6
            self.h[key] += np.square(grads[key])
            params[key] -= (
                self.learning_rate * grads[key] / np.sqrt(self.h[key] + offset)
            )


class RMSProp:
    def __init__(self, learning_rate: float = 0.01, weight_factor: float = 0.9) -> None:
        self.h: Optional[dict[str, NDArray]] = None
        self.learning_rate = learning_rate
        self.weight_factor = weight_factor

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        if self.h is None:
            self.h = dict()
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            offset: float = 1e-6
            self.h[key] = self.weight_factor * self.h[key] + (
                1 - self.weight_factor
            ) * np.square(grads[key])
            params[key] -= (
                self.learning_rate * grads[key] / np.sqrt(self.h[key] + offset)
            )


class Adam:
    def __init__(self) -> None:
        pass

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        pass
