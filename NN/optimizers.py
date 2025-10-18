from typing import Protocol, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class Optimizer(Protocol):
    learning_rate: float

    def __init__(self, learning_rate: float) -> None: ...

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None: ...


class SGD:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate: float = learning_rate

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        for key in params.keys():
            params[key] -= grads[key] * self.learning_rate
