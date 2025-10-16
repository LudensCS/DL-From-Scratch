import dataset
import matplotlib.pyplot as plt
import numpy as np
from NN import TwoLayerNet
from numpy.typing import NDArray
from rich.progress import track

if __name__ == "__main__":
    (x_train, y_train, x_test, y_test) = dataset.load(one_hot=True)
    # random number generator
    nn = TwoLayerNet(x_train.shape[1], 10, 10)
    rng = np.random.default_rng()
    idx: NDArray = np.arange(0, x_train.shape[0], 1)
    acc: list = list()
    epoch_size: int = 10
    batch_size: int = 100
    acc.append(nn.accuracy(x_test, y_test))

    for epoch in track(range(epoch_size), description="[cyan]Training..."):
        rng.shuffle(idx)
        for batch in track(
            range(x_train.shape[0] // batch_size),
            description=f"[magenta]current epoch: {epoch + 1}",
            transient=True,
        ):
            st: int = batch * batch_size
            ed: int = (batch + 1) * batch_size
            nn.gradient_descent(x_train[idx[st:ed]], y_train[idx[st:ed]])
        acc.append(nn.accuracy(x_test, y_test))

    plt.plot(np.arange(len(acc)), acc)
    plt.show()
