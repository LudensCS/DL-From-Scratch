"""
Methods for training models
"""

import dataset
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress

from NN import MultiLayerNet

from .persistence import load_model, save_model


def train(epoch_size: int = 1, batch_size: int = 100, learning_rate: float = 0.01):
    (x_train, y_train, x_test, y_test) = dataset.load(one_hot=True)
    # random number generator
    # nn = TwoLayerNet(x_train.shape[1], 10, 10)
    nn = MultiLayerNet(x_train.shape[1], 10, 10)
    # nn = load_model("./models/TLN.pkl")
    rng = np.random.default_rng()
    acc: list = list()
    lss: list = list()
    acc.append(nn.accuracy(x_test, y_test))
    with Progress(transient=True) as progress:
        task_epoch = progress.add_task(
            description="[cyan]Training...", total=epoch_size
        )
        Loss = nn.loss(x_test, y_test)
        lss.append(Loss)
        for epoch in range(epoch_size):
            idx = rng.permutation(x_train.shape[0])
            task_batch = progress.add_task(
                description=f"[magenta]Epoch {epoch + 1} Loss={Loss:.4f}",
                total=x_train.shape[0] // batch_size,
            )
            for batch in range(x_train.shape[0] // batch_size):
                st: int = batch * batch_size
                ed: int = (batch + 1) * batch_size
                nn.gradient_descent(
                    x_train[idx[st:ed]],
                    y_train[idx[st:ed]],
                    learning_rate=learning_rate,
                )
                if batch % 10 == 9:
                    Loss = nn.loss(x_train[idx[st:ed]], y_train[idx[st:ed]])
                    lss.append(Loss)
                    progress.update(
                        task_batch,
                        advance=1,
                        description=f"[magenta]Epoch {epoch + 1} Loss={Loss:.4f}",
                    )
                else:
                    progress.update(task_batch, advance=1)
                progress.update(
                    task_epoch, advance=float(batch_size) / x_train.shape[0]
                )
            acc.append(nn.accuracy(x_test, y_test))
            progress.remove_task(task_batch)

    save_model(nn, "./models/MLN.pkl")
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(acc)), acc, label="Accuracy", linewidth=2)
    plt.xlabel("epoch times")
    plt.ylabel("accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(0, len(lss) * 10, step=10), lss, label="Loss", linewidth=1)
    plt.xlabel("batch times")
    plt.ylabel("loss function")
    plt.title("Loss Function Curve")
    plt.legend()
    plt.show()
