import numpy as np
from datasets import DatasetDict, load_dataset
from numpy._typing import NDArray


def load() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    load data from the remote 'mnist' dataset

    x_train.shape = (60000,28,28), which represents (epoch_size,height,width)

    y_train.shape = (60000,), which represents(epoch_size,)
    """
    data: DatasetDict = load_dataset("ylecun/mnist")  # type:ignore
    assert isinstance(data, DatasetDict)
    x_train = np.array([np.array(img) for img in data["train"]["image"]])
    y_train = np.array(data["train"]["label"])
    x_test = np.array([np.array(img) for img in data["test"]["image"]])
    y_test = np.array(data["test"]["label"])
    return (x_train, y_train, x_test, y_test)
