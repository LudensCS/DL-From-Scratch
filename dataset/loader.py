import numpy as np
from datasets import DatasetDict, DownloadMode, load_dataset
from numpy._typing import NDArray


def load(one_hot: bool = False) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    load data from the remote 'mnist' dataset

    x_train.shape = (60000,28*28), which represents (epoch_size,height*width)

    y_train.shape = (60000,), which represents(epoch_size,)

    When one_hot=True, the y_train array has the shape (60000, 10).

    For example, the label 1 is represented in one-hot encoding as [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
    """
    data: DatasetDict = load_dataset("ylecun/mnist")  # type:ignore
    assert isinstance(data, DatasetDict)
    x_train = np.array([np.array(img) for img in data["train"]["image"]])
    y_train = np.array(data["train"]["label"])
    x_test = np.array([np.array(img) for img in data["test"]["image"]])
    y_test = np.array(data["test"]["label"])
    # flatten
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    # normalization
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    # one-hot encoding
    if one_hot:
        # number of output classes
        classes: int = 10
        y_train = np.eye(classes)[y_train]
        y_test = np.eye(classes)[y_test]
    return (x_train, y_train, x_test, y_test)
