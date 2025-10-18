import dataset
import numpy as np
from numpy.typing import NDArray
from torch.autograd import grad

import NN.architecture


def check_gradient():
    (_, _, x_test, y_test) = dataset.load(one_hot=True)
    nn = NN.MultiLayerNet(x_test.shape[1], 10, 10)
    x_test = x_test[:10]
    y_test = y_test[:10]
    grad_numerical: dict[str, NDArray] = nn.numerical_gradient(x_test, y_test)
    grad_anylysis: dict[str, NDArray] = nn.gradient(x_test, y_test)
    for key in grad_numerical.keys():
        print(key, np.average(np.abs(grad_numerical[key] - grad_anylysis[key])))
