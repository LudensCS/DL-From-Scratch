import numpy as np
from numpy.typing import NDArray


def im2col(x: NDArray, FH: int, FW: int, stride: int = 1, padding: int = 0) -> NDArray:
    batch_size, C, H, W = x.shape
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1
    img = np.pad(
        x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], "constant"
    )
    col = np.zeros((batch_size, C, FH, FW, OH, OW))
    for r in range(FH):
        r_max = r + OH * stride
        for c in range(FW):
            c_max = c + OW * stride
            col[:, :, r, c, :, :] = img[:, :, r:r_max:stride, c:c_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * OH * OW, -1)
    return col


def col2im(
    col: NDArray, shape: tuple, FH: int, FW: int, stride: int = 1, padding: int = 0
):
    batch_size, C, H, W = shape
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1
    col = col.reshape(batch_size, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros(
        (batch_size, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1)
    )
    for r in range(FH):
        r_max = r + OH * stride
        for c in range(FW):
            c_max = c + OW * stride
            img[:, :, r:r_max:stride, c:c_max:stride] += col[:, :, r, c, :, :]
    return img[:, :, padding : padding + OH, padding : padding + OW]
