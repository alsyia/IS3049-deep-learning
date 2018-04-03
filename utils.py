import numpy as np
import tensorflow as tf


def subpixel(x, scale=2):
    """
    function for Lambda - subpixel layer
    """
    return tf.depth_to_space(x, scale)


def mirror_padding(img, p):
    """
    function for mirror padding, p is the border added
    """

    padded = np.zeros([img.shape[0] + 2 * p, img.shape[1] + 2 * p, img.shape[2]])
    padded[:, :, 0] = np.pad(img[:, :, 0], p, mode="reflect")
    padded[:, :, 1] = np.pad(img[:, :, 1], p, mode="reflect")
    padded[:, :, 2] = np.pad(img[:, :, 2], p, mode="reflect")
    return padded


def itervalues(dic):
    grid = {}
    for i in dic.keys():
        for j in range(len(dic[i])):
            if j not in grid.keys():
                grid[j] = {}
            grid[j][i] = dic[i][j]
    return grid
