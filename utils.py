from datetime import datetime
import os
import shutil

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

    padded = np.zeros(
        [img.shape[0] + 2 * p, img.shape[1] + 2 * p, img.shape[2]])
    padded[:, :, 0] = np.pad(img[:, :, 0], p, mode="reflect")
    padded[:, :, 1] = np.pad(img[:, :, 1], p, mode="reflect")
    padded[:, :, 2] = np.pad(img[:, :, 2], p, mode="reflect")
    return padded


def generate_experiment():
    # create a folder for the experiment
    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_name = "exp_" + time
    exp_path = "experiments/" + exp_name
    os.mkdir(exp_path)
    shutil.copy("ModelConfig.py", exp_path)
    return exp_path


def add_weight_experiment(exp_path, weight_path):
    if not os.path.exists(exp_path):
        raise Exception("experiment path does not exists")
    if not os.path.exists(weight_path):
        raise Exception("weight path does not exists")
    shutil.copy(weight_path, exp_path)
