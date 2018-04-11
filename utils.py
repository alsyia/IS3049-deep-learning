import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import shutil


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


def generate_patch(img, patch_size, strides):
    patch_i = img.shape[0]//strides[0]
    patch_j = img.shape[1]//strides[1]

    patchs = np.zeros([patch_size[0],patch_size[1], img.shape[2], patch_i*patch_j])
    print(img)
    patch_idx = 0
    for i in range(patch_i):
        for j in range(patch_j):
            print(img[i*strides[0]*patch_size[0]:(i+1)*strides[0]*patch_size[0],
                    j*strides[1]*patch_size[1]:(j+1)*strides[1]*patch_size[1],:])
            patchs[:, :, :, patch_idx] = img[i*patch_size[0]:(i+1)*patch_size[0],
                                                j*patch_size[1]:(j+1)*patch_size[1],
                                                :]
            patch_idx += 1
    
    return patchs


a = np.reshape(np.arange(48),(4,4,3))
print(generate_patch(a,(2,2),(1,1)))