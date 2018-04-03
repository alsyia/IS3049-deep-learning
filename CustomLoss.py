from keras.losses import mse
import tensorflow as tf
import numpy as np

from ModelConfig import *


def loss(x_true, x_pred):
    loss =  loss_params["mse"] * d(x_true, x_pred)
    return loss

def code(x_true,x_pred):
    return loss_params["bit"] * tf.nn.moments(x_pred,[1,2,3])[1]

def d(x_true, x_pred):
    return mse(x_true, x_pred)

def q(x_true, x_pred):
    return mse(x_true, x_pred)