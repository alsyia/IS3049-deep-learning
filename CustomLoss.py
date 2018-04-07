import tensorflow as tf
from keras.losses import mse

from ModelConfig import *


def loss(x_true, x_pred):
    loss = loss_params["mse"] * mse(x_true, x_pred)
    return loss


def code(x_true,x_pred):
    return loss_params["bit"] * tf.nn.moments(x_pred,[1,2,3])[1]


def perceptual_2(x_true, x_pred):
    return loss_params["perceptual_2"]*mse(x_true, x_pred)


def perceptual_5(x_true, x_pred):
    return loss_params["perceptual_5"]*mse(x_true, x_pred)