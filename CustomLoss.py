import tensorflow as tf
from keras.losses import mse, mae
import keras.backend as K

from ModelConfig import *


def loss(x_true, x_pred):
    loss = loss_params["mse"] * mae(x_true, x_pred)
    return loss


def code(x_true,x_pred):
    return loss_params["bit"] * tf.nn.moments(x_pred,[1,2,3])[1]


def perceptual_2(x_true, x_pred):
    return loss_params["perceptual_2"]*mse(x_true, x_pred)


def perceptual_5(x_true, x_pred):
    return loss_params["perceptual_5"]*mse(x_true, x_pred)


def texture(x_true, x_pred):
    
    reshape_true = tf.reshape(x_true,[tf.shape(x_true)[0],tf.shape(x_true)[1]*tf.shape(x_true)[2],tf.shape(x_true)[3]])
    transpose_true = tf.transpose(reshape_true, perm = [0,2,1])
    reshape_pred = tf.reshape(x_pred,[tf.shape(x_pred)[0],tf.shape(x_pred)[1]*tf.shape(x_pred)[2],tf.shape(x_pred)[3]])
    transpose_pred = tf.transpose(reshape_pred, perm = [0,2,1])

    gram_true = tf.matmul(transpose_true,reshape_true)
    gram_pred = tf.matmul(transpose_pred,reshape_pred)

    return loss_params["texture"]*mse(gram_true, gram_pred)