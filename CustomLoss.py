from keras.losses import mse
import tensorflow as tf
import numpy as np

from ModelConfig import *


def loss(x_true, x_pred):
    loss =  loss_params["mse"] * d(x_true, x_pred)
    return loss

def code(x_true,x_pred):
    return loss_params["bit"] * tf.nn.moments(x_pred,[1,2,3])[1]

def get_compressed_size(X):
    U = tf.unique_with_counts(X)
    
    # We look for a and b so U.count.size = 2**a - b
    a = tf.to_int32(tf.ceil(tf.log(tf.to_float(tf.size(U.y))) / tf.log(2.)))
    b = tf.pow(2, a) - tf.size(U.y)
    size = tf.reduce_sum(U.count * a) - tf.reduce_sum(U.count[:b])
    return loss_params["bit"] * size

def d(x_true, x_pred):
    return mse(x_true, x_pred)

def q(x_true, x_pred):
    return mse(x_true, x_pred)