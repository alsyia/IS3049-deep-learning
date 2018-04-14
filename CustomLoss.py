import tensorflow as tf
from keras.losses import mse, mae
import keras.backend as K

from ModelConfig import *


def loss(x_true, x_pred):
    loss = loss_params["mse"] * mse(x_true, x_pred)
    return loss


def code(x_true, x_pred):
    return loss_params["bit"] * tf.nn.moments(x_pred, [1, 2, 3])[1]


def entropy(x_true, x_pred):
    print(x_pred.shape)
    flat = tf.reshape(x_pred, shape = [tf.shape(x_pred)[0]*tf.shape(x_pred)[1]*tf.shape(x_pred)[2]*tf.shape(x_pred)[3]])
    y, idx, count = tf.unique_with_counts(flat, out_idx=tf.int32)
    count = tf.cast(count, tf.float32)

    sum_count = tf.reduce_sum(count)

    freq = tf.divide(count,sum_count)
    entropy = tf.reduce_mean(-tf.reduce_sum(freq * tf.log(freq)))
    return loss_params["entropy"] * entropy


def perceptual_2(x_true, x_pred):
    return loss_params["perceptual_2"]*mse(x_true, x_pred)


def perceptual_5(x_true, x_pred):
    return loss_params["perceptual_5"]*mse(x_true, x_pred)


def texture(x_true, x_pred):
    """
    fonction qui applique une mse sur les matrices de gram respectives des vecteurs
    """

    # on reshape pour avoir les dimensions spatiales ramenées sur une seule dimension
    reshape_true = tf.reshape(x_true, [tf.shape(x_true)[0], tf.shape(x_true)[
                              1]*tf.shape(x_true)[2], tf.shape(x_true)[3]])
    # on calcule la transposée (en figeant la dimension de batch)
    transpose_true = tf.transpose(reshape_true, perm=[0, 2, 1])

    # idem pour y_pred
    reshape_pred = tf.reshape(x_pred, [tf.shape(x_pred)[0], tf.shape(x_pred)[
                              1]*tf.shape(x_pred)[2], tf.shape(x_pred)[3]])
    transpose_pred = tf.transpose(reshape_pred, perm=[0, 2, 1])

    # on fait le produit matriciel du vecteur et sa transposée, ce qui revient à faire un produit scalaire
    # entre les images
    gram_true = tf.matmul(transpose_true, reshape_true)
    gram_pred = tf.matmul(transpose_pred, reshape_pred)

    return loss_params["texture"]*mse(gram_true, gram_pred)
