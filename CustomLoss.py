from keras.losses import mse

from ModelConfig import *


def loss(x_true,x_pred):
    loss = 0 + loss_params["beta"] * d(x_true,x_pred)
    return loss

def d(x_true,x_pred):
    return mse(x_true,x_pred)
