from itertools import count
from keras.layers import Input, Conv2D, Add, LeakyReLU, Lambda
from keras.models import Model
from CustomLayers import ClippingLayer, RoundingLayer
from ModelConfig import *
from utils import subpixel


def encoder(e_input):

    # Counters
    conv_index = count(start=1)
    leaky_index = count(start=1)
    add_index = count(start=1)

    # TODO: Add normalization and padding here
    e = Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e_input)
    e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)
    e = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e)
    e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)

    e_skip_connection = e

    # Create three residual blocks
    for i in range(3):
        e = Conv2D(name=f"e_conv_{next(conv_index)}", **e_res_block_conv_params)(e)
        e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)
        e = Conv2D(name=f"e_conv_{next(conv_index)}", **e_res_block_conv_params)(e)
        e = Add(name=f"e_add_{next(add_index)}")([e, e_skip_connection])
        e_skip_connection = e

    e = Conv2D(filters=96, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e)

    encoded = RoundingLayer()(e)

    return encoded

def decoder(encoded):
    # Counters
    conv_index = count(start=1)
    lambda_index = count(start=1)
    leaky_index = count(start=1)
    add_index = count(start=1)

    d = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(encoded)
    d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

    d_skip_connection = d

    # Add three residual blocks
    for j in range(3):
        d = Conv2D(name=f"d_conv_{next(conv_index)}", **d_res_block_conv_params)(d)
        d = LeakyReLU(alpha=a, name=f"d_leaky_{next(leaky_index)}")(d)
        d = Conv2D(name=f"d_conv_{next(conv_index)}", **d_res_block_conv_params)(d)
        d = Add(name=f"d_add_{next(add_index)}")([d, d_skip_connection])
        d_skip_connection = d

    d = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(d)
    d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

    d = Conv2D(filters=12, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(d)
    d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

    # TODO: Add denormalize

    decoded = ClippingLayer()(d)

    return decoded

def build_model(input_shape=e_input_shape):
    e_input = Input(shape=input_shape, name="e_input_1")
    return Model(e_input, decoder(encoder(e_input)))