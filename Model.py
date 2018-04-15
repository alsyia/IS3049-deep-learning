from itertools import count

from keras.layers import Input, Conv2D, Add, LeakyReLU, Lambda
from keras.models import Model

from CustomLayers import ClippingLayer, RoundingLayer
from ModelConfig import *
from Utils import subpixel


def encoder(e_input):
    # Counters
    conv_index = count(start=1)
    leaky_index = count(start=1)
    add_index = count(start=1)

    e = Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2),
               name="e_conv_" + str(str(next(conv_index))))(e_input)
    e = LeakyReLU(alpha=a, name="e_leaky_" + str(next(leaky_index)))(e)
    e = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2), name="e_conv_" + str(next(conv_index)))(
        e)
    e = LeakyReLU(alpha=a, name="e_leaky_" + str(next(leaky_index)))(e)

    e_skip_connection = e

    # Create three residual blocks
    for i in range(3):
        e = Conv2D(name="e_conv_" + str(next(conv_index)), **e_res_block_conv_params)(e)
        e = LeakyReLU(alpha=a, name="e_leaky_" + str(next(leaky_index)))(e)
        e = Conv2D(name="e_conv_" + str(next(conv_index)), **e_res_block_conv_params)(e)
        e = Add(name="e_add_" + str(next(add_index)))([e, e_skip_connection])
        e_skip_connection = e

    e = Conv2D(filters=96, kernel_size=(5, 5), padding='same', strides=(2, 2), name="e_conv_" + str(next(conv_index)))(
        e)
    e = RoundingLayer()(e)

    return e


def decoder(encoded):
    # Counters
    conv_index = count(start=1)
    lambda_index = count(start=1)
    leaky_index = count(start=1)
    add_index = count(start=1)

    d = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1, 1), name="d_conv_" + str(next(conv_index)))(
        encoded)
    d = Lambda(function=subpixel, name="d_lambda_" + str(next(lambda_index)))(d)

    d_skip_connection = d

    # Add three residual blocks
    for j in range(3):
        d = Conv2D(name="d_conv_" + str(next(conv_index)), **d_res_block_conv_params)(d)
        d = LeakyReLU(alpha=a, name="d_leaky_" + str(next(leaky_index)))(d)
        d = Conv2D(name="d_conv_" + str(next(conv_index)), **d_res_block_conv_params)(d)
        d = Add(name="d_add_" + str(next(add_index)))([d, d_skip_connection])
        d_skip_connection = d

    d = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), name="d_conv_" + str(next(conv_index)))(
        d)
    d = Lambda(function=subpixel, name="d_lambda_" + str(next(lambda_index)))(d)

    d = Conv2D(filters=12, kernel_size=(3, 3), padding='same', strides=(1, 1), name="d_conv_" + str(next(conv_index)))(
        d)
    d = Lambda(function=subpixel, name="d_lambda_" + str(next(lambda_index)))(d)

    d = ClippingLayer(0, 1)(d)

    return d


def build_model(perceptual_model):
    # Define input layer
    e_input = Input(shape=e_input_shape, name="e_input_1")
    # Chain models
    encoded = encoder(e_input)
    decoded = decoder(encoded)
    featured = perceptual_model(decoded)
    # Define global models with multiple outputs
    # autoencodeur = Model(e_input, [encoded, decoded, *featured])

    block_2 = Lambda(lambda x: x, name="VGG_block_2")(featured[0])
    block_5 = Lambda(lambda x: x, name="VGG_block_5")(featured[1])
    autoencodeur = Model(e_input, [encoded, decoded, block_2, block_5])
    # Return autoencodeur (we are going to train it) and perceptual_model (will be used in the loss)
    return autoencodeur, perceptual_model


def get_encoder(autoencoder):
    return autoencoder.layers[1]


def get_decoder(autoencoder):
    return autoencoder.layers[2]
