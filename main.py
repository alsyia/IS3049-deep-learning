from itertools import count
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, LeakyReLU, Lambda
from keras.models import Model
from keras.utils import plot_model

# import cv2

# use the following name : d_ for decoder, e_ for encoder

def subpixel(x, scale=2):
    """
    function for Lambda - subpixel layer
    """
    return tf.depth_to_space(x, scale)


##### Encoder model #####

# TODO: Add normalization and mirror padding

# Encoder parameters
a = 0.3
e_input_shape = (178, 178, 3)
e_res_block_conv_params = {
    "filters": 128,
    "kernel_size": (3, 3),
    "padding": "same",
}

# Counters
conv_index = count(start=1)
leaky_index = count(start=1)
add_index = count(start=1)

# TODO: Add normalization and padding here
e_input = Input(shape=e_input_shape, name=f"e_input_1")
e = Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e_input)
e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)
e = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e)
e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)

skip_connection = e

# Create three residual blocks
for i in range(3):
    e = Conv2D(name=f"e_conv_{next(conv_index)}", **e_res_block_conv_params)(e)
    e = LeakyReLU(alpha=a, name=f"e_leaky_{next(leaky_index)}")(e)
    e = Conv2D(name=f"e_conv_{next(conv_index)}", **e_res_block_conv_params)(e)
    e = Add(name=f"e_add_{next(add_index)}")([e, skip_connection])
    skip_connection = e

e = Conv2D(filters=96, kernel_size=(5, 5), padding='same', strides=(2, 2), name=f"e_conv_{next(conv_index)}")(e)

encoder = Model(e_input,e)
plot_model(encoder, to_file='encoder.png')

# TODO: Round output, add GSM and codes

##### Decoder model #####

# Decoder parameters
d_input_shape = (178, 178, 3)  # will be round output_shape
d_res_block_conv_params = {
    "filters": 128,
    "kernel_size": (3, 3),
    "padding": "same",
}

# Counters
conv_index = count(start=1)
lambda_index = count(start=1)
leaky_index = count(start=1)
add_index = count(start=1)

d_input = Input(shape=d_input_shape, name='d_input_1')
d = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(d_input)
d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

skip_connection = d

# Add three residual blocks
for j in range(3):
    d = Conv2D(name=f"d_conv_{next(conv_index)}", **d_res_block_conv_params)(d)
    d = LeakyReLU(alpha=a, name=f"d_leaky_{next(leaky_index)}")(d)
    d = Conv2D(name=f"d_conv_{next(conv_index)}", **d_res_block_conv_params)(d)
    d = Add(name=f"d_add_{next(add_index)}")([d, skip_connection])
    skip_connection = d

d = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(d)
d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

d = Conv2D(filters=12, kernel_size=(3, 3), padding='same', strides=(1, 1), name=f"d_conv_{next(conv_index)}")(d)
d = Lambda(function=subpixel, name=f"d_lambda_{next(lambda_index)}")(d)

decoder = Model(d_input, d)
plot_model(decoder, to_file='decoder.png')

# TODO: Add denormalization and clipping

##### Loss #####
# TODO: Add loss

##### Training #####
# TODO: Add training