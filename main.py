from keras.layers import Input, Conv2D, Add, LeakyReLU, Lambda
from keras.models import Model
import tensorflow as tf
from sub_pixel import SubpixelConv2D
import cv2

# use the following name : d_ for decoder, e_ for encoder

# missing normalization
# missing mirror padding

# encoder
a = 0.3
e_input_shape = (178,178,3)
# input takes a normalized and mirror padded image
e_input = Input(shape = e_input_shape,name = "e_input_1")
e = Conv2D(filters = 64, kernel_size = (5,5),padding = 'same', strides = (2,2), name = 'e_conv_1')(e_input)
e = LeakyReLU(alpha = a, name = 'e_leaky_1')(e)
e = Conv2D(filters = 128, kernel_size = (5,5),padding = 'same', strides = (2,2), name = 'e_conv_2')(e)
e = LeakyReLU(alpha = a, name = 'e_leaky_2')(e)

skip_connection = e

# residual block 1
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_3')(e)
e = LeakyReLU(alpha = a, name = 'e_leaky_3')(e)
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_4')(e)
e = Add(name = 'e_add_1')([e,skip_connection])

skip_connection = e

# residual block 2
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_5')(e)
e = LeakyReLU(alpha = a, name = 'e_leaky_4')(e)
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_6')(e)
e = Add(name = 'e_add_2')([e,skip_connection])

skip_connection = e

# residual block 3
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_7')(e)
e = LeakyReLU(alpha = a, name = 'e_leaky_5')(e)
e = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'e_conv_8')(e)
e = Add(name = 'e_add_3')([e,skip_connection])

e = Conv2D(filters = 96, kernel_size = (5,5),padding = 'same', strides = (2,2), name = 'e_conv_9')(e)

encoder = Model(e_input,e)

# missing round
# missing GSM
# missing code

# decoder
d_input_shape = (178,178,3) # will be round output_shape

# scale used by subpix layer
scale = 2
d_input = Input(shape = d_input_shape, name = 'd_input_1')

d = Conv2D(filters = 512, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_1')(d_input)

def subpixel(x):
    """
    function for Lambda - subpixel layer
    """
    return tf.depth_to_space(x, scale)
d = Lambda(function = subpixel,name = 'd_lambda_1')(d)

skip_connection = d

d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_2')(d)
d = LeakyReLU(alpha = a, name = 'd_leaky_1')(d)
d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_3')(d)
d = Add(name = 'd_add_1')([d,skip_connection])

skip_connection = d
d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_4')(d)
d = LeakyReLU(alpha = a, name = 'd_leaky_2')(d)
d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_5')(d)
d = Add(name = 'd_add_2')([d,skip_connection])

skip_connection = d
d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_6')(d)
d = LeakyReLU(alpha = a, name = 'd_leaky_3')(d)
d = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_7')(d)
d = Add(name = 'd_add_3')([d,skip_connection])

d = Conv2D(filters = 256, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_8')(d)
d = Lambda(function = subpixel,name = 'd_lambda_2')(d)


d = Conv2D(filters = 12, kernel_size = (3,3),padding = 'same', strides = (1,1), name = 'd_conv_9')(d)
d = Lambda(function = subpixel,name = 'd_lambda_3')(d)

decoder = Model(d_input,d)

# missing denormalization
# missing clipping

# missing loss

# missing ae training