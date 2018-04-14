import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

grad_clip = "GradientClipping"

@tf.RegisterGradient(grad_clip)
def _grad_identity(op, grad):
    return grad

def clipping(X, min_val, max_val):

    g = K.get_session().graph
    with g.gradient_override_map({'clip_by_value': grad_clip}):
        y = tf.clip_by_value(X, min_val, max_val, name = "clipping_layer_1")

    return y

grad_round = "GradientRounding"
@tf.RegisterGradient(grad_round)
def _grad__identity(op, grad):
    return grad

def rounding(X):

    g = K.get_session().graph
    with g.gradient_override_map({'Round': grad_round}):
        y = tf.round(X, name = "rounding_layer_1")

    return y


def masking(x, mask):
    return tf.multiply(x, mask)


class ClippingLayer(Layer):

    def __init__(self, min_val=0, max_val=1, **kwargs):
        super(ClippingLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.min_val = min_val
        self.max_val = max_val

    def build(self, input_shape):
        self.trainable_weights = []
        self.name = "clipping_layer_1"
        super(ClippingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return clipping(x, self.min_val, self.max_val)

    def compute_output_shape(self, input_shape):
        return input_shape


class RoundingLayer(Layer):

    def __init__(self, **kwargs):
        super(RoundingLayer, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self.name = "rounding_layer_1"
        self.trainable_weights = []
        super(RoundingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return rounding(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class MaskingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__()
        self.supports_masking = False
        self.mask_idx = kwargs["mask_idx"]

    def build(self, input_shape):
        self.trainable_weights = []
        self.mask = np.zeros(input_shape[1:])
        for i in range(self.mask_idx):
            self.mask[i % input_shape[1], i // input_shape[1] % input_shape[2], :] = 1.0

        super(MaskingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.multiply(x, self.mask)

    def compute_output_shape(self, input_shape):
        return input_shape
