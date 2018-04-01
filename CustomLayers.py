from keras.layers import Layer
import tensorflow as tf
import keras.backend as K
import numpy as np

def clipping(X):

    grad_name = "GradientClipping"

    @tf.RegisterGradient(grad_name)
    def _grad_identity(op, grad):
        return grad

    g = K.get_session().graph
    with g.gradient_override_map({'clip_by_value': grad_name}):
        y = tf.clip_by_value(X, 0, 255)

    return y

def rounding(X):

    grad_name = "GradientRounding"

    @tf.RegisterGradient(grad_name)
    def _grad__identity(op, grad):
        return grad

    g = K.get_session().graph
    with g.gradient_override_map({'Round': grad_name}):
        y = tf.round(X)

    return y

def masking(x, mask):
    return tf.multiply(x, mask)

class ClippingLayer(Layer):

    def __init__(self, **kwargs):
        super(ClippingLayer, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(ClippingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return clipping(x)

    def compute_output_shape(self, input_shape):
        return input_shape

class RoundingLayer(Layer):

    def __init__(self, **kwargs):
        super(RoundingLayer, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
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
            self.mask[i%input_shape[1],i//input_shape[1]%input_shape[2],:] = 1.0

        super(MaskingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.multiply(x,self.mask)

    def compute_output_shape(self, input_shape):
        return input_shape