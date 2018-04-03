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
    
class MeanLayer(Layer):
    def __init__(self, **kwargs):
        super(MeanLayer, self).__init__()
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(MeanLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.reshape(tf.reduce_mean(x,axis = (1,2)),(-1,1,1,3))

    def compute_output_shape(self, input_shape):
        return input_shape

class StdLayer(Layer):
    def __init__(self, **kwargs):
        super(StdLayer, self).__init__()
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(StdLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.reshape(tf.keras.backend.std(x,axis = (1,2)),(-1,1,1,3))

    def compute_output_shape(self, input_shape):
        return input_shape

class MirrorPaddingLayer(Layer):
    def __init__(self, **kwargs):
        super(MirrorPaddingLayer, self).__init__()
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(MirrorPaddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        paddings = tf.constant([[0,0],[12,12],[12,12],[0,0]])
        return tf.pad(x, paddings = paddings, mode = "REFLECT")

    def compute_output_shape(self, input_shape):
        return input_shape

class MultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiplyLayer, self).__init__()
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(MultiplyLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.multiply(x[0],x[1])

    def compute_output_shape(self, input_shape):
        return input_shape

class NormalizeLayer(Layer):
    def __init__(self, **kwargs):
        super(NormalizeLayer, self).__init__()
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []
        super(NormalizeLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.divide(tf.subtract(x[0],x[1]),x[2])

    def compute_output_shape(self, input_shape):
        return input_shape
