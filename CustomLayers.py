import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Layer


@tf.RegisterGradient("GradientClipping")
def _grad_identity(op, grad):
    return grad


def clipping(X, min_val, max_val):

    g = K.get_session().graph
    with g.gradient_override_map({'clip_by_value': "GradientClipping"}):
        y = tf.clip_by_value(X, min_val, max_val)

    return y


@tf.RegisterGradient("GradientRounding")
def _grad__identity(op, grad):
    return grad


def rounding(X):

    g = K.get_session().graph
    with g.gradient_override_map({'Round': "GradientRounding"}):
        y = tf.round(X)

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
        self.trainable_weights = []
        super(RoundingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return rounding(x)

    def compute_output_shape(self, input_shape):
        return input_shape

# not used
#class MaskingLayer(Layer):
#    def __init__(self, **kwargs):
#        super(MaskingLayer, self).__init__(**kwargs)
#        self.supports_masking = False
#        self.mask_idx = kwargs["mask_idx"]
#
#    def build(self, input_shape):
#        self.trainable_weights = []
#        self.mask = np.zeros(input_shape[1:])
#        for i in range(self.mask_idx):
#            self.mask[i % input_shape[1], i // input_shape[1] %
#                      input_shape[2], :] = 1.0
#
#        super(MaskingLayer, self).build(input_shape)
#
#    def call(self, x, mask=None):
#        return tf.multiply(x, self.mask)
#
#    def compute_output_shape(self, input_shape):
#        return input_shape


class PatchingLayer(Layer):
    def __init__(self, **kwargs):
        super(PatchingLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.patch_size = (8, 8)
        self.strides = (8, 8)
        self.output_size = (64, 64)

    def build(self, input_shape):
        self.trainable_weights = []
        super(PatchingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        output = tf.extract_image_patches(x,
                                          ksizes=(1, *self.patch_size, 1),
                                          strides=(1, *self.strides, 1),
                                          rates=(1, 1, 1, 1), padding='SAME')

        # on reshape pour séparer les patchs des channels
        # TO DO : verifier que les patchs et channels sont correctement séparés.
        output = tf.reshape(output, (tf.shape(x)[0], *self.patch_size, x.shape[3], output.shape[3]//3))

        # on pad pour atteindre une taille (32,64,64,3,8)
        pad = [(self.output_size[0] - self.patch_size[0])//2,
               (self.output_size[1] - self.patch_size[1])//2]
        output = tf.pad(output, [[0, 0], pad, pad, [0, 0], [0, 0]])

        # on génère une liste de tenseur, chaque tenseur est un patch paddé
        final_size = (tf.shape(x)[0], *self.output_size, x.shape[3])
        outputs = [tf.reshape(tf.slice(output,
                                       begin=(0, 0, 0, 0, patch_idx),
                                       size=(*final_size, 1)), shape=final_size) for patch_idx in range(output.shape[-1])]
        return outputs

    def compute_output_shape(self, input_shape):
        # TO DO : le nombre de patch est toujours hard codé
        return [(input_shape[0], self.patch_size[0], self.patch_size[1], input_shape[3]) for idx in range(64)]
