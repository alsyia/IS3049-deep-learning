import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from ModelConfig import *

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
# class MaskingLayer(Layer):
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
        self.patch_size = TEXTURE_PARAMS["patch_size"]
        self.strides = TEXTURE_PARAMS["patch_size"]
        self.output_size = INPUT_SHAPE
        self.patches_per_img = (INPUT_SHAPE[0] / self.patch_size[0]) * (INPUT_SHAPE[1] / self.patch_size[1])

    def build(self, input_shape):
        self.trainable_weights = []
        super(PatchingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        permute = tf.transpose(x, perm=[3, 1, 2, 0])
        patches = tf.extract_image_patches(permute,
                                           ksizes=(1, *self.patch_size, 1),
                                           strides=(1, *self.strides, 1),
                                           rates=(1, 1, 1, 1), padding='SAME')
        permute_2 = tf.transpose(patches, perm=[3, 1, 2, 0])

        pad = [(self.output_size[0] - self.patch_size[0]) // 2,
               (self.output_size[1] - self.patch_size[1]) // 2]
        print("[Patching] Output before padding shape: " + str(permute_2.shape))
        padded = tf.pad(permute_2, [[0, 0], pad, pad, [0, 0]])
        print("[Patching] Output after padding shape: " + str(padded.shape))
        return padded

    def compute_output_shape(self, input_shape):
        # TO DO : le nombre de patch est toujours hard codé
        return (BATCH_SIZE * self.patches_per_img, *self.output_size)


class DePatchingLayer(Layer):
    def __init__(self, **kwargs):
        super(DePatchingLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.patch_size = TEXTURE_PARAMS["patch_size"]
        self.strides = TEXTURE_PARAMS["patch_size"]
        self.patches_per_img = (INPUT_SHAPE[0] // self.patch_size[0]) * (INPUT_SHAPE[1] // self.patch_size[1])

    def build(self, input_shape):
        self.trainable_weights = []
        super(DePatchingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        depatch = tf.reshape(x, (tf.shape(x)[0] // self.patches_per_img, self.patches_per_img, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]))
        return depatch

    def compute_output_shape(self, input_shape):
        # TO DO : le nombre de patch est toujours hard codé
        return (input_shape[0] // self.patches_per_img, self.patches_per_img, input_shape[1], input_shape[2], input_shape[3])
