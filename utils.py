import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

def subpixel(x, scale=2):
    """
    function for Lambda - subpixel layer
    """
    return tf.depth_to_space(x, scale)