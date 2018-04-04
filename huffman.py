"""
Source :
https://rosettacode.org/wiki/Huffman_coding#Python
"""
import math
import numpy as np
import tensorflow as tf

def get_compressed_size(X):
    U = tf.unique_with_counts(X)
    
    # We look for a and b so U.count.size = 2**a - b
    a = tf.to_int32(tf.ceil(tf.log(tf.to_float(tf.size(U.y))) / tf.log(2.)))
    b = tf.pow(2, a) - tf.size(U.y)
    return tf.reduce_sum(U.count * a) - tf.reduce_sum(U.count[:b])

X = tf.constant([1, 1, 0, 0, 1, 2, 2, 0, 1, 2])


sess = tf.InteractiveSession()
res = sess.run(get_compressed_size(X))
print(res)


