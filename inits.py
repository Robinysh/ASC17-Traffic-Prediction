import tensorflow as tf
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    newShape = []
    for subshape in shape:
      if not isinstance(subshape, tuple):
        #Verticle shape: [[1],[2]]
        subshape = (subshape, 1)
      newShape.append(subshape)
    shape = newShape
    weight_shape = (shape[0][1], shape[1][1])
    init_range = np.sqrt(6.0/sum(weight_shape))
    #init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(weight_shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    if not isinstance(shape, tuple):
       shape = (shape,1)
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
