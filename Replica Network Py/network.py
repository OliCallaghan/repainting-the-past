from __future__ import division, print_function, absolute_import

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import network_input

FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 1
DATA_DIR = './models'
USE_FP16 = False

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 100
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 3e-5
WEIGHT_FACTOR = 20
TRAINING = True

TOWER_NAME = 'tower'


def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initialiser):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initialiser, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv(x, k, b, stride=1):
    # Maybe declare kernels and biases inside scope here?
    x = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv_d(x, k, b, stride=1, dilation=2):
    # Maybe declare kernels and biases inside scope here?
    x = tf.nn.atrous_conv2d(x, k, dilation, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv_t(x, k, b, frac_stride=1, stride=1):
    # Maybe declare kernels and biases inside scope here?
    x_shape = tf.shape(x)
    x = tf.nn.conv2d_transpose(x, k, [1,56,56,128], strides=[1, frac_stride, frac_stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def network(x):
    # x = [batch_size, width, height, channels]
    deviation = 1

    # conv1
    with tf.variable_scope('conv1_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 1, 64],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv1_1 = conv(x, kernel, bias)
    with tf.variable_scope('conv1_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 64],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv1_2 = conv(conv1_1, kernel, bias, stride=2)
        conv1_2_bn = tf.layers.batch_normalization(conv1_2, training=True)

    # conv2
    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 128],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv2_1 = conv(conv1_2_bn, kernel, bias)
    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 128],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv2_2 = conv(conv2_1, kernel, bias, stride=2)
        conv2_2_bn = tf.layers.batch_normalization(conv2_2, training=True)

    # conv3
    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv3_1 = conv(conv2_2_bn, kernel, bias)

    with tf.variable_scope('conv3_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv3_2 = conv(conv3_1, kernel, bias)

    with tf.variable_scope('conv3_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv3_3 = conv(conv3_2, kernel, bias, stride=2)
        conv3_3_bn = tf.layers.batch_normalization(conv3_3, training=True)

    # conv4
    with tf.variable_scope('conv4_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv4_1 = conv(conv3_3_bn, kernel, bias)

    with tf.variable_scope('conv4_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv4_2 = conv(conv4_1, kernel, bias)

    with tf.variable_scope('conv4_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv4_3 = conv(conv4_2, kernel, bias)
        conv4_3_bn = tf.layers.batch_normalization(conv4_3, training=True)

    # conv5
    with tf.variable_scope('conv5_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_1 = conv_d(conv4_3_bn, kernel, bias)

    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_2 = conv_d(conv5_1, kernel, bias)

    with tf.variable_scope('conv5_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_3 = conv_d(conv5_2, kernel, bias)
        conv5_3_bn = tf.layers.batch_normalization(conv5_3, training=True)

    # conv6
    with tf.variable_scope('conv6_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_1 = conv_d(conv5_3_bn, kernel, bias)

    with tf.variable_scope('conv6_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_2 = conv_d(conv6_1, kernel, bias)

    with tf.variable_scope('conv6_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_3 = conv_d(conv6_2, kernel, bias)
        conv6_3_bn = tf.layers.batch_normalization(conv6_3, training=True)

    with tf.variable_scope('conv7_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv7_1 = conv(conv6_3_bn, kernel, bias)

    with tf.variable_scope('conv7_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv7_2 = conv(conv7_1, kernel, bias)

    with tf.variable_scope('conv7_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv7_3 = conv(conv7_2, kernel, bias)
        conv7_3_bn = tf.layers.batch_normalization(conv7_3, training=True)

    with tf.variable_scope('conv8_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[4, 4, 128, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv8_1 = conv_t(conv7_3_bn, kernel, bias, frac_stride=2)

    with tf.variable_scope('conv8_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 32],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        conv8_2 = conv(conv8_1, kernel, bias)

    with tf.variable_scope('conv8_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 3],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [3], tf.constant_initializer(0.0))
        conv8_3 = conv(conv8_2, kernel, bias)

    with tf.variable_scope('conv8_3_act') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 3, 3],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [3], tf.constant_initializer(0.0))
        conv8_3_act = conv(conv8_3, kernel, bias)
        conv8_3_act = tf.sigmoid(conv8_3_act)

    return conv8_3_act


def loss(AB_expected, AB_channels):
    L2loss = tf.reduce_mean(tf.pow(AB_expected - AB_channels, 2))

    tf.add_to_collection('losses', L2loss)


def input():
    return network_input.inputs(False, './resources', BATCH_SIZE)
