from __future__ import division, print_function, absolute_import

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import network_input

FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 4
DATA_DIR = './models'
USE_FP16 = False

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 3e-9
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


def network(x):
    # x = [batch_size, width, height, channels]
    deviation = 5e-2

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

    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 128],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv2_1 = conv(conv1_2, kernel, bias)
    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 128],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv2_2 = conv(conv2_1, kernel, bias, stride=2)

    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv3_1 = conv(conv2_2, kernel, bias)
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

    with tf.variable_scope('conv4_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv4_1 = conv(conv3_3, kernel, bias)
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

    with tf.variable_scope('conv5_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_1 = conv(conv4_3, kernel, bias)
    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_2 = conv(conv5_1, kernel, bias)
    with tf.variable_scope('conv5_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv5_3 = conv(conv5_2, kernel, bias)

    with tf.variable_scope('conv6_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_1 = conv(conv5_3, kernel, bias)
    with tf.variable_scope('conv6_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_2 = conv(conv6_1, kernel, bias)
    with tf.variable_scope('conv6_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        conv6_3 = conv(conv6_2, kernel, bias)

    with tf.variable_scope('conv7_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 256],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        conv7_1 = conv(conv6_3, kernel, bias)
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

    with tf.variable_scope('conv8_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 128],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        conv8_1 = conv(conv7_3, kernel, bias)
    with tf.variable_scope('conv8_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 32],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        conv8_2 = conv(conv8_1, kernel, bias)
    with tf.variable_scope('conv8_3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 2],
                                             stddev=deviation,
                                             wd=0)
        bias = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
        conv8_3 = conv(conv8_2, kernel, bias)

    return conv8_3


def loss(L_channel, AB_channels):
    L2loss = tf.reduce_sum(tf.pow(L_channel - AB_channels, 2))/(2*28*28)

    tf.add_to_collection('losses', L2loss)


def input():
    return network_input.inputs(False, './resources', BATCH_SIZE)
