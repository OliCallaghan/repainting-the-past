from __future__ import division, print_function, absolute_import

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import network_input

FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 128
DATA_DIR = './models'
USE_FP16 = False

# MOVING_AVERAGE_DECAY = 0.9999
# NUM_EPOCHS_PER_DECAY = 350
# LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.00000000000001

TOWER_NAME = 'tower'

def variable_on_cpu(name, shape, initialiser):
    with tf.device('/cpu:0'):
        dtype = tf.float32 if FLAGS.use_fp16 else tf.float16
        var = tf.get_variable(name, shape, initialiser=initialiser, dtype=dtype)
    return var


def conv(x, k, b, stride=1):
    x = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def network(x):
    # x = [batch_size, width, height, channels]
    weights = {
        'conv1_1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'conv1_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'conv2_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'conv2_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),

        'conv3_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'conv3_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'conv3_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),

        'conv4_1': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        'conv4_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'conv4_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),

        'conv5_1': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'conv5_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'conv5_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),

        'conv6_1': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'conv6_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'conv6_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),

        'conv7_1': tf.Variable(tf.random_normal([3, 3, 512, 256])),
        'conv7_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'conv7_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),

        'conv8_1': tf.Variable(tf.random_normal([3, 3, 256, 128])),
        'conv8_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
        'conv8_3': tf.Variable(tf.random_normal([3, 3, 128, 2]))
    }

    biases = {
        'conv1_1': tf.Variable(tf.random_normal([64])),
        'conv1_2': tf.Variable(tf.random_normal([64])),

        'conv2_1': tf.Variable(tf.random_normal([128])),
        'conv2_2': tf.Variable(tf.random_normal([128])),

        'conv3_1': tf.Variable(tf.random_normal([256])),
        'conv3_2': tf.Variable(tf.random_normal([256])),
        'conv3_3': tf.Variable(tf.random_normal([256])),

        'conv4_1': tf.Variable(tf.random_normal([512])),
        'conv4_2': tf.Variable(tf.random_normal([512])),
        'conv4_3': tf.Variable(tf.random_normal([512])),

        'conv5_1': tf.Variable(tf.random_normal([512])),
        'conv5_2': tf.Variable(tf.random_normal([512])),
        'conv5_3': tf.Variable(tf.random_normal([512])),

        'conv6_1': tf.Variable(tf.random_normal([512])),
        'conv6_2': tf.Variable(tf.random_normal([512])),
        'conv6_3': tf.Variable(tf.random_normal([512])),

        'conv7_1': tf.Variable(tf.random_normal([256])),
        'conv7_2': tf.Variable(tf.random_normal([256])),
        'conv7_3': tf.Variable(tf.random_normal([256])),

        'conv8_1': tf.Variable(tf.random_normal([128])),
        'conv8_2': tf.Variable(tf.random_normal([128])),
        'conv8_3': tf.Variable(tf.random_normal([2]))
    }

    # x = tf.reshape(x, shape=[-1, f_in_x, f_in_y, 1])

    conv1_1 = conv(x, weights['conv1_1'], biases['conv1_1'])
    conv1_2 = conv(conv1_1, weights['conv1_2'], biases['conv1_2'], stride=2)

    conv2_1 = conv(conv1_2, weights['conv2_1'], biases['conv2_1'])
    conv2_2 = conv(conv2_1, weights['conv2_2'], biases['conv2_2'], stride=2)

    conv3_1 = conv(conv2_2, weights['conv3_1'], biases['conv3_1'])
    conv3_2 = conv(conv3_1, weights['conv3_2'], biases['conv3_2'])
    conv3_3 = conv(conv3_2, weights['conv3_3'], biases['conv3_3'], stride=2)

    conv4_1 = conv(conv3_3, weights['conv4_1'], biases['conv4_1'])
    conv4_2 = conv(conv4_1, weights['conv4_2'], biases['conv4_2'])
    conv4_3 = conv(conv4_2, weights['conv4_3'], biases['conv4_3'])

    conv5_1 = conv(conv4_3, weights['conv5_1'], biases['conv5_1'])
    conv5_2 = conv(conv5_1, weights['conv5_2'], biases['conv5_2'])
    conv5_3 = conv(conv5_2, weights['conv5_3'], biases['conv5_3'])

    conv6_1 = conv(conv5_3, weights['conv6_1'], biases['conv6_1'])
    conv6_2 = conv(conv6_1, weights['conv6_2'], biases['conv6_2'])
    conv6_3 = conv(conv6_2, weights['conv6_3'], biases['conv6_3'])

    conv7_1 = conv(conv6_3, weights['conv7_1'], biases['conv7_1'])
    conv7_2 = conv(conv7_1, weights['conv7_2'], biases['conv7_2'])
    conv7_3 = conv(conv7_2, weights['conv7_3'], biases['conv7_3'])

    conv8_1 = conv(conv7_3, weights['conv8_1'], biases['conv8_1'])
    conv8_2 = conv(conv8_1, weights['conv8_2'], biases['conv8_2'])
    conv8_3 = conv(conv8_2, weights['conv8_3'], biases['conv8_3'])

    return conv8_3


def input():
    return network_input.inputs(False, './resources', 2)
