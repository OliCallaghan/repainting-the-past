from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import os
from PIL import Image

import network

TRAIN_DIR = 'train_dir'
OUT_DIR = 'out'


def run():
    with tf.Graph().as_default() as g:
        L_channel, AB_channels = network.input()

        RGB_channels = network.network(L_channel)

        variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('restored to ' + str(global_step))
            else:
                print('No checkpoint found')
                return
            tf.train.start_queue_runners(sess=sess)
            for o in xrange(50):
                try:
                    img = sess.run(tf.reshape(RGB_channels, [56,56,3])) * 255
                    img = Image.fromarray(img, "RGB")
                    img.save(os.path.join(OUT_DIR, 'out' + str(o) + '.jpg'))
                except Exception as e:
                    print(e)
                else:
                    print('Saved ' + os.path.join(OUT_DIR, 'out' + str(o) + '.jpg'))


run()
