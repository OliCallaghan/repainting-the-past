from __future__ import division, print_function, absolute_import
import tensorflow as tf
from datetime import datetime
import time

import network

LOG_FREQ = 5
LOG_DEV_PLACEMENT = True
TRAIN_DIR = 'train_dir'
MAX_STEP = 100


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            L_channel, AB_channels = network.input()

        logits = network.network(L_channel)
        loss_op = tf.reduce_sum(tf.pow(logits - AB_channels, 2))/(2*28*28)

        optimiser = tf.train.AdamOptimizer(learning_rate=network.INITIAL_LEARNING_RATE)
        train_op = optimiser.minimize(loss_op)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss_op)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % LOG_FREQ == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = LOG_FREQ * network.BATCH_SIZE / duration
                    sec_per_batch = float(duration / LOG_FREQ)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=TRAIN_DIR,
            save_checkpoint_secs=60,
            hooks=[tf.train.StopAtStepHook(last_step=300000),
                   tf.train.NanTensorHook(loss_op),
                   _LoggerHook()],
            config=tf.ConfigProto(log_device_placement=LOG_DEV_PLACEMENT)
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

train()
