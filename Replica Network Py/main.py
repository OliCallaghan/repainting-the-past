from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
from skimage import color

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Input frames location
model_path = "./model.ckpt"

# Input frame size
f_in_x = 640
f_in_y = 480

# Conv_net Params
learning_rate = 0.01
num_steps = 500
batch_size = 128
display_step = 10

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def read_frames(dataset_path, mode, batch_size):
    image_paths = list()
    if mode == 'file':
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            image_paths.append(d.split(' ')[0])
    elif mode == 'folder':
        filenames = os.listdir(dataset_path)
        for file in filenames:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                image_paths.append(os.path.join(dataset_path, file))
    else:
        raise Exception('Unknown mode')

    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    print(image_paths.get_shape())

    # Labels won't matter in the end, will strip AB from image and use as label

    # Queue
    image = tf.train.slice_input_producer([image_paths], shuffle=True)

    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)

    image = color.rgb2lab(image)

    # L channel
    image_l = image.slice

    image = tf.image.resize_images(image, [f_in_y, f_in_x])

    X, Y = tf.train.batch([image], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)

    print("HELLO")

    return X, Y


def conv(x, k, b, stride=1):
    x = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def network(x, weights, biases):
    # x = [batch_size, width, height, channels]
    x = tf.reshape(x, shape=[-1, f_in_x, f_in_y, 1])

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


weights = {
    'conv1_1': tf.Variable(tf.random_normal([7, 7, 1, 64])),
    'conv1_2': tf.Variable(tf.random_normal([7, 7, 64, 64])),

    'conv2_1': tf.Variable(tf.random_normal([7, 7, 64, 128])),
    'conv2_2': tf.Variable(tf.random_normal([7, 7, 128, 128])),

    'conv3_1': tf.Variable(tf.random_normal([7, 7, 128, 256])),
    'conv3_2': tf.Variable(tf.random_normal([7, 7, 256, 256])),
    'conv3_3': tf.Variable(tf.random_normal([7, 7, 256, 256])),

    'conv4_1': tf.Variable(tf.random_normal([7, 7, 256, 512])),
    'conv4_2': tf.Variable(tf.random_normal([7, 7, 512, 512])),
    'conv4_3': tf.Variable(tf.random_normal([7, 7, 512, 512])),

    'conv5_1': tf.Variable(tf.random_normal([7, 7, 512, 512])),
    'conv5_2': tf.Variable(tf.random_normal([7, 7, 512, 512])),
    'conv5_3': tf.Variable(tf.random_normal([7, 7, 512, 512])),

    'conv6_1': tf.Variable(tf.random_normal([7, 7, 512, 512])),
    'conv6_2': tf.Variable(tf.random_normal([7, 7, 512, 512])),
    'conv6_3': tf.Variable(tf.random_normal([7, 7, 512, 512])),

    'conv7_1': tf.Variable(tf.random_normal([7, 7, 512, 256])),
    'conv7_2': tf.Variable(tf.random_normal([7, 7, 256, 256])),
    'conv7_3': tf.Variable(tf.random_normal([7, 7, 256, 256])),

    'conv8_1': tf.Variable(tf.random_normal([7, 7, 256, 128])),
    'conv8_2': tf.Variable(tf.random_normal([7, 7, 128, 128])),
    'conv8_3': tf.Variable(tf.random_normal([7, 7, 128, 2]))
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

X, Y = read_frames('./resources', 'folder', 1)

logits = network(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimiser.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # load_path = saver.restore(sess, model_path)
    # print("Restored from: " + str(load_path))

    tf.train.start_queue_runners()

    for step in range(1, num_steps + 1):
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss = " + "{:.4f}".format(loss) + ", Training Accuracy = " + "{:.3f}".format(acc))

    print("Optimisation finished")

    print("Testing accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256], keep_prob: 1.0}))

    save_path = saver.save(sess, model_path)
    print("Saved at: " + str(save_path))
