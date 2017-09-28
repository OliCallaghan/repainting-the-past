from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os

data_dir = "./bin"

f_in_x = 224
f_in_y = 224

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_frame(filename_queue):
    # File format
    # W x H x 3
    # LABLAB...LAB (float)
    # List in order of width, height
    class Frame(object):
        pass

    frame = Frame()
    frame.height = 224
    frame.width = 224
    frame.AB_bins = [22, 22]

    image_reader = tf.WholeFileReader()

    _, jpeg_data = image_reader.read(filename_queue)
    rgb_data = tf.image.decode_jpeg(jpeg_data, 3)

    lab_data = rgb_to_lab(tf.cast(rgb_data, tf.float32))
    # Extract L channel
    frame.L_channel = tf.slice(
        lab_data, [0, 0, 0], [frame.height, frame.width, 1]
    )

    # Extract AB channels
    frame.AB_channels = tf.image.resize_images(tf.slice(
        lab_data, [0, 0, 1], [frame.height, frame.width, 2]
    ), [28, 28])

    # Normalize data
    frame.L_channel = frame.L_channel / 50 - 1
    frame.AB_channels = frame.AB_channels / 110

    return (frame.L_channel, frame.AB_channels)


def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def gen_L_AB_batch(frame_L, frame_AB, min_queue_examples, batch_size, shuffle):
    num_process_threads = 16
    if shuffle:
        frames_L, frames_AB = tf.train.shuffle_batch(
            [frame_L, frame_AB],
            batch_size=batch_size,
            num_threads=num_process_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        frames_L, frames_AB = tf.train.batch(
            [frame_L, frame_AB],
            batch_size=batch_size,
            num_threads=num_process_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    tf.summary.image('images', frames_L)
    return frames_L, frames_AB


def inputs(eval_data, data_dir, batch_size):
    if not (eval_data):
        filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Missing file ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    L_channel, AB_channel = read_frame(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return gen_L_AB_batch(L_channel, AB_channel, min_queue_examples, batch_size, shuffle=False)


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image
