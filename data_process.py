import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess_image(img_path):
    im = tf.image.decode_image(tf.io.read_file(img_path))
    if im.shape[-1] == 4:
        im = im[..., :-1]
    print(img_path, im.shape)
    im = tf.image.resize(im, [1754, 1239])
    im = tf.cast(im, tf.float32)
    return tf.expand_dims(im, 0)


def add_mask(image, mask):
    bool_mask = tf.reduce_sum(tf.sqrt(1/3) * mask, -1, keepdims=1) < 420    # out of 441.673
    bool_image = tf.reduce_sum(tf.sqrt(1/3) * image, -1, keepdims=1) < 420
    average = tf.logical_and(bool_mask, bool_image)

    # remove "average" pixels from the rest of the masks
    bool_mask = tf.logical_and(tf.logical_not(average), bool_mask)
    bool_image = tf.logical_and(tf.logical_not(average), bool_image)

    one = tf.cast(bool_mask, tf.float32) * mask
    two = tf.cast(tf.logical_not(bool_mask), tf.float32) * image
    three = (-image + mask) / 2 * tf.cast(average, tf.float32)

    image = one + two + three
    return image


def sample_train_example(marked, unmarked, sample_size):
    input = tf.squeeze(marked).numpy()
    output = tf.squeeze(unmarked).numpy()
    shape = input.shape
    if sample_size > min(shape[0], shape[1]):
        raise ValueError
    initial_height = int(np.random.random() * min(sample_size, shape[0] - sample_size))
    initial_width = int(np.random.random() * min(sample_size, shape[1] - sample_size))
    store_width = initial_width
    samples_in = np.zeros((0,sample_size, sample_size, 3))
    samples_out = np.zeros((0,sample_size, sample_size, 3))
    while initial_height + sample_size <= shape[0]:
        initial_width = store_width
        while initial_width + sample_size <= shape[1]:
            samples_in = np.append(samples_in,
                                [input[initial_height:initial_height + sample_size,
                                       initial_width:initial_width + sample_size,:]], axis=0)
            samples_out = np.append(samples_out,
                                [output[initial_height:initial_height + sample_size,
                                        initial_width:initial_width + sample_size,:]], axis=0)

            initial_width += sample_size
        initial_height += sample_size
    return tf.convert_to_tensor(samples_in), tf.convert_to_tensor(samples_out)


def create_samples_tensor(raw_image_path, examples, sample_size=128):
    input_samples = 0
    output_samples = 0
    for i in examples:
        unmarked = preprocess_image(raw_image_path + i[1])
        marked = preprocess_image(raw_image_path + i[0])
        marked, unmarked = sample_train_example(marked, unmarked, sample_size=sample_size)
        if isinstance(input_samples, tf.Tensor):

            input_samples = tf.concat([input_samples, marked], axis=0)
            output_samples = tf.concat([output_samples, unmarked], axis=0)

        else:
            input_samples = marked
            output_samples = unmarked

    return input_samples / 255, output_samples / 255


def save_image(image, filename):
    image = tf.squeeze(image)
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
    """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


