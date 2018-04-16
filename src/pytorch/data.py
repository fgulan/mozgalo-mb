import numpy as np


def normalize(image):
    """
    Normalizes image values to [-0.5, 0.5] range
    :param image: Numpy array image
    :return: Normalized image (numpy array)
    """
    return image / 255. - 0.5


def crop_upper_part(image, percent=0.4):
    """
    Crops the upper part of an image
    :param image: Image (numpy array)
    :param percent: Percentage of an upper part to preserve
    :return: Cropped numpy image
    """
    height, _, _ = image.shape
    point = int(percent * height)
    return image[:point]


def random_erase(value):
    """
    Performs random erasing augmentation technique.
    https://arxiv.org/pdf/1708.04896.pdf
    """
    h, w, _ = value.shape

    r_width = np.random.randint(20, w - 20)
    r_height = np.random.randint(20, h - 20)

    top_x = np.random.randint(0, w - r_width)
    top_y = np.random.randint(0, h - r_height)

    value[top_y:r_height + top_y, top_x:top_x + r_width, :] = np.mean(value)

    return value
