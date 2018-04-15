import numpy as np

def normalize(image):
    image /= 255.
    image -= 0.5

    return image


def crop_upper_part(image, percent=0.4):
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

def to_grayscale(image):
    if len(image.shape) == 3:
        # RGB Image, convert to grayscale
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

    # Grayscale image, expand to RGB by copying channels
    image = np.stack((image,) * 3, axis=-1)
    return image.astype(np.uint8)
