import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random

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


def random_erase(value, prob):
    """
    Performs random erasing augmentation technique.
    https://arxiv.org/pdf/1708.04896.pdf
    """
    if random.random() > prob:
        return value
        
    h, w, _ = value.shape

    r_width = np.random.randint(20, w - 20)
    r_height = np.random.randint(20, h - 20)

    top_x = np.random.randint(0, w - r_width)
    top_y = np.random.randint(0, h - r_height)

    value[top_y:r_height + top_y, top_x:top_x + r_width, :] = np.mean(value)

    return value


class BinaryDataset(Dataset):
    """
    Binary dataset where classes are Other or Non-Other.
    """
    def __init__(self, images_dir, other_folder_name="Other", transform=None):
        positive_folders = [os.path.join(images_dir, folder) for folder in os.listdir(
            images_dir) if folder != other_folder_name]
        negative_folder = os.path.join(images_dir, other_folder_name)

        self.pos_images = []
        for pos_f in positive_folders:
            self.pos_images.extend([os.path.join(pos_f, img) for img in
                                    os.listdir(pos_f) if not img.startswith(".")])

        self.neg_images = [os.path.join(negative_folder, img) for img in
                           os.listdir(negative_folder) if not img.startswith(".")]

        self.im_paths = self.pos_images + self.neg_images
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        img_name = self.im_paths[idx]

        with open(img_name, 'rb') as f:
            image = Image.open(f).convert("RGB")

        if img_name in self.pos_images:
            label = np.array([1], dtype=np.float32)
        else:
            label = np.array([0], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
