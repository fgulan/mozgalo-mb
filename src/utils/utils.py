"""
Utility Functions
"""
import os
import argparse
import random as rn

import numpy as np
import tensorflow as tf


def list_dir(dir_path, extension=None):
    """
    Creates a list of directory files. It is not recursive and it lists FILES ONLY.
    If the extension argument is given it will filter files that end with the given extension.

    :param extension: File extension filter
    :param dir_path: Directory path
    :return: List of absolute file paths in the given directory
    """
    files = [os.path.join(dir_path, p) for p in os.listdir(dir_path) if
             os.path.isfile(os.path.join(dir_path, p))]
    if extension:
        return list(filter(lambda x: x.endswith(extension), files))
    else:
        return files


def set_random_seeds():
    """
    Sets random seed for multiple modules according to:
    See: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    rn.seed(69)
    tf.set_random_seed(42)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')