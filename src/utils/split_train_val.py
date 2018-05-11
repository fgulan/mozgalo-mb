"""
Splits the original dataset into training and validation datasets.
"""

import argparse
import os
import random as rn
from shutil import copyfile
from shutil import rmtree

import numpy as np


def main(origin_folder, export_folder, val_size=0.2):
    assert 0 <= val_size <= 1

    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    else:
        rmtree(export_folder)

    os.makedirs(os.path.join(export_folder, "train"))
    os.makedirs(os.path.join(export_folder, "validation"))

    for dir_name in sorted(os.listdir(origin_folder)):
        dir_path = os.path.join(origin_folder, dir_name)

        if not os.path.isdir(dir_path):
            continue

        print("Now processing {} directory...".format(dir_name))

        # Sort and shuffle for consistency
        class_files = sorted(os.listdir(dir_path))
        np.random.shuffle(class_files)

        edge_index = round(val_size * len(class_files))
        train_files = class_files[edge_index:]
        val_files = class_files[:edge_index]

        # Create class directory copy for train and val
        os.makedirs(os.path.join(export_folder, "train", dir_name))
        os.makedirs(os.path.join(export_folder, "validation", dir_name))

        for img in train_files:
            img_path = os.path.join(dir_path, img)
            end_path = os.path.join(export_folder, "train", dir_name, img)
            copyfile(img_path, end_path)

        for img in val_files:
            img_path = os.path.join(dir_path, img)
            end_path = os.path.join(export_folder, "validation", dir_name, img)
            copyfile(img_path, end_path)


if __name__ == "__main__":
    np.random.seed(1337)
    rn.seed(1234)

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--origin', type=str,
                        help='Original data folder path.')
    parser.add_argument('-e', '--export', type=str,
                        help='Folder where the new data will be created.')
    parser.add_argument('-v', '--val_size', type=float, default=0.2,
                        help='Validation dataset size. Must be in the [0-1] range.')
    args = parser.parse_args()

    main(origin_folder=args.origin,
         export_folder=args.export, val_size=args.val_size)
