import argparse
import pickle
import sys

import numpy as np
import os
import torch
from PIL import Image
from PIL import ImageFile
from model import SqueezeModelSoftmax
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data import crop_upper_part

sys.path.insert(0, '..')

ImageFile.LOAD_TRUNCATED_IMAGES = True
NUM_CLASSES = 26


class TestDataset(Dataset):
    """Store recipts test dataset."""

    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = list_input_images(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx][0])
        image = open_image(image_path)
        image = self.transform(image)
        return image


def get_class_labels():
    class_labels = {
        'Albertsons': 0,
        'BJs': 1,
        'CVSPharmacy': 2,
        'Costco': 3,
        'FredMeyer': 4,
        'Frys': 5,
        'HEB': 6,
        'HarrisTeeter': 7,
        'HyVee': 8,
        'JewelOsco': 9,
        'KingSoopers': 10,
        'Kroger': 11,
        'Meijer': 12,
        'Other': 13,
        'Publix': 14,
        'Safeway': 15,
        'SamsClub': 16,
        'ShopRite': 17,
        'Smiths': 18,
        'StopShop': 19,
        'Target': 20,
        'Walgreens': 21,
        'Walmart': 22,
        'Wegmans': 23,
        'WholeFoodsMarket': 24,
        'WinCoFoods': 25}

    return class_labels


def get_class_dict():
    class_labels = get_class_labels()
    class_dict = {v: k for k, v in class_labels.items()}

    return class_dict


def var(tensor, use_gpu):
    if use_gpu:
        tensor = tensor.cuda(0)
    return tensor


def data_preprocess_transformations(input_shape, crop_perc=0.5):
    """Preprocess object for transforming image to model input
    Args:
        input_shape: model input shape (channels x height x width)
        crop_perc: percent of how much image would be cropped from

    Returns:
        Composite of transforms objects.
    """

    num_channels, height, width = input_shape

    return transforms.Compose([
        transforms.Lambda(lambda x: crop_upper_part(np.array(x), crop_perc)),
        transforms.ToPILImage(),
        transforms.Grayscale(num_channels),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])


def list_input_images(images_folder):
    """
    Args:
        images_folder: Folder with input images with name template:
        <int_id>.jpg

    Returns:
        List of tuples: (image_file_name, image_int_id)
    """
    files = os.listdir(images_folder)
    images = []

    for file in files:

        name_components = file.split(".")
        extension = name_components[1].lower()

        if extension.lower() == 'jpg' or extension.lower() == 'jpeg':
            image_id = int(name_components[0])
            images.append((file, image_id))

    return sorted(images, key=lambda x: x[1])


def open_image(image_path):
    """
    Args:
        image_path: Path to an image.

    Returns:
        PIL Image in RGB format.
    """
    with open(image_path, 'rb') as f:
        image = Image.open(f).convert("RGB")
    return image


def predicted_store(prediction, class_dict):
    """
    Args:
        prediction: Model probability output.

    Returns:
        Most probable store (argmax)
    """
    class_indice = np.argmax(prediction)
    return class_dict[class_indice]


def print_predicted_store_stats(predicted_stores):
    unique, counts = np.unique(predicted_stores, return_counts=True)
    stats_dict = dict(zip(unique, counts))
    for store, count in stats_dict.items():
        print("{:<16} => {}".format(store, count))


def threshold_heuristic(predictions, thr_dict):
    """
    Per-class threshold heuristic
    """
    print("Doing tresholded predictions")

    new_preds = []
    class_dict = get_class_dict()
    class_labels = get_class_labels()

    for act in predictions:
        pred_label = np.argmax(act)
        if class_dict[pred_label] != "Other":
            if np.max(act) < max(thr_dict[class_dict[pred_label]], 0.5):
                # Replace as if the Other is predicted because it
                # doesn't satsify the threshold
                other_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
                other_vector[class_labels["Other"]] = 1

                new_preds.append(other_vector)
            else:
                new_preds.append(act)
        else:
            new_preds.append(act)

    return np.array(new_preds)


def identity_heuristic(predictions):
    return predictions


def process_predictions(predictions, thr_path):
    if thr_path is None:
        return predictions

    with open(thr_path, "rb") as f:
        thr_dict = pickle.load(f)
        return threshold_heuristic(predictions, thr_dict)


def write_to_csv(predicted_stores, csv_path):
    with open(csv_path, "w") as f:
        for i, store in enumerate(predicted_stores):
            f.write(store)
            if i < len(predicted_stores) - 1:
                f.write("\n")


def evaluate(args):
    model = SqueezeModelSoftmax(num_classes=NUM_CLASSES)
    model_state_dict = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state_dict)
    model.eval()
    torch.set_num_threads(args.num_threads)

    if args.use_gpu:
        model.cuda(0)

    input_shape = (args.num_channels, args.height, args.width)
    predictions = []
    preprocess_transformations = data_preprocess_transformations(input_shape)
    test_set = TestDataset(args.dataset, preprocess_transformations)
    loader = DataLoader(test_set,
                        batch_size=args.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=args.num_threads)

    num_batches = len(loader)
    for batch_index, test_batch in enumerate(loader):
        batch_input_tensors = var(test_batch, args.use_gpu)

        batch_predictions = model(batch_input_tensors).cpu().data.numpy()
        predictions.extend(batch_predictions)

        print('Batch {}/{}'.format(batch_index + 1,
                                   num_batches), end="\r", flush=True)

    predictions = np.array(predictions)
    predictions = process_predictions(predictions, args.thrs_path)

    class_dict = get_class_dict()
    predicted_stores = [predicted_store(
        prediction, class_dict) for prediction in predictions]

    write_to_csv(predicted_stores, args.csv_path)
    print_predicted_store_stats(predicted_stores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use-gpu', help="GPU use flag", action='store_true', required=False)
    parser.add_argument(
        '--csv-path', help="Output CSV saving path",
        default='eval_results.csv')
    parser.add_argument(
        '--model', help="Path to the trained model file", required=True)
    parser.add_argument(
        '--thrs-path', help="Path to the thresholds file", default=None)
    parser.add_argument('--dataset', help="Path to the dataset",
                        default='../data/test_dataset', required=True)
    parser.add_argument(
        '--num-threads', help="Number of threads to use", default=8, type=int)
    parser.add_argument('--batch-size', help="Batch size",
                        default=64, type=int)
    parser.add_argument('--num-channels', default=3, type=int)
    parser.add_argument('--height', default=370, type=int)
    parser.add_argument('--width', default=400, type=int)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
