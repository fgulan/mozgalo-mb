import argparse

import onnx
import os
import torch
from model import SqueezeModelSoftmax
from onnx_coreml import convert
from torch.autograd import Variable


def get_class_lables(num_classes):
    if num_classes == 26:
        return ['Albertsons', 'BJs', 'CVSPharmacy', 'Costco',
                'FredMeyer', 'Frys', 'HEB', 'HarrisTeeter',
                'HyVee', 'JewelOsco', 'KingSoopers', 'Kroger',
                'Meijer', 'Other', 'Publix', 'Safeway',
                'SamsClub', 'ShopRite', 'Smiths', 'StopShop',
                'Target', 'Walgreens', 'Walmart', 'Wegmans',
                'WholeFoodsMarket', 'WinCoFoods']
    elif num_classes == 2:
        return ['Other', 'Receipt']
    else:
        return ['Albertsons', 'BJs', 'CVSPharmacy', 'Costco',
                'FredMeyer', 'Frys', 'HEB', 'HarrisTeeter',
                'HyVee', 'JewelOsco', 'KingSoopers', 'Kroger',
                'Meijer', 'Publix', 'Safeway',
                'SamsClub', 'ShopRite', 'Smiths', 'StopShop',
                'Target', 'Walgreens', 'Walmart', 'Wegmans',
                'WholeFoodsMarket', 'WinCoFoods']


def export(args):
    """
    Exports the model for the Apple CoreML engine.
    :param args: Command line arguments from the argparse
    :return:
    """
    model = SqueezeModelSoftmax(num_classes=args.num_classes)

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    dummy_input = Variable(torch.FloatTensor(1, 3, args.height, args.width))
    output_path = os.path.join(args.save_dir, args.name + ".proto")
    torch.onnx.export(model, dummy_input, output_path, verbose=True)

    proto_model = onnx.load(output_path)
    coreml_model = convert(proto_model, 'classifier',
                           class_labels=get_class_lables(args.num_classes))
    coreml_model.save(os.path.join(args.save_dir, args.name + ".mlmodel"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', help="Number of classes",
                        required=True, type=int)
    parser.add_argument('--height', help="Input height",
                        required=True, type=int)
    parser.add_argument('--width', help="Input width",
                        required=True, type=int)
    parser.add_argument('--save-dir', help="Model saving folder",
                        required=True, type=str)
    parser.add_argument('--name', help="Model name prefix used for saving",
                        required=True, type=str)
    parser.add_argument('--weights', help="Path to the trained model file",
                        required=True, type=str)
    args = parser.parse_args()
    export(args)


if __name__ == '__main__':
    main()
