import argparse
import os

import torch
from center_loss import CenterLoss
from focal_loss import FocalLoss
from model import SqueezeModel
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from train import data_transformations, train_epoch, evaluate


def print_eval_info(eval_info, epoch):
    print("\n")
    print('Epoch {}: Total Valid Loss = {:.5f}'.format(
        epoch, eval_info['total_loss']))
    print('Epoch {}: Valid accuracy = {:.5f}'.format(
        epoch, eval_info['accuracy']))
    print('Epoch {}: Valid f1 = {:.5f}'.format(epoch, eval_info['f1']))
    print('Epoch {}: Valid precision = {:.5f}'.format(
        epoch, eval_info['precision']))
    print('Epoch {}: Valid recall = {:.5f}\n'.format(
        epoch, eval_info['recall']))


def train(args):
    # model
    model = SqueezeModel(num_classes=args.num_classes)

    if args.model:
        model.load_state_dict(torch.load(args.model))
        print("Loaded model from:", args.model)

    use_gpu = False
    if args.gpu > -1:
        use_gpu = True
        model.cuda(args.gpu)

    # dataset
    input_shape = (args.num_channels, args.height, args.width)
    train_transform, val_transform = data_transformations(input_shape)

    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform=train_transform)
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_threads,
                                      pin_memory=True)

    validation_dataset = ImageFolder(root=os.path.join(args.dataset, 'validation'),
                                     transform=val_transform)
    validation_dataset_loader = DataLoader(validation_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_threads,
                                           pin_memory=True)

    # losses
    # model_criterion = CrossEntropyLoss()
    model_criterion = FocalLoss(class_num=args.num_classes, gamma=-0.5)
    center_criterion = CenterLoss(num_classes=args.num_classes,
                                  feat_dim=model.num_features,
                                  use_gpu=use_gpu)

    # optimizers
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'sgd':
        model_optimizer = SGD(params=optim_params, lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        model_optimizer = Adam(optim_params, lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer')

    center_optimizer = Adam(center_criterion.parameters(), lr=args.center_learning_rate,
                            weight_decay=args.weight_decay)

    # schedulers
    model_lr_scheduler = ReduceLROnPlateau(
        model_optimizer, factor=0.25, patience=5, verbose=True)
    center_lr_scheduler = ReduceLROnPlateau(
        center_optimizer, factor=0.25, patience=5, verbose=True)

    for epoch in range(1, args.max_epoch + 1):
        _ = train_epoch(train_dataset_loader,
                        model, model_criterion, center_criterion,
                        model_optimizer, center_optimizer, use_gpu)

        eval_info = evaluate(validation_dataset_loader, model,
                             model_criterion, center_criterion, use_gpu)

        model_lr_scheduler.step(eval_info['model_loss'])
        center_lr_scheduler.step(eval_info['center_loss'])

        print_eval_info(eval_info, epoch)

        if epoch == 1:
            best_f1_val = eval_info['f1']

        if eval_info['f1'] >= best_f1_val:
            model_filename = (args.name + '_epoch_{:02d}'
                                          '-valLoss_{:.5f}'
                                          '-valF1_{:.5f}'.format(epoch,
                                                                 eval_info['total_loss'],
                                                                 eval_info['f1']))
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print('Epoch {}: Saved the new best model to: {}'.format(
                epoch, model_path))
            best_f1_val = eval_info['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--center-learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--max-epoch', default=60, type=int)
    parser.add_argument(
        '--gpu', help="GPU use flag. 0 will use, -1 will not.", default=0, type=int)
    parser.add_argument(
        '--save-dir', help="Model saving folder", default='./models', required=True)
    parser.add_argument(
        '--name', help="Model name prefix used for saving", required=True, type=str)
    parser.add_argument(
        '--model', help="Path to the trained model file", default=None, required=False)
    parser.add_argument('--dataset', help="Path to the dataset",
                        default='../data/dataset', required=True)
    parser.add_argument(
        '--num-classes', help="Number of classes", default=26, type=int)
    parser.add_argument(
        '--num-threads', help="Number of threads to use", default=4, type=int)
    parser.add_argument('--batch-size', help="Batch size",
                        default=64, type=int)
    parser.add_argument('--num-channels', default=3, type=int)
    parser.add_argument('--height', default=370, type=int)
    parser.add_argument('--width', default=400, type=int)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
