import argparse
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision import transforms

from data import random_erase, crop_upper_part, HingeDataset
from model import HingeModel

DATASET_ROOT_PATH = '../data/mozgalo_split'
CPU_CORES = 8
BATCH_SIZE = 4
NUM_CLASSES = 25
LEARNING_RATE = 0.001


def data_transformations(input_shape):
    """

    :param input_shape:
    :return:
    """
    crop_perc = 0.5

    train_trans = transforms.Compose([
        transforms.Lambda(lambda x: crop_upper_part(np.array(x, dtype=np.uint8), crop_perc)),
        transforms.ToPILImage(),
        # Requires the master branch of the torchvision package
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.4, 1.2)),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
        transforms.Grayscale(3),
        transforms.Lambda(lambda x: random_erase(np.array(x, dtype=np.uint8))),
        transforms.ToTensor()
    ])
    val_trans = transforms.Compose([
        transforms.Lambda(lambda x: crop_upper_part(np.array(x, dtype=np.uint8), crop_perc)),
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor()
    ])
    return train_trans, val_trans


def train(args):
    model = HingeModel(fine_tune=args.fine_tune)

    if args.model:
        model.load_state_dict(torch.load(args.model))
        print("Loaded model from:", args.model)

    train_transform, val_transform = data_transformations(model.model.input_size)

    # Train dataset
    train_dataset = HingeDataset(images_dir=os.path.join(DATASET_ROOT_PATH, 'train'),
                                 transform=train_transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=CPU_CORES,
                                                       pin_memory=True)

    # Validation dataset
    validation_dataset = HingeDataset(images_dir=os.path.join(DATASET_ROOT_PATH, 'validation'),
                                      transform=val_transform)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            num_workers=CPU_CORES)

    if args.gpu > -1:
        model.cuda(args.gpu)
    criterion = nn.SoftMarginLoss()

    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=optim_params, lr=0.001, momentum=0.9,
                              weight_decay=0.0005)
        min_lr = 0.00001
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(optim_params, lr=LEARNING_RATE, weight_decay=0.0005)
        min_lr = 0.00001
    else:
        raise ValueError('Unknown optimizer')

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.1, patience=2, verbose=True,
        min_lr=min_lr)

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

    def var(tensor, volatile=False):
        if args.gpu > -1:
            tensor = tensor.cuda(args.gpu)
        return Variable(tensor, volatile=volatile)

    global_step = 0

    def train_epoch():
        nonlocal global_step
        batch_count = len(train_dataset_loader)
        loss_sum = num_correct = 0
        model.train()
        for i, train_batch in enumerate(train_dataset_loader):
            train_x, train_y = var(train_batch[0]), var(train_batch[1])
            logit = model(input=train_x, target=train_y)
            y_pred = logit.sign()
            loss = criterion(input=logit, target=train_y)
            loss_sum += loss.data[0]

            correct = y_pred.eq(train_y).long().sum().data[0]
            num_correct += correct / len(train_y)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()
            global_step += 1

            avg_loss = loss_sum / (i + 1)
            avg_acc = num_correct / (i + 1)

            print('batch {}/{} | loss = {:.5f} | accuracy = {:.5f}'.format(i, batch_count,
                                                                           avg_loss, avg_acc),
                  end="\r", flush=True)

            summary_writer.add_scalar(
                tag='train_loss', scalar_value=loss.data[0],
                global_step=global_step)

    def validate():
        model.eval()
        loss_sum = num_correct = denom = 0
        for valid_batch in validation_dataset_loader:
            valid_x, valid_y = (var(valid_batch[0], volatile=True),
                                var(valid_batch[1], volatile=True))
            logit = model(valid_x)
            y_pred = logit.sign()
            loss = criterion(input=logit, target=valid_y)
            loss_sum += loss.data[0] * valid_x.size(0)
            num_correct += y_pred.eq(valid_y).long().sum().data[0]
            denom += valid_x.size(0)
        loss = loss_sum / denom
        accuracy = num_correct / denom
        summary_writer.add_scalar(tag='valid_loss', scalar_value=loss,
                                  global_step=global_step)
        summary_writer.add_scalar(tag='valid_accuracy', scalar_value=accuracy,
                                  global_step=global_step)
        lr_scheduler.step(accuracy)
        return loss, accuracy

    for epoch in range(1, args.max_epoch + 1):
        train_epoch()
        valid_loss, valid_accuracy = validate()

        if epoch == 1:
            best_valid_loss = valid_loss

        print('Epoch {}: Valid loss = {:.5f}'.format(epoch, valid_loss))
        print('Epoch {}: Valid accuracy = {:.5f}'.format(epoch, valid_accuracy))

        if valid_loss <= best_valid_loss:
            model_filename = ('epoch_{:02d}'
                              '-valLoss_{:.5f}'
                              '-valAcc_{:.5f}'.format(epoch, valid_loss, valid_accuracy))
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print('Epoch {}: Saved the new best model to: {}'.format(epoch, model_path))
            best_valid_loss = valid_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', help="Margin parameter from the LSoftmax paper", default=1,
                        type=int)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--max-epoch', default=50, type=int)
    parser.add_argument('--fine-tune', dest="fine_tune",
                        help="If true then the whole network is trained, otherwise only the top",
                        action="store_true")
    parser.add_argument('--gpu', help="GPU use flag. 0 will use, -1 will not.", default=0,
                        type=int)
    parser.add_argument('--save-dir', help="Model saving folder", required=True)
    parser.add_argument('--model', help="Path to the trained model file", default=None,
                        required=False)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
