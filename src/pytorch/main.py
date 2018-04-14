import argparse
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data import random_erase, crop_upper_part, normalize
from model import LModel

DATASET_ROOT_PATH = '/home/gulan_filip/mb-dataset/'
CPU_CORES = 4
BATCH_SIZE = 32
NUM_CLASSES = 25

def data_transformations(model, input_shape):
    train_trans = transforms.Compose([
        #transforms.Lambda(lambda x: crop_upper_part(np.array(x), 0.5)),
        transforms.Resize((input_shape[1], input_shape[2])),
        # transforms.RandomResizedCrop((input_shape[1], input_shape[2])),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Lambda(lambda x: random_erase(np.array(x, dtype=np.float32))),
        transforms.Lambda(lambda x: normalize(x)),
        transforms.ToTensor(),
    ])
    val_trans = transforms.Compose([
        #transforms.Lambda(lambda x: crop_upper_part(np.array(x, dtype=np.float32), 0.5)),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.Lambda(lambda x: normalize(np.array(x, dtype=np.float32))),
        transforms.ToTensor(),
    ])
    return train_trans, val_trans


def train(args):
    model = LModel(margin=args.margin, num_classes=NUM_CLASSES)

    train_transform, val_transform = data_transformations(model, model.model.input_size)

    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_PATH, 'train'),
                                         transform=train_transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=CPU_CORES)

    validation_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_ROOT_PATH, 'validation'),
        transform=val_transform)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=False,
                                                            num_workers=CPU_CORES)

    if args.gpu > -1:
        model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=0.0005)
        min_lr = 0.001
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)
        min_lr = 0.00001
    else:
        raise ValueError('Unknown optimizer')

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.1, patience=5, verbose=True,
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
        model.train()
        for i, train_batch in enumerate(train_dataset_loader):
            train_x, train_y = var(train_batch[0]), var(train_batch[1])
            logit = model(input=train_x, target=train_y)
            loss = criterion(input=logit, target=train_y)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()
            global_step += 1

            print('Batch: {0}/{1}, batch loss: {2}'.format(i, batch_count, loss.data[0]),
                  end='\r', flush=True)

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
            y_pred = logit.max(1)[1]
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

    """
    def test():
        model.eval()
        num_correct = denom = 0
        for test_batch in test_loader:
            test_x, test_y = (var(test_batch[0], volatile=True),
                              var(test_batch[1], volatile=True))
            logit = model(test_x)
            y_pred = logit.max(1)[1]
            num_correct += y_pred.eq(test_y).long().sum().data[0]
            denom += test_x.size(0)
        accuracy = num_correct / denom
        summary_writer.add_scalar(tag='test_accuracy', scalar_value=accuracy,
                                  global_step=global_step)
        return accuracy
    """

    best_valid_accuracy = 0
    for epoch in range(1, args.max_epoch + 1):
        train_epoch()
        valid_loss, valid_accuracy = validate()
        print(f'Epoch {epoch}: Valid loss = {valid_loss:.5f}')
        print(f'Epoch {epoch}: Valid accuracy = {valid_accuracy:.5f}')
        # test_accuracy = 0#test()
        # print(f'Epoch {epoch}: Test accuracy = {test_accuracy:.5f}')
        if valid_accuracy > best_valid_accuracy:
            model_filename = (f'{epoch:02d}'
                              f'-{valid_loss:.5f}'
                              f'-{valid_accuracy:.5f}')
            # f'-{test_accuracy:.5f}.pt')
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f'Epoch {epoch}: Saved the new best model to: {model_path}')
            best_valid_accuracy = valid_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', default=1, type=int)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--max-epoch', default=50, type=int)
    parser.add_argument('--fine-tune', default=True)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save-dir', required=True)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
