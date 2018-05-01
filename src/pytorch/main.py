import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import datetime
from data import random_erase, crop_upper_part
from model import SqueezeModel
from center_loss import CenterLoss
from utils import AverageMeter

DATASET_ROOT_PATH = '/home/gulan_filip/dataset'
CPU_CORES = 8
BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
INPUT_SHAPE = (3, 370, 400) # C x H x W
CENTER_LOSS_WEIGHT = 0.003
CENTER_LOSS_LR = 1e-3

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
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.4, 1.4)),
        transforms.RandomHorizontalFlip(),
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
    model = SqueezeModel(num_classes=NUM_CLASSES, fine_tune=args.fine_tune)

    if args.model:
        model.load_state_dict(torch.load(args.model))
        print("Loaded model from:", args.model)

    train_transform, val_transform = data_transformations(INPUT_SHAPE)

    # Train dataset
    #train_dataset = BinaryDataset(images_dir=os.path.join(DATASET_ROOT_PATH, 'train'),
    #                               transform=train_transform)
    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_PATH, 'train'),
                                         transform=train_transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=CPU_CORES,
                                                       pin_memory=True)

    # Validation dataset
    #validation_dataset = BinaryDataset(images_dir=os.path.join(DATASET_ROOT_PATH, 'validation'),
    #                                    transform=val_transform)
    validation_dataset = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_PATH, 'validation'),
                                              transform=val_transform)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            num_workers=CPU_CORES)
    use_gpu = False
    if args.gpu > -1:
        use_gpu = True
        model.cuda(args.gpu)

    # Define Losses
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=NUM_CLASSES, feat_dim=model.num_features, use_gpu=use_gpu)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=CENTER_LOSS_LR, weight_decay=0.0005)

    min_lr = 0.000001
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'sgd':
        optimizer_model = optim.SGD(params=optim_params, lr=LEARNING_RATE, momentum=0.9,
                              weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_model = optim.Adam(optim_params, lr=LEARNING_RATE, weight_decay=0.0005)
    else:
        raise ValueError('Unknown optimizer')

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_model, mode='max', factor=0.1, patience=3, verbose=True,
        min_lr=min_lr)

    cent_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_centloss, mode='max', factor=0.1, patience=3, verbose=True,
        min_lr=min_lr)


    time_folder = str(datetime.datetime.now())
    summary_writer = SummaryWriter(os.path.join("../logs", time_folder))

    def var(tensor):
        if args.gpu > -1:
            tensor = tensor.cuda(args.gpu)
        return tensor

    global_step = 0

    def train_epoch():
        nonlocal global_step
        batch_count = len(train_dataset_loader)
        num_correct = 0
        model.train()

        xent_losses = AverageMeter()
        cent_losses = AverageMeter()
        losses = AverageMeter()

        for i, train_batch in enumerate(train_dataset_loader):
            train_x, train_y = var(train_batch[0]), var(train_batch[1])
            sample_count = len(train_y)
            logit, features = model(input=train_x, target=train_y)
            y_pred = logit.max(1)[1]

            loss_xent = criterion_xent(input=logit, target=train_y)
            loss_cent = criterion_cent(features, train_y)
            loss_cent *= CENTER_LOSS_WEIGHT
            loss = loss_xent + loss_cent

            correct = y_pred.eq(train_y).long().sum().item()
            num_correct += correct

            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer_model.step()

            for param in criterion_cent.parameters():
                param.grad.data *= (1. / CENTER_LOSS_WEIGHT)
            optimizer_centloss.step()

            global_step += 1

            avg_acc = float(num_correct) / (float(i + 1) * sample_count)
            losses.update(loss.item(), sample_count)
            xent_losses.update(loss_xent.item(), sample_count)
            cent_losses.update(loss_cent.item(), sample_count)

            print('batch {}/{} | Loss {:.6f} CEntLoss {:.6f} CenterLoss {:.6f} | accuracy = {:.6f}'
                  .format(i + 1, batch_count, losses.avg, xent_losses.avg, cent_losses.avg, avg_acc),
                  end="\r", flush=True)

            if i % 50 == 0:
                # Write to Tensorboard
                summary_writer.add_scalar(tag='train_loss_total', scalar_value=losses.avg,
                                          global_step=global_step)
                summary_writer.add_scalar(tag='train_CE_Loss', scalar_value=xent_losses.avg,
                                          global_step=global_step)
                summary_writer.add_scalar(tag='train_center_Loss', scalar_value=cent_losses.avg,
                                          global_step=global_step)
                summary_writer.add_scalar(tag='train_accuracy', scalar_value=avg_acc,
                                          global_step=global_step)


    def validate():
        model.eval()
        num_correct = denom = 0.0
        predicted, gt = [], []

        losses = AverageMeter()
        xent_losses = AverageMeter()
        cent_losses = AverageMeter()

        print("Starting validation")
        for valid_batch in validation_dataset_loader:
            valid_x, valid_y = (var(valid_batch[0]),
                                var(valid_batch[1]))
            logit, features = model(valid_x)
            y_pred = logit.max(1)[1]

            # Batch size for averaging
            sample_count = len(valid_y)

            # Calculate losses
            loss_xent = criterion_xent(input=logit, target=valid_y)
            loss_cent = criterion_cent(features, valid_y) * CENTER_LOSS_WEIGHT
            loss = loss_xent + loss_cent

            # Update losses
            losses.update(loss.item(), sample_count)
            xent_losses.update(loss_xent.item(), sample_count)
            cent_losses.update(loss_cent.item(), sample_count)

            # Count predictions
            for act in y_pred.cpu().data.numpy():
                predicted.append(act)
            gt.extend(valid_y.cpu().data.numpy())
            num_correct += float(y_pred.eq(valid_y).long().sum().item())
            denom += float(sample_count)

        accuracy = float(num_correct) / float(denom)

        # Write to Tensorboard
        summary_writer.add_scalar(tag='valid_loss_total', scalar_value=losses.avg,
                                  global_step=global_step)
        summary_writer.add_scalar(tag='valid_CE_Loss', scalar_value=xent_losses.avg,
                                  global_step=global_step)
        summary_writer.add_scalar(tag='valid_center_Loss', scalar_value=cent_losses.avg,
                                  global_step=global_step)
        summary_writer.add_scalar(tag='valid_accuracy', scalar_value=accuracy,
                                  global_step=global_step)

        gt = np.array(gt).flatten()
        predicted = np.array(predicted)

        f1 = f1_score(gt, predicted, average='macro')
        prec = precision_score(gt, predicted, average='macro')
        rec = recall_score(gt, predicted, average='macro')

        lr_scheduler.step(1. / cent_losses.avg)
        cent_lr_scheduler.step(1. / xent_losses.avg)

        return losses.avg, accuracy, f1, prec, rec

    for epoch in range(1, args.max_epoch + 1):
        train_epoch()
        valid_loss, valid_accuracy, f1_val, prec_val, rec_val = validate()

        if epoch == 1:
            best_f1_val = f1_val

        print('\nEpoch {}: Total Valid Loss = {:.5f}'.format(epoch, valid_loss))
        print('Epoch {}: Valid accuracy = {:.5f}'.format(epoch, valid_accuracy))
        print('Epoch {}: Valid f1 = {:.5f}'.format(epoch, f1_val))
        print('Epoch {}: Valid precision = {:.5f}'.format(epoch, prec_val))
        print('Epoch {}: Valid recall = {:.5f}\n'.format(epoch, rec_val))

        if f1_val >= best_f1_val:
            model_filename = (args.name + '_epoch_{:02d}'
                                          '-valLoss_{:.5f}'
                                          '-valF1_{:.5f}'.format(epoch, valid_loss,
                                                                  f1_val))
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print('Epoch {}: Saved the new best model to: {}'.format(epoch, model_path))
            best_f1_val = f1_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--max-epoch', default=15, type=int)
    parser.add_argument('--fine-tune', dest="fine_tune",
                        help="If true then the whole network is trained, otherwise only the top",
                        action="store_true")
    parser.add_argument('--gpu', help="GPU use flag. 0 will use, -1 will not.", default=0,
                        type=int)
    parser.add_argument('--save-dir', help="Model saving folder", required=True)
    parser.add_argument('--name', help="Model name prefix used for saving",
                        required=True, type=str)
    parser.add_argument('--model', help="Path to the trained model file", default=None,
                        required=False)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
