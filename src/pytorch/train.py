import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms

from data import random_erase, crop_upper_part
from utils import AverageMeter


def train_data_transformations(input_shape, crop_perc=0.5):
    return transforms.Compose([
        transforms.Lambda(lambda x: crop_upper_part(
            np.array(x, dtype=np.uint8), crop_perc)),
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.4, 1.4)),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
        transforms.Grayscale(3),
        transforms.Lambda(lambda x: random_erase(np.array(x, dtype=np.uint8))),
        transforms.ToTensor()
    ])


def eval_data_transformations(input_shape, crop_perc=0.5):
    return transforms.Compose([
        transforms.Lambda(lambda x: crop_upper_part(
            np.array(x, dtype=np.uint8), crop_perc)),
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor()
    ])


def data_transformations(input_shape, crop_perc=0.5):
    train_trans = train_data_transformations(input_shape, crop_perc)
    eval_trans = eval_data_transformations(input_shape, crop_perc)
    return train_trans, eval_trans


def to_gpu(tensor, use_gpu=True):
    if use_gpu:
        tensor = tensor.cuda(0)
    return tensor


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def to_onehot(labels, num_classes, use_gpu=True):
    """
    Converts int category labels to one-hot representation.

    :param labels: Torch Tensor of int labels representing classes
    :param num_classes: Total number of classes
    :param use_gpu: Use GPU or not flag
    :return: One-hot torch tensor
    """
    one_hot = to_gpu(torch.zeros(labels.size(0), num_classes), use_gpu)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1)


def train_epoch(loader, model, model_criterion, center_criterion,
                model_optimizer, center_optimizer, use_gpu=True):
    batch_count = len(loader)
    total_correct = 0.0

    model.train()

    model_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for i, (input, target) in enumerate(loader):
        input_var, target_var = to_gpu(input, use_gpu), to_gpu(target, use_gpu)
        sample_count = len(target_var)

        logit, features = model(input=input_var, target=target_var)
        y_pred = logit.max(1)[1]

        one_hot_target = to_onehot(target_var, num_classes=model.num_classes, use_gpu=use_gpu)

        model_loss = model_criterion(input=logit, target=one_hot_target)
        center_loss = center_criterion(features, target_var)
        loss = model_loss + center_loss

        model_optimizer.zero_grad()
        center_optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=10)
        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_criterion.loss_weight)

        model_optimizer.step()
        center_optimizer.step()

        num_correct = float(y_pred.eq(target_var).long().sum().item())
        total_correct += num_correct

        avg_acc = float(total_correct) / (float(i + 1) * sample_count)
        total_loss_meter.update(loss.item(), sample_count)
        model_loss_meter.update(model_loss.item(), sample_count)
        center_loss_meter.update(center_loss.item(), sample_count)

        print(
            'Batch {}/{} | loss {:.6f} model_loss {:.6f} center_loss {:.6f} | accuracy = {:.6f}'
                .format(i + 1, batch_count, total_loss_meter.avg, model_loss_meter.avg,
                        center_loss_meter.avg, avg_acc),
            end="\r", flush=True)

    num_samples = float(len(loader.dataset))
    accuracy = total_correct / num_samples

    print("\n")
    return {
        'total_loss': total_loss_meter.avg,
        'model_loss': model_loss_meter.avg,
        'center_loss': center_loss_meter.avg,
        'accuracy': accuracy,
    }


def evaluate(loader, model, model_criterion,
             center_criterion, use_gpu=True):
    batch_count = len(loader)
    total_correct = 0.0

    model.eval()

    model_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    predictions, targets = [], []

    for i, (input, target) in enumerate(loader):
        input_var, target_var = to_gpu(input, use_gpu), to_gpu(target, use_gpu)
        sample_count = len(target_var)

        print('Eval batch {}/{}'
              .format(i + 1, batch_count),
              end="\r", flush=True)
        logit, features = model(input_var)
        y_pred = logit.max(1)[1]

        one_hot_target = to_onehot(target_var, num_classes=model.num_classes, use_gpu=use_gpu)

        model_loss = model_criterion(input=logit, target=one_hot_target)
        center_loss = center_criterion(features, target_var)
        loss = model_loss + center_loss

        for prediction in y_pred.cpu().data.numpy():
            predictions.append(prediction)
        targets.extend(target_var.cpu().data.numpy())

        total_correct += float(y_pred.eq(target_var).long().sum().item())
        total_loss_meter.update(loss.item(), sample_count)
        model_loss_meter.update(model_loss.item(), sample_count)
        center_loss_meter.update(center_loss.item(), sample_count)

    num_samples = float(len(loader.dataset))

    predictions = np.array(predictions)
    targets = np.array(targets).flatten()

    accuracy = total_correct / num_samples
    f1 = f1_score(targets, predictions, average='macro')
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')

    return {
        'total_loss': total_loss_meter.avg,
        'model_loss': model_loss_meter.avg,
        'center_loss': center_loss_meter.avg,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
