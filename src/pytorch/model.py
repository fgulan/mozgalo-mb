import torch.nn.functional as F
from squeezenet import squeezenet1_1
from torch import nn


class SqueezeModel(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        """
        SqueezeNet 1.1 Model used for training.

        :param num_classes: Number classes
        :param fine_tune: If true then all layers are trained. Otherwise only the top
        layers are trained and the main network layers are locked
        """
        super().__init__()

        self.features = squeezenet1_1(pretrained=True).features
        self.num_classes = num_classes
        self.num_features = 512
        logits = nn.Linear(self.num_features, self.num_classes)
        self.classificator = nn.Sequential(logits)

        for param in self.features.parameters():
            param.requires_grad = fine_tune

    def forward(self, input, target=None):
        conv_output = self.features(input)
        avg_kernel_size = conv_output[-1].shape[-2:]
        global_pooling = F.avg_pool2d(conv_output, avg_kernel_size)
        batch_size = conv_output.size(0)
        features = global_pooling.view(batch_size, -1)

        return self.classificator(features), features


class SqueezeModelSoftmax(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        """
        SqueezeNet 1.1 Model used for the test set evaluation.

        :param num_classes: Number classes
        :param fine_tune: If true then all layers are trained. Otherwise only the top
        layers are trained and the main network layers are locked
        """
        super().__init__()

        self.features = squeezenet1_1(pretrained=True).features
        self.num_classes = num_classes
        self.num_features = 512
        logits = nn.Linear(self.num_features, self.num_classes)
        self.classificator = nn.Sequential(logits)

        for param in self.features.parameters():
            param.requires_grad = fine_tune

    def forward(self, input, target=None):
        conv_output = self.features(input)
        avg_kernel_size = conv_output[-1].shape[-2:]
        global_pooling = F.avg_pool2d(conv_output, avg_kernel_size)
        batch_size = conv_output.size(0)
        features = global_pooling.view(batch_size, -1)
        logits = self.classificator(features)
        return F.softmax(logits, dim=1)
