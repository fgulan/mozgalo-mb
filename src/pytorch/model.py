import pretrainedmodels
import torch.nn.functional as F
from torch import nn
from squeezenet import squeezenet1_1


class XCeptionModel(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        """

        :param margin:
        :param num_classes: Number classes
        :param fine_tune: If true then all layers are trained. Otherwise only the top
        layers are trained and the main network layers are locked
        """
        super().__init__()
        self.model = pretrainedmodels.__dict__['xception'](num_classes=1000,
                                                           pretrained='imagenet')
        self.net = nn.Sequential(*list(self.model.children())[:-1])

        if not fine_tune:
            print("Base model layers will NOT be trained")
            for param in self.net.parameters():
                param.requires_grad = False

        self.logits = nn.Linear(2048, num_classes)

    def forward(self, input, target=None):
        conv_output = self.net(input)

        avg_kernel_size = conv_output[-1].shape[-2]
        global_pooling = F.avg_pool2d(conv_output, avg_kernel_size)

        batch_size = conv_output.size(0)
        return self.logits(global_pooling.view(batch_size, -1))


class SqueezeModel(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        """

        :param margin:
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
        global_pooling = F.max_pool2d(conv_output, avg_kernel_size, stride=1)
        batch_size = conv_output.size(0)
        features = global_pooling.view(batch_size, -1)
        return self.classificator(features), features
