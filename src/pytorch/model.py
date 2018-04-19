import pretrainedmodels
import torch.nn.functional as F
from torch import nn
from torchvision.models import squeezenet1_1


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

        self.model = squeezenet1_1(pretrained=True)
        self.input_size = (3, 299, 299)
        self.net = nn.Sequential(*list(self.model.children())[:-1])

        if not fine_tune:
            print("Base model layers will NOT be trained")
            for param in self.net.parameters():
                param.requires_grad = False

        self.logits = nn.Linear(512, num_classes)

    def forward(self, input, target=None):
        conv_output = self.net(input)

        avg_kernel_size = conv_output[-1].shape[-2]
        global_pooling = F.avg_pool2d(conv_output, avg_kernel_size)

        batch_size = conv_output.size(0)
        return self.logits(global_pooling.view(batch_size, -1))
