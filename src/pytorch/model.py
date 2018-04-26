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

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True)
        )

        for param in self.features.parameters():
            param.requires_grad = fine_tune

    def forward(self, input, target=None):
        x = self.features(input)
        x = self.conv(x)
        
        avg_kernel_size = x[-1].shape[-2]
        global_pooling = F.avg_pool2d(x, avg_kernel_size)
        batch_size = x.size(0)

        return global_pooling.view(batch_size, self.num_classes)
        