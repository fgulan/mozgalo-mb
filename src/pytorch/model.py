from collections import OrderedDict
from torch import nn
from torch.nn import init

import pretrainedmodels
from lsoftmax import LSoftmaxLinear

class LModel(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

        model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        self.model = model
        self.net = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Sequential(OrderedDict([
            ('av0', nn.AvgPool2d(10)),
            # ('fc0', nn.Linear(in_features=2048, out_features=256)),
            # ('fc1', nn.Linear(in_features=256, out_features=10))
            ('fc0_bn', nn.BatchNorm1d(2048))
        ]))

        self.lsoftmax_linear = LSoftmaxLinear(input_dim=2048, output_dim=25, margin=margin)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, input, target=None):
        conv_output = self.net(input)
        batch_size = conv_output.size(0)
        fc_output = self.fc(conv_output)
        logit = self.lsoftmax_linear(input=fc_output.view(batch_size, -1), target=target)
        return logit
