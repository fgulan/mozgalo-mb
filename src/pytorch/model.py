import pretrainedmodels
from torch import nn
import torch.nn.functional as F
from pytorch.lsoftmax import LSoftmaxLinear


class LModel(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

        model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        self.model = model
        self.net = nn.Sequential(*list(model.children())[:-1])

        self.lsoftmax_linear = LSoftmaxLinear(input_dim=2048, output_dim=25, margin=margin)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, input, target=None):
        conv_output = self.net(input)
        avg_kernel_size = conv_output[-1].shape[-2]
        batch_size = conv_output.size(0)
        fc_output = F.avg_pool2d(conv_output, avg_kernel_size)
        logit = self.lsoftmax_linear(input=fc_output.view(batch_size, -1), target=target)
        return logit
