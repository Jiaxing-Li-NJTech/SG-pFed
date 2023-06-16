import copy

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
import torchsummary
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import identity, square, safe_log
from braindecode.torch_ext.util import np_to_var


class convnet(nn.Module):
    def __init__(self,class_num=4, eeg_channels=22, input_time_length=875, batch_norm=True, final_conv_length='auto'):
        super(convnet, self).__init__()
        self.class_num = class_num
        self.eeg_channels = eeg_channels
        self.input_time_length = input_time_length
        self.batch_norm = batch_norm
        self.final_conv_length = final_conv_length

        self.features = nn.Sequential()
        # self.features.add_module('dimshuffle', Expression(_transpose_time_to_spat))
        self.features.add_module('conv_time', nn.Conv2d(1, 40, kernel_size=(1, 25), stride=(1, 1)))
        self.features.add_module('conv_spat', nn.Conv2d(40, 20, kernel_size=(eeg_channels, 1), stride=(1,1), bias=True))
        self.features.add_module('bnorm', nn.BatchNorm2d(20, momentum=0.1, affine=True,track_running_stats=True))
        self.features.add_module('conv_nonlin', Expression(square))
        self.features.add_module('pool', nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)))
        self.features.add_module('pool_nonlin', Expression(safe_log))
        # self.features.add_module('drop',nn.Dropout(p=0.38))

        if self.final_conv_length == 'auto':
            out = self.features(np_to_var(np.ones(
                (1, 1, self.eeg_channels, self.input_time_length),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.classifier = nn.Sequential()
        # self.classifier.add_module('conv_classifier',nn.Conv2d(40, self.class_num,( 1,self.final_conv_length), bias=True))
        self.classifier.add_module('fc1', nn.Linear(20 * self.final_conv_length, self.class_num, bias=True))
        # self.classifier.add_module('squeeze',  Expression(_squeeze_final_output))
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.features.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        init.constant_(self.features.conv_time.bias, 0)

        init.xavier_uniform_(self.features.conv_spat.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.features.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.features.bnorm.weight, 1)
            init.constant_(self.features.bnorm.bias, 0)
        init.xavier_uniform_(self.classifier.fc1.weight, gain=1)
        init.constant_(self.classifier.fc1.bias, 0)
        # init.xavier_uniform_(self.classifier.conv_classifier.weight, gain=1)
        # init.constant_(self.classifier.conv_classifier.bias, 0)

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.shape[0], -1)
        x = self.classifier(feature)
        return feature,x


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

if __name__ == '__main__':
    net = Shallow(class_num=2,eeg_channels=3,input_time_length=875)
    x = torch.rand(288,1,3,875)
    fea,out = net(x)
    # print(fea.shape,out.shape)
    for k in net.state_dict().keys():
        print(k)