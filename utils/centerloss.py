import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



class center_loss(nn.Module):
    def __init__(self):
        super(center_loss, self).__init__()
        # self.center = torch.zeros((4,128))

    def forward(self, feature, label,center_vector,class_num):
        global f_c
        for j in range(class_num):
            index = torch.eq(label, j).int()
            f_c_0 = feature.mul(index) - center_vector[j]
            if j==0:
                f_c = torch.norm(f_c_0.mul(index), p=2, dim=1)
            else:
                f_c = f_c + torch.norm(f_c_0.mul(index), p=2, dim=1)
        loss = f_c.sum() / feature.shape[0]

        return loss

def Center_vector(feature,label,class_num):
    # sum_feature = torch.sum(feature, dim=0)
    global sum_feature, sum_index
    for i in range(class_num):
        index = torch.eq(label, i).int()
        index_i = torch.sum(index)
        feature_i = feature.mul(index)
        if i == 0:
            sum_feature = torch.sum(feature_i, dim=0).view(1, -1)
            sum_index = torch.sum(index_i).view(1, 1)
        else:
            sum_feature = torch.cat([sum_feature, torch.sum(feature_i, dim=0).view(1, feature.shape[1])], dim=0)
            sum_index = torch.cat([sum_index, torch.sum(index_i).view(1, 1)], dim=0)
    sum_index = sum_index+1
    center = sum_feature / sum_index
    # center = torch.div(sum_feature, sum_index, dim=0)
    return center

