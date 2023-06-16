import copy
import torch
from torch import nn


def Avgcenter(localcenter,fre):
    center = copy.deepcopy(localcenter[0])*fre[0]
    for i in range(1,len(localcenter)):
        center += localcenter[i]*fre[i]
    return center

