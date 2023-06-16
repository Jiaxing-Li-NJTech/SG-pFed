import torch
from torch.autograd import Variable

from options import args_parser
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import random

args = args_parser()
checkpoint_path = os.path.join('Network/', 'epoch_0.pth')

def get_ACC(model,dataLoader,num):
    training = model.training
    model.eval()
    #初始化预测正确的个数=0
    cor_num = 0.0
    with torch.no_grad():
        for step,batch in enumerate(dataLoader):
            inputs,labels = batch
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda().long()
            _,pres = model(inputs)
            preds_labels = torch.argmax(pres,dim=1)
            cor_num += (preds_labels==labels).sum().item()

        acc = cor_num/num
    model.train(training)
    return  acc